//! Q-046 gates: speculation never changes greedy output; the policy turns itself off when useless.

use gnostral_speculative::policy::{
    choose_k, expected_tokens, keep_draft, speedup, AcceptanceTracker,
};
use gnostral_speculative::reference::{generate, greedy, Draft, Target, Token};

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// Deterministic "model": the next token is a hash of the last few tokens.
struct Hashed {
    seed: u64,
    vocab: u64,
    window: usize,
}

impl Hashed {
    fn next(&self, context: &[Token]) -> Token {
        let start = context.len().saturating_sub(self.window);
        let mut h = self.seed;
        for &t in &context[start..] {
            h = mix(h ^ u64::from(t));
        }
        (h % self.vocab) as Token
    }
}

impl Target for Hashed {
    fn verify(&mut self, context: &[Token], drafted: &[Token]) -> Vec<Token> {
        let mut ctx = context.to_vec();
        let mut out = Vec::with_capacity(drafted.len() + 1);
        for &d in drafted {
            out.push(self.next(&ctx));
            ctx.push(d);
        }
        out.push(self.next(&ctx));
        out
    }
}

/// Draft that copies the target with probability `p` per position, else guesses.
struct Noisy {
    target: Hashed,
    p_per_mille: u64,
    salt: u64,
}

impl Draft for Noisy {
    fn propose(&mut self, context: &[Token], k: usize) -> Vec<Token> {
        let mut ctx = context.to_vec();
        let mut out = Vec::new();
        for i in 0..k {
            self.salt = mix(self.salt + i as u64);
            let t = if self.salt % 1000 < self.p_per_mille {
                self.target.next(&ctx)
            } else {
                (self.salt >> 20) as Token % 7
            };
            out.push(t);
            ctx.push(t);
        }
        out
    }
}

#[test]
fn speculation_is_token_identical_to_greedy_for_any_draft() {
    for seed in 0..40u64 {
        let target = || Hashed {
            seed,
            vocab: 50 + seed,
            window: 3,
        };
        let prompt: Vec<Token> = (0..(seed % 5) as Token).collect();
        let want = greedy(&mut target(), &prompt, 64);
        for p in [0, 300, 800, 1000] {
            for k in [1usize, 2, 4, 7] {
                let mut draft = Noisy {
                    target: target(),
                    p_per_mille: p,
                    salt: seed,
                };
                let (got, stats) = generate(&mut target(), &mut draft, &prompt, 64, |_| k);
                assert_eq!(got, want, "seed {seed} p {p} k {k}");
                assert_eq!(stats.emitted, 64);
                if p == 1000 {
                    assert_eq!(
                        stats.accepted, stats.drafted,
                        "perfect draft fully accepted"
                    );
                    assert!(stats.target_passes <= 64usize.div_ceil(k + 1) as u64 + 1);
                }
            }
        }
    }
}

#[test]
fn adaptive_k_keeps_greedy_output() {
    let target = || Hashed {
        seed: 9,
        vocab: 80,
        window: 2,
    };
    let want = greedy(&mut target(), &[1, 2, 3], 100);
    let mut draft = Noisy {
        target: target(),
        p_per_mille: 700,
        salt: 1,
    };
    let mut tracker = AcceptanceTracker::new(0.3);
    let mut last = gnostral_speculative::reference::Stats::default();
    let (got, stats) = generate(&mut target(), &mut draft, &[1, 2, 3], 100, |s| {
        tracker.observe(
            (s.drafted - last.drafted) as usize,
            (s.accepted - last.accepted) as usize,
        );
        last = *s;
        choose_k(tracker.estimate(), 0.1, 6, 2).k
    });
    assert_eq!(got, want);
    assert!(stats.tokens_per_pass().unwrap() > 1.5, "{stats:?}");
}

#[test]
fn cost_model_matches_closed_forms() {
    assert_eq!(expected_tokens(0.0, 4), 1.0);
    assert_eq!(expected_tokens(1.0, 4), 5.0);
    assert!((expected_tokens(0.5, 2) - 1.75).abs() < 1e-12);
    assert!((speedup(0.5, 2, 0.25) - 1.75 / 1.5).abs() < 1e-12);
}

#[test]
fn policy_turns_speculation_off_when_it_does_not_pay() {
    // Cheap, accurate draft: speculate deep.
    let good = choose_k(Some(0.9), 0.05, 8, 2);
    assert!(good.k >= 5 && good.speedup > 3.0, "{good:?}");
    // Expensive or inaccurate draft: off.
    assert_eq!(choose_k(Some(0.2), 0.5, 8, 2).k, 0);
    assert_eq!(choose_k(Some(0.0), 0.0, 8, 2).k, 0);
    // UNKNOWN acceptance probes.
    assert_eq!(choose_k(None, 0.1, 8, 2).k, 2);
    assert_eq!(keep_draft(None, 0.1, 8, 1.05), None);
    assert_eq!(keep_draft(Some(0.2), 0.5, 8, 1.05), Some(false));
    assert_eq!(keep_draft(Some(0.8), 0.1, 8, 1.05), Some(true));
}

#[test]
fn tracker_ignores_empty_passes_and_converges() {
    let mut t = AcceptanceTracker::new(0.5);
    t.observe(0, 0);
    assert_eq!(t.estimate(), None);
    t.observe(4, 4);
    assert_eq!(t.estimate(), Some(1.0));
    for _ in 0..40 {
        t.observe(4, 2);
    }
    assert!((t.estimate().unwrap() - 0.5).abs() < 1e-9);
}
