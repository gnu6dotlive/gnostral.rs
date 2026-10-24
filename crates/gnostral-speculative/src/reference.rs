//! Greedy speculation over abstract models.
//!
//! A [`Target`] answers one verification pass: for a context and `k` drafted
//! tokens it returns its greedy token at each of the `k + 1` positions. The loop
//! keeps the longest drafted prefix the target agrees with, then appends the
//! target's own token at the first disagreement (or after a full accept). Every
//! emitted token is therefore the target's greedy choice given the tokens before
//! it, so the output equals plain greedy decoding whatever the draft proposes.

pub type Token = u32;

/// The large model: greedy token after `context ++ drafted[..i]` for every `i` in `0..=k`.
pub trait Target {
    fn verify(&mut self, context: &[Token], drafted: &[Token]) -> Vec<Token>;
}

/// The small model: up to `k` proposed continuations of `context`.
pub trait Draft {
    fn propose(&mut self, context: &[Token], k: usize) -> Vec<Token>;
}

/// Counters for one generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Target verification passes (the expensive calls).
    pub target_passes: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub emitted: u64,
}

impl Stats {
    /// Accepted drafted tokens over drafted tokens; `None` before any draft.
    pub fn acceptance(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }
    /// Emitted tokens per target pass (1.0 without speculation).
    pub fn tokens_per_pass(&self) -> Option<f64> {
        (self.target_passes > 0).then(|| self.emitted as f64 / self.target_passes as f64)
    }
}

/// Generate `n` tokens after `prompt`; `k_for` chooses the draft length before every pass.
///
/// Panics if the target returns a number of tokens other than `drafted + 1`: that is a broken
/// target, not a speculation outcome.
pub fn generate(
    target: &mut impl Target,
    draft: &mut impl Draft,
    prompt: &[Token],
    n: usize,
    mut k_for: impl FnMut(&Stats) -> usize,
) -> (Vec<Token>, Stats) {
    let mut context = prompt.to_vec();
    let mut stats = Stats::default();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let remaining = n - out.len();
        let k = k_for(&stats).min(remaining - 1);
        let mut drafted = if k > 0 {
            draft.propose(&context, k)
        } else {
            Vec::new()
        };
        drafted.truncate(k);
        let verified = target.verify(&context, &drafted);
        assert_eq!(
            verified.len(),
            drafted.len() + 1,
            "target must answer k + 1 positions"
        );
        stats.target_passes += 1;
        stats.drafted += drafted.len() as u64;
        let agreed = drafted
            .iter()
            .zip(&verified)
            .take_while(|(d, v)| d == v)
            .count();
        stats.accepted += agreed as u64;
        for &t in &verified[..=agreed] {
            context.push(t);
            out.push(t);
        }
        stats.emitted = out.len() as u64;
    }
    (out, stats)
}

/// Plain greedy decoding through the same [`Target`] (one pass per token).
pub fn greedy(target: &mut impl Target, prompt: &[Token], n: usize) -> Vec<Token> {
    let mut context = prompt.to_vec();
    for _ in 0..n {
        let t = target.verify(&context, &[])[0];
        context.push(t);
    }
    context.split_off(prompt.len())
}
