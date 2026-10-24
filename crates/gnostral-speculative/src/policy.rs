//! How many tokens to draft, and whether the draft is worth keeping.
//!
//! With per-token acceptance `a` (draft and target agree, independently per
//! position) and `k` drafted tokens, one target pass emits on average
//! `E(k) = (1 - a^(k+1)) / (1 - a)` tokens (`k + 1` when `a = 1`). The pass costs
//! one target step plus `k` draft steps, a draft step costing `c` target steps.
//! The expected speedup over plain greedy is `E(k) / (1 + k c)`. The policy picks
//! the `k` in `0..=k_max` with the best speedup; `k = 0` (speedup 1) wins when no
//! draft length pays for itself.
//!
//! This is a model for choosing, not a measurement. The real `a` comes from an
//! [`AcceptanceTracker`] fed by the engine, and the real `c` from Q-042 timers.

/// Exponentially weighted per-token acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceTracker {
    /// Weight of the newest observation, in `(0, 1]`.
    pub alpha: f64,
    estimate: Option<f64>,
}

impl AcceptanceTracker {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self {
            alpha,
            estimate: None,
        }
    }

    /// One verification: `accepted` of `drafted` tokens kept. A pass without drafts says nothing.
    pub fn observe(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = accepted.min(drafted) as f64 / drafted as f64;
        self.estimate = Some(match self.estimate {
            None => rate,
            Some(e) => e + self.alpha * (rate - e),
        });
    }

    /// `None` is `UNKNOWN`: nothing observed yet.
    pub fn estimate(&self) -> Option<f64> {
        self.estimate
    }
}

/// Expected emitted tokens per target pass.
pub fn expected_tokens(acceptance: f64, k: usize) -> f64 {
    let a = acceptance.clamp(0.0, 1.0);
    if a >= 1.0 {
        return (k + 1) as f64;
    }
    (1.0 - a.powi(k as i32 + 1)) / (1.0 - a)
}

/// Expected speedup over plain greedy for draft length `k` and draft/target step cost ratio `c`.
pub fn speedup(acceptance: f64, k: usize, draft_cost: f64) -> f64 {
    expected_tokens(acceptance, k) / (1.0 + k as f64 * draft_cost.max(0.0))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub k: usize,
    pub speedup: f64,
}

/// Best `k` in `0..=k_max`; ties go to the smaller `k`. `UNKNOWN` acceptance probes with `probe_k`.
pub fn choose_k(acceptance: Option<f64>, draft_cost: f64, k_max: usize, probe_k: usize) -> Choice {
    let Some(a) = acceptance else {
        return Choice {
            k: probe_k.min(k_max),
            speedup: f64::NAN,
        };
    };
    let mut best = Choice { k: 0, speedup: 1.0 };
    for k in 1..=k_max {
        let s = speedup(a, k, draft_cost);
        if s > best.speedup + 1e-12 {
            best = Choice { k, speedup: s };
        }
    }
    best
}

/// Whether the draft model earns its memory: its best speedup must beat `min_gain`
/// (e.g. 1.05). When it does not, the draft can be parked by the co-hosting broker.
pub fn keep_draft(
    acceptance: Option<f64>,
    draft_cost: f64,
    k_max: usize,
    min_gain: f64,
) -> Option<bool> {
    let a = acceptance?;
    Some(choose_k(Some(a), draft_cost, k_max, 0).speedup >= min_gain)
}
