//! Per-request fairness receipt with every sample kept.

use crate::{Class, Finished, Outcome, Rejection};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestRecord {
    pub id: u64,
    pub class: Class,
    pub outcome: Outcome,
    pub queue_wait_us: Option<u64>,
    pub ttft_us: Option<u64>,
    /// Mean time per output token after the first, in microseconds; `None` with fewer than two tokens.
    pub tpot_us: Option<u64>,
    pub generated: u64,
    pub preemptions: u32,
    pub deadline_met: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClassStats {
    pub completed: usize,
    pub rejected: usize,
    pub ttft_samples_us: Vec<u64>,
    pub tpot_samples_us: Vec<u64>,
    pub wait_samples_us: Vec<u64>,
    pub ttft_p50_p95_p99: Option<(u64, u64, u64)>,
    pub tpot_p50_p95_p99: Option<(u64, u64, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub submitted: usize,
    pub records: Vec<RequestRecord>,
    /// Requests neither finished nor rejected when the run stopped.
    pub unfinished: usize,
    pub makespan_us: u64,
    pub steps: u64,
    pub generated_tokens: u64,
    pub per_class: BTreeMap<Class, ClassStats>,
    pub rejections: BTreeMap<String, usize>,
}

/// Nearest-rank percentile (`ceil(p/100 * n)`-th smallest); `None` for no samples.
pub fn percentile_nearest_rank(samples: &[u64], p: u64) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (p * sorted.len() as u64).div_ceil(100).max(1) as usize;
    Some(sorted[rank.min(sorted.len()) - 1])
}

fn tails(samples: &[u64]) -> Option<(u64, u64, u64)> {
    Some((
        percentile_nearest_rank(samples, 50)?,
        percentile_nearest_rank(samples, 95)?,
        percentile_nearest_rank(samples, 99)?,
    ))
}

impl Receipt {
    pub fn from_finished(
        finished: &[Finished],
        submitted: usize,
        makespan_us: u64,
        steps: u64,
    ) -> Self {
        let mut records: Vec<RequestRecord> = finished
            .iter()
            .map(|f| {
                let r = &f.request;
                let ttft = f.first_token_us.map(|t| t - r.arrival_us);
                let tpot = match (f.first_token_us, f.generated) {
                    (Some(first), n) if n >= 2 => Some((f.finish_us - first) / (n - 1)),
                    _ => None,
                };
                RequestRecord {
                    id: r.id,
                    class: r.class,
                    outcome: f.outcome.clone(),
                    queue_wait_us: f.first_scheduled_us.map(|t| t - r.arrival_us),
                    ttft_us: ttft,
                    tpot_us: tpot,
                    generated: f.generated,
                    preemptions: f.preemptions,
                    deadline_met: r.ttft_deadline_us.zip(ttft).map(|(d, t)| t <= d),
                }
            })
            .collect();
        records.sort_by_key(|r| r.id);
        let mut per_class: BTreeMap<Class, ClassStats> = BTreeMap::new();
        let mut rejections: BTreeMap<String, usize> = BTreeMap::new();
        for r in &records {
            let c = per_class.entry(r.class).or_default();
            match &r.outcome {
                Outcome::Completed => {
                    c.completed += 1;
                    c.ttft_samples_us.extend(r.ttft_us);
                    c.tpot_samples_us.extend(r.tpot_us);
                    c.wait_samples_us.extend(r.queue_wait_us);
                }
                Outcome::Rejected(why) => {
                    c.rejected += 1;
                    *rejections
                        .entry(rejection_name(why).to_string())
                        .or_default() += 1;
                }
            }
        }
        for c in per_class.values_mut() {
            c.ttft_p50_p95_p99 = tails(&c.ttft_samples_us);
            c.tpot_p50_p95_p99 = tails(&c.tpot_samples_us);
        }
        Self {
            submitted,
            unfinished: submitted - records.len(),
            generated_tokens: records.iter().map(|r| r.generated).sum(),
            records,
            makespan_us,
            steps,
            per_class,
            rejections,
        }
    }

    /// Aggregate output tokens per second over the makespan.
    pub fn throughput_tok_s(&self) -> f64 {
        if self.makespan_us == 0 {
            return 0.0;
        }
        self.generated_tokens as f64 * 1e6 / self.makespan_us as f64
    }
}

fn rejection_name(r: &Rejection) -> &'static str {
    match r {
        Rejection::DeadlineUnmeetable { .. } => "deadline_unmeetable",
        Rejection::TooLargeForKv { .. } => "too_large_for_kv",
        Rejection::CostModelUnknown => "cost_model_unknown",
        Rejection::DuplicateId => "duplicate_id",
        Rejection::Empty => "empty",
    }
}
