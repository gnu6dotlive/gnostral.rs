//! Sharing one GPU between deadline-bound audio and throughput-bound text.
//!
//! Realtime streams (speech-to-text chunks, text-to-speech chunks) release a job
//! every `period_us` that must finish within that period. The language model
//! has an endless backlog of decode steps, each `step_us` long and not
//! preemptible: once a step is on the GPU, audio waits for it.
//!
//! [`admit`] applies the processor-demand test for non-preemptive
//! earliest-deadline scheduling with implicit deadlines: at every absolute
//! deadline `t` up to `max(max_period, blocking / (1 - U))`,
//! `sum_j floor(t / period_j) * cost_j + B(t) <= t`, where `B(t)` is the longest
//! job that cannot be interrupted and has a deadline after `t`: any language
//! model step, or an audio job of a longer period. It is sufficient for any
//! release phasing. It also reports the longest step the language model may use,
//! which is how the engine sizes decode batches while someone is talking.
//! [`simulate`] replays a schedule for a policy so the test and the policy can
//! be checked against each other.

/// Upper bound on deadlines examined by the admission test per stream set.
const MAX_CHECK_POINTS: u64 = 1_000_000;

/// A periodic deadline-bound stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stream {
    pub name: &'static str,
    pub period_us: u64,
    pub cost_us: u64,
    /// Release time of the first job.
    pub phase_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Admission {
    pub utilization: f64,
    /// Longest non-preemptible step that keeps every audio deadline under the test.
    pub max_step_us: u64,
    pub admitted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShareError {
    /// A stream with a zero period, or a cost larger than its period.
    InvalidStream(&'static str),
}

/// Sufficient admission test for `streams` sharing the GPU with steps of `step_us`.
pub fn admit(streams: &[Stream], step_us: u64) -> Result<Admission, ShareError> {
    for s in streams {
        if s.period_us == 0 || s.cost_us > s.period_us {
            return Err(ShareError::InvalidStream(s.name));
        }
    }
    let utilization: f64 = streams
        .iter()
        .map(|s| s.cost_us as f64 / s.period_us as f64)
        .sum();
    if streams.is_empty() {
        return Ok(Admission {
            utilization,
            max_step_us: u64::MAX,
            admitted: true,
        });
    }
    let max_period = streams.iter().map(|s| s.period_us).max().unwrap_or(0);
    let max_step_us = if feasible(streams, utilization, 0) {
        // Largest step that still passes; the test is monotone in the step.
        let (mut lo, mut hi) = (0u64, max_period);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if feasible(streams, utilization, mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    } else {
        0
    };
    Ok(Admission {
        utilization,
        max_step_us,
        admitted: feasible(streams, utilization, step_us),
    })
}

fn feasible(streams: &[Stream], utilization: f64, step_us: u64) -> bool {
    if utilization >= 1.0 {
        return false;
    }
    let max_period = streams.iter().map(|s| s.period_us).max().unwrap_or(0);
    let worst_block = streams
        .iter()
        .map(|s| s.cost_us)
        .max()
        .unwrap_or(0)
        .max(step_us);
    let horizon = (max_period as f64)
        .max(worst_block as f64 / (1.0 - utilization))
        .ceil() as u64;
    let min_period = streams.iter().map(|s| s.period_us).min().unwrap_or(1);
    // Utilisation so close to 1 that the check would be enormous: refuse rather than guess.
    if horizon / min_period > MAX_CHECK_POINTS {
        return false;
    }
    let mut deadlines: Vec<u64> = streams
        .iter()
        .flat_map(|s| (1..=horizon / s.period_us).map(move |k| k * s.period_us))
        .collect();
    deadlines.sort_unstable();
    deadlines.dedup();
    deadlines.into_iter().all(|t| {
        let demand: u64 = streams.iter().map(|s| (t / s.period_us) * s.cost_us).sum();
        let blocking = streams
            .iter()
            .filter(|s| s.period_us > t)
            .map(|s| s.cost_us)
            .max()
            .unwrap_or(0)
            .max(step_us);
        demand + blocking <= t
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Whoever was released first runs next, a language model step being released when the
    /// previous one ends (no notion of deadlines).
    Fifo,
    /// Earliest-deadline audio job first; language model steps only when no audio is ready.
    RealtimeFirst,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Outcome {
    pub audio_jobs: u64,
    pub deadline_misses: u64,
    pub worst_lateness_us: u64,
    pub lm_steps: u64,
    /// Fraction of the horizon spent on language model steps, in per mille.
    pub lm_share_per_mille: u64,
}

#[derive(Clone, Copy)]
struct Job {
    stream: usize,
    release: u64,
    deadline: u64,
}

/// Non-preemptive single-GPU schedule over `horizon_us`.
pub fn simulate(streams: &[Stream], step_us: u64, policy: Policy, horizon_us: u64) -> Outcome {
    let mut next_release: Vec<u64> = streams.iter().map(|s| s.phase_us).collect();
    let mut ready: Vec<Job> = Vec::new();
    let mut out = Outcome::default();
    let mut now = 0u64;
    let mut lm_release = 0u64;
    let mut lm_busy = 0u64;
    while now < horizon_us {
        for (i, s) in streams.iter().enumerate() {
            while next_release[i] <= now {
                ready.push(Job {
                    stream: i,
                    release: next_release[i],
                    deadline: next_release[i] + s.period_us,
                });
                next_release[i] += s.period_us;
            }
        }
        let audio = match policy {
            Policy::RealtimeFirst => ready
                .iter()
                .enumerate()
                .min_by_key(|(_, j)| (j.deadline, j.stream))
                .map(|(i, _)| i),
            Policy::Fifo => ready
                .iter()
                .enumerate()
                .filter(|(_, j)| step_us == 0 || j.release <= lm_release)
                .min_by_key(|(_, j)| (j.release, j.stream))
                .map(|(i, _)| i),
        };
        match audio {
            Some(i) => {
                let job = ready.swap_remove(i);
                now += streams[job.stream].cost_us;
                out.audio_jobs += 1;
                if now > job.deadline {
                    out.deadline_misses += 1;
                    out.worst_lateness_us = out.worst_lateness_us.max(now - job.deadline);
                }
            }
            None if step_us > 0 => {
                now += step_us;
                lm_busy += step_us;
                out.lm_steps += 1;
                lm_release = now;
            }
            None => {
                now = next_release
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(horizon_us)
                    .max(now + 1);
            }
        }
    }
    out.lm_share_per_mille = lm_busy * 1000 / now.max(1);
    out
}
