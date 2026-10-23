//! Continuous-batching scheduler policy (Q-028 scheduler side, Q-039 SLO admission).
//!
//! The scheduler decides, per engine step, which running sequences decode one
//! token and which prompts advance by a prefill chunk, under three limits:
//! `max_running` sequences, `max_step_tokens` tokens per step and the paged KV
//! pool from `gnostral-memory-plan` (with a free-block watermark for new prefills).
//! When a decode append finds no free block, the lowest-priority, most recently
//! started sequence is preempted by recomputation and requeued.
//!
//! Two policies:
//! - [`Policy::Fifo`]: arrival order. The default posture until paired-run
//!   evidence exists (Q-039 gate "default posture").
//! - [`Policy::Slo`]: class, then deadline, then arrival; a waiting request's
//!   class improves by one rank per `aging_us` of waiting, so nothing starves.
//!   With a [`CostModel`], a request whose TTFT deadline cannot be met is
//!   rejected at submission with the estimate in the rejection.
//!
//! Time is a logical microsecond clock supplied by the caller; nothing reads a
//! wall clock. [`simulate`] drives the scheduler against a cost model and returns
//! a per-request receipt with every sample.

use gnostral_memory_plan::kv::{KvLayout, KvManager, SeqId};
use gnostral_memory_plan::MemoryError;
use std::collections::BTreeMap;

pub mod receipt;
pub use receipt::{percentile_nearest_rank, ClassStats, Receipt, RequestRecord};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    Interactive,
    Standard,
    Background,
}

impl Class {
    fn rank(self) -> u64 {
        match self {
            Class::Interactive => 0,
            Class::Standard => 1,
            Class::Background => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub id: u64,
    pub class: Class,
    pub arrival_us: u64,
    pub prompt_tokens: u64,
    pub max_new_tokens: u64,
    /// TTFT deadline relative to arrival.
    pub ttft_deadline_us: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Fifo,
    Slo { aging_us: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub policy: Policy,
    pub max_running: usize,
    pub max_step_tokens: u64,
    pub kv_watermark_blocks: usize,
}

/// Linear step-cost model from a measured profile. Planning input, not a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostModel {
    pub prefill_us_per_token: u64,
    pub decode_base_us: u64,
    pub decode_us_per_seq: u64,
}

impl CostModel {
    pub fn step_us(&self, prefill_tokens: u64, decode_seqs: u64) -> u64 {
        if prefill_tokens == 0 && decode_seqs == 0 {
            return 0;
        }
        self.decode_base_us
            + self.decode_us_per_seq * decode_seqs
            + self.prefill_us_per_token * prefill_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    DeadlineUnmeetable {
        estimate_us: u64,
        deadline_us: u64,
    },
    /// The prompt plus its generation can never fit the KV pool.
    TooLargeForKv {
        blocks: u64,
        total: u64,
    },
    /// A deadline was given under the SLO policy but no cost model is known.
    CostModelUnknown,
    DuplicateId,
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StepBatch {
    /// `(request id, prompt tokens computed this step)`.
    pub prefill: Vec<(u64, u64)>,
    /// Request ids decoding one token this step.
    pub decode: Vec<u64>,
    /// Requests preempted (recompute) while forming this step.
    pub preempted: Vec<u64>,
}

impl StepBatch {
    pub fn prefill_tokens(&self) -> u64 {
        self.prefill.iter().map(|&(_, t)| t).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

#[derive(Debug, Clone)]
struct Entry {
    req: Request,
    prefilled: u64,
    generated: u64,
    started_seq: u64,
    first_scheduled_us: Option<u64>,
    first_token_us: Option<u64>,
    preemptions: u32,
}

impl Entry {
    fn decoding(&self) -> bool {
        self.prefilled == self.req.prompt_tokens
    }
    fn context(&self) -> u64 {
        self.prefilled + self.generated
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Rejected(Rejection),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub request: Request,
    pub outcome: Outcome,
    pub first_scheduled_us: Option<u64>,
    pub first_token_us: Option<u64>,
    pub finish_us: u64,
    pub generated: u64,
    pub preemptions: u32,
}

#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    cost: Option<CostModel>,
    kv: KvManager,
    waiting: Vec<Entry>,
    running: Vec<Entry>,
    finished: Vec<Finished>,
    start_counter: u64,
    now_us: u64,
}

impl Scheduler {
    pub fn new(
        config: SchedulerConfig,
        cost: Option<CostModel>,
        layout: KvLayout,
        kv_blocks: u32,
    ) -> Result<Self, MemoryError> {
        Ok(Self {
            config,
            cost,
            kv: KvManager::new(layout, kv_blocks)?,
            waiting: Vec::new(),
            running: Vec::new(),
            finished: Vec::new(),
            start_counter: 0,
            now_us: 0,
        })
    }

    pub fn now_us(&self) -> u64 {
        self.now_us
    }
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
    pub fn finished(&self) -> &[Finished] {
        &self.finished
    }
    pub fn kv(&self) -> &KvManager {
        &self.kv
    }

    fn known(&self, id: u64) -> bool {
        self.waiting
            .iter()
            .chain(&self.running)
            .any(|e| e.req.id == id)
            || self.finished.iter().any(|f| f.request.id == id)
    }

    fn reject(&mut self, req: Request, rejection: Rejection) -> Rejection {
        self.finished.push(Finished {
            request: req,
            outcome: Outcome::Rejected(rejection),
            first_scheduled_us: None,
            first_token_us: None,
            finish_us: self.now_us,
            generated: 0,
            preemptions: 0,
        });
        rejection
    }

    /// Admission control at arrival.
    pub fn submit(&mut self, req: Request) -> Result<(), Rejection> {
        if self.known(req.id) {
            return Err(Rejection::DuplicateId);
        }
        if req.prompt_tokens == 0 || req.max_new_tokens == 0 {
            return Err(self.reject(req, Rejection::Empty));
        }
        let layout = self.kv.layout();
        let blocks = layout
            .blocks_for(req.prompt_tokens + req.max_new_tokens)
            .max(layout.blocks_for(req.prompt_tokens) + self.config.kv_watermark_blocks as u64);
        let total = self.kv.total_blocks() as u64;
        if blocks > total {
            return Err(self.reject(req, Rejection::TooLargeForKv { blocks, total }));
        }
        if let (Policy::Slo { .. }, Some(deadline_us)) = (self.config.policy, req.ttft_deadline_us)
        {
            let Some(cost) = self.cost else {
                return Err(self.reject(req, Rejection::CostModelUnknown));
            };
            let estimate_us = self.estimate_ttft_us(&req, &cost);
            if estimate_us > deadline_us {
                return Err(self.reject(
                    req,
                    Rejection::DeadlineUnmeetable {
                        estimate_us,
                        deadline_us,
                    },
                ));
            }
        }
        self.waiting.push(Entry {
            req,
            prefilled: 0,
            generated: 0,
            started_seq: 0,
            first_scheduled_us: None,
            first_token_us: None,
            preemptions: 0,
        });
        Ok(())
    }

    /// Prefill work ahead of `req` (same or better class) plus its own prompt,
    /// spread over steps that also carry the running decodes.
    fn estimate_ttft_us(&self, req: &Request, cost: &CostModel) -> u64 {
        let ahead: u64 = self
            .waiting
            .iter()
            .chain(&self.running)
            .filter(|e| e.req.class.rank() <= req.class.rank())
            .map(|e| e.req.prompt_tokens - e.prefilled)
            .sum();
        let tokens = ahead + req.prompt_tokens;
        let steps = tokens.div_ceil(self.config.max_step_tokens.max(1));
        let decoding = self.running.iter().filter(|e| e.decoding()).count() as u64;
        steps * cost.step_us(0, decoding) + tokens * cost.prefill_us_per_token
    }

    fn priority_key(&self, e: &Entry) -> (u64, u64, u64, u64) {
        match self.config.policy {
            Policy::Fifo => (0, 0, e.req.arrival_us, e.req.id),
            Policy::Slo { aging_us } => {
                let waited = self.now_us.saturating_sub(e.req.arrival_us);
                let boost = waited.checked_div(aging_us).unwrap_or(0);
                let rank = e.req.class.rank().saturating_sub(boost);
                let deadline = e
                    .req
                    .ttft_deadline_us
                    .map_or(u64::MAX, |d| e.req.arrival_us.saturating_add(d));
                (rank, deadline, e.req.arrival_us, e.req.id)
            }
        }
    }

    /// Lowest priority, most recently started running entry.
    fn victim(&self) -> Option<usize> {
        (0..self.running.len()).max_by_key(|&i| {
            let e = &self.running[i];
            (self.priority_key(e), e.started_seq)
        })
    }

    fn preempt(&mut self, index: usize, batch: &mut StepBatch) {
        let mut e = self.running.remove(index);
        self.kv
            .free(SeqId(e.req.id))
            .expect("running entries own KV");
        e.prefilled = 0;
        e.generated = 0;
        e.preemptions += 1;
        batch.preempted.push(e.req.id);
        batch.decode.retain(|&id| id != e.req.id);
        batch.prefill.retain(|&(id, _)| id != e.req.id);
        self.waiting.push(e);
    }

    /// Form the next step at logical time `now_us`.
    pub fn schedule(&mut self, now_us: u64) -> StepBatch {
        self.now_us = self.now_us.max(now_us);
        let mut batch = StepBatch::default();
        let mut budget = self.config.max_step_tokens;

        // Decode first: one token per decoding sequence, preempting on KV exhaustion.
        let mut order: Vec<u64> = {
            let mut r: Vec<&Entry> = self.running.iter().filter(|e| e.decoding()).collect();
            r.sort_by_key(|e| (self.priority_key(e), e.started_seq));
            r.iter().map(|e| e.req.id).collect()
        };
        while let Some(id) = order.first().copied() {
            order.remove(0);
            if budget == 0 {
                break;
            }
            loop {
                match self.kv.append(SeqId(id), 1) {
                    Ok(_) => {
                        batch.decode.push(id);
                        budget -= 1;
                        break;
                    }
                    Err(MemoryError::NoBlocks { .. }) => {
                        let v = self.victim().expect("running set is non-empty");
                        let victim_id = self.running[v].req.id;
                        self.preempt(v, &mut batch);
                        order.retain(|&x| x != victim_id);
                        if victim_id == id {
                            break;
                        }
                    }
                    Err(e) => panic!("kv append on a running sequence failed: {e:?}"),
                }
            }
        }

        // Continue partially prefilled sequences, then admit waiting ones.
        let mut partial: Vec<usize> = (0..self.running.len())
            .filter(|&i| !self.running[i].decoding())
            .collect();
        partial.sort_by_key(|&i| {
            (
                self.priority_key(&self.running[i]),
                self.running[i].started_seq,
            )
        });
        for i in partial {
            if budget == 0 {
                break;
            }
            let e = &self.running[i];
            let chunk = (e.req.prompt_tokens - e.prefilled).min(budget);
            budget -= chunk;
            batch.prefill.push((e.req.id, chunk));
        }

        let mut candidates: Vec<usize> = (0..self.waiting.len()).collect();
        candidates.sort_by_key(|&i| self.priority_key(&self.waiting[i]));
        let mut admitted = Vec::new();
        for i in candidates {
            if budget == 0 || self.running.len() + admitted.len() >= self.config.max_running {
                break;
            }
            let e = &self.waiting[i];
            let need = self.kv.layout().blocks_for(e.req.prompt_tokens) as usize;
            if self.kv.free_blocks() < need + self.config.kv_watermark_blocks {
                break; // head-of-line by design: lower-priority work may not jump the queue on memory
            }
            self.kv
                .allocate(SeqId(e.req.id), e.req.prompt_tokens)
                .expect("checked free blocks");
            let chunk = e.req.prompt_tokens.min(budget);
            budget -= chunk;
            batch.prefill.push((e.req.id, chunk));
            admitted.push(i);
        }
        admitted.sort_unstable();
        for i in admitted.into_iter().rev() {
            let mut e = self.waiting.remove(i);
            self.start_counter += 1;
            e.started_seq = self.start_counter;
            e.first_scheduled_us.get_or_insert(self.now_us);
            self.running.push(e);
        }
        batch
    }

    /// Account a completed step that ended at `end_us`.
    pub fn complete(&mut self, batch: &StepBatch, end_us: u64) {
        self.now_us = self.now_us.max(end_us);
        for &(id, tokens) in &batch.prefill {
            if let Some(e) = self.running.iter_mut().find(|e| e.req.id == id) {
                e.prefilled += tokens;
                if e.decoding() && e.generated == 0 {
                    // The step that finishes the prompt samples the first token.
                    e.generated = 1;
                    e.first_token_us.get_or_insert(end_us);
                }
            }
        }
        for &id in &batch.decode {
            if let Some(e) = self.running.iter_mut().find(|e| e.req.id == id) {
                e.generated += 1;
            }
        }
        let mut i = 0;
        while i < self.running.len() {
            let e = &self.running[i];
            if e.decoding() && e.generated >= e.req.max_new_tokens {
                let e = self.running.remove(i);
                debug_assert!(e.context() >= e.req.prompt_tokens);
                self.kv
                    .free(SeqId(e.req.id))
                    .expect("running entries own KV");
                self.finished.push(Finished {
                    request: e.req,
                    outcome: Outcome::Completed,
                    first_scheduled_us: e.first_scheduled_us,
                    first_token_us: e.first_token_us,
                    finish_us: end_us,
                    generated: e.generated,
                    preemptions: e.preemptions,
                });
            } else {
                i += 1;
            }
        }
    }
}

/// Drive a scheduler to completion against `cost`; returns the receipt.
///
/// Arrivals are submitted when the logical clock reaches them; an idle scheduler
/// jumps to the next arrival. `max_steps` bounds the run (a starvation guard).
pub fn simulate(
    config: SchedulerConfig,
    cost: CostModel,
    layout: KvLayout,
    kv_blocks: u32,
    requests: &[Request],
    max_steps: u64,
) -> Result<Receipt, MemoryError> {
    let mut s = Scheduler::new(config, Some(cost), layout, kv_blocks)?;
    let mut pending: BTreeMap<(u64, u64), Request> = requests
        .iter()
        .map(|r| ((r.arrival_us, r.id), *r))
        .collect();
    let mut now = 0u64;
    let mut steps = 0u64;
    loop {
        while let Some((&(arrival, id), _)) = pending.iter().next() {
            if arrival > now {
                break;
            }
            let r = pending.remove(&(arrival, id)).expect("present");
            let _ = s.submit(r);
        }
        if s.is_idle() {
            match pending.keys().next() {
                Some(&(arrival, _)) => {
                    now = arrival;
                    continue;
                }
                None => break,
            }
        }
        let batch = s.schedule(now);
        let elapsed = cost
            .step_us(batch.prefill_tokens(), batch.decode.len() as u64)
            .max(1);
        now += elapsed;
        s.complete(&batch, now);
        steps += 1;
        if steps >= max_steps {
            break;
        }
    }
    Ok(Receipt::from_finished(
        s.finished(),
        requests.len(),
        now,
        steps,
    ))
}
