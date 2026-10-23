//! Q-027 dynamic residency: online histogram, explicit admission/eviction, leases and pins.
//!
//! The cache owns **metadata only**. The engine owns allocations, performs uploads,
//! and reports completion with the ticket it was given. Every decision is a pure
//! function of the observed route history and the configuration: no wall clock,
//! no hash iteration, ties broken by `ExpertKey`.
//!
//! Lifecycle of a slot:
//!
//! ```text
//! Vacant --admit--> Pending(ticket) --complete(ok)--> Resident --evict--> Vacant
//!                           \--complete(fail)--> Vacant
//! ```
//!
//! A `Resident` slot with an outstanding lease or pin is never chosen as a victim,
//! and a `Pending` slot can never be acquired: a miss is either admitted
//! (`Decision::Admit`) or refused with a typed reason; it is never served from a
//! stale slot.

use crate::{ExpertKey, ResidencyError};
use std::collections::BTreeMap;

/// Fixed-point unit for one hit in the decayed score.
pub const HIT_SCORE: u64 = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicConfig {
    /// Bytes the expert cache may hold (its share of the memory envelope).
    pub capacity_bytes: u64,
    /// Steps after which a score halves.
    pub half_life_steps: u64,
    /// Minimum decayed score before a miss is considered for admission.
    pub admit_min_score: u64,
    /// A victim is evicted only if `victim * hysteresis_num < candidate * hysteresis_den`.
    pub hysteresis_num: u64,
    pub hysteresis_den: u64,
    /// Upload bytes the engine may start per step (0 = unbounded).
    pub max_upload_bytes_per_step: u64,
}

impl DynamicConfig {
    pub fn validate(&self) -> Result<(), ResidencyError> {
        if self.half_life_steps == 0 || self.hysteresis_num == 0 || self.hysteresis_den == 0 {
            return Err(ResidencyError::InvalidGeometry);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Pending(u64),
    Resident,
}

#[derive(Debug, Clone)]
struct Entry {
    bytes: u64,
    score: u64,
    last_step: u64,
    hits: u64,
    state: Option<SlotState>,
    leases: u32,
    pins: u32,
}

/// Why a miss was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Score below `admit_min_score`.
    Cold,
    /// Larger than the whole cache.
    TooLarge,
    /// No unleased, unpinned victim set with a low enough score frees the space.
    NoVictim,
    /// The per-step upload budget is spent.
    UploadBudget,
}

/// One decision of [`DynamicCache::decide`], applied in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Free this slot's allocation (no lease or pin exists).
    Evict {
        key: ExpertKey,
        bytes: u64,
    },
    /// Allocate and upload; report with `complete_upload(ticket, ok)`.
    Admit {
        key: ExpertKey,
        bytes: u64,
        ticket: AdmitTicket,
    },
    Refuse {
        key: ExpertKey,
        reason: Refusal,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmitTicket {
    key: ExpertKey,
    nonce: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicLease {
    key: ExpertKey,
    nonce: u64,
}
impl DynamicLease {
    pub fn key(&self) -> ExpertKey {
        self.key
    }
}

/// Observed counters. `None` means `UNKNOWN`: the engine has not reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Telemetry {
    pub steps: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub refusals: BTreeMap<String, u64>,
    pub failed_uploads: u64,
    pub resident_slots: usize,
    pub planned_bytes: u64,
    /// Engine-measured device bytes for resident experts, if reported.
    pub measured_bytes: Option<u64>,
}

#[derive(Debug)]
pub struct DynamicCache {
    config: DynamicConfig,
    entries: BTreeMap<ExpertKey, Entry>,
    leases: BTreeMap<u64, ExpertKey>,
    step: u64,
    nonce: u64,
    used_bytes: u64,
    telemetry: Telemetry,
}

fn decayed(score: u64, last: u64, now: u64, half_life: u64) -> u64 {
    let halvings = (now - last) / half_life;
    if halvings >= 64 {
        0
    } else {
        score >> halvings
    }
}

impl DynamicCache {
    /// `sizes` declares every expert the cache may ever hold, with its padded byte size.
    pub fn new(config: DynamicConfig, sizes: &[(ExpertKey, u64)]) -> Result<Self, ResidencyError> {
        config.validate()?;
        let mut entries = BTreeMap::new();
        for &(key, bytes) in sizes {
            if bytes == 0 {
                return Err(ResidencyError::InvalidProfile);
            }
            let entry = Entry {
                bytes,
                score: 0,
                last_step: 0,
                hits: 0,
                state: None,
                leases: 0,
                pins: 0,
            };
            if entries.insert(key, entry).is_some() {
                return Err(ResidencyError::InvalidProfile);
            }
        }
        Ok(Self {
            config,
            entries,
            leases: BTreeMap::new(),
            step: 0,
            nonce: 0,
            used_bytes: 0,
            telemetry: Telemetry::default(),
        })
    }

    fn next_nonce(&mut self) -> Result<u64, ResidencyError> {
        self.nonce = self.nonce.checked_add(1).ok_or(ResidencyError::Overflow)?;
        Ok(self.nonce)
    }

    fn entry(&self, key: ExpertKey) -> Result<&Entry, ResidencyError> {
        self.entries.get(&key).ok_or(ResidencyError::InvalidProfile)
    }

    fn entry_mut(&mut self, key: ExpertKey) -> Result<&mut Entry, ResidencyError> {
        self.entries
            .get_mut(&key)
            .ok_or(ResidencyError::InvalidProfile)
    }

    pub fn score(&self, key: ExpertKey) -> Result<u64, ResidencyError> {
        let e = self.entry(key)?;
        Ok(decayed(
            e.score,
            e.last_step,
            self.step,
            self.config.half_life_steps,
        ))
    }

    pub fn is_resident(&self, key: ExpertKey) -> bool {
        matches!(
            self.entries.get(&key).and_then(|e| e.state),
            Some(SlotState::Resident)
        )
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    /// Record one forward step's routed experts (duplicates count once per occurrence).
    pub fn observe(&mut self, keys: &[ExpertKey]) -> Result<(), ResidencyError> {
        for &key in keys {
            self.entry(key)?;
        }
        self.step = self.step.checked_add(1).ok_or(ResidencyError::Overflow)?;
        self.telemetry.steps += 1;
        let (now, half_life) = (self.step, self.config.half_life_steps);
        for &key in keys {
            let e = self.entries.get_mut(&key).expect("validated above");
            e.score = decayed(e.score, e.last_step, now, half_life).saturating_add(HIT_SCORE);
            e.last_step = now;
            e.hits += 1;
            if matches!(e.state, Some(SlotState::Resident)) {
                self.telemetry.hits += 1;
            } else {
                self.telemetry.misses += 1;
            }
        }
        Ok(())
    }

    /// Admission/eviction decisions for the current histogram.
    ///
    /// Candidates are non-resident experts with a score of at least
    /// `admit_min_score`, hottest first. Evictions and admissions are applied
    /// to the metadata immediately; the caller performs them in order.
    pub fn decide(&mut self) -> Result<Vec<Decision>, ResidencyError> {
        let (now, half_life) = (self.step, self.config.half_life_steps);
        let mut candidates: Vec<(u64, ExpertKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.state.is_none())
            .map(|(&k, e)| (decayed(e.score, e.last_step, now, half_life), k))
            .filter(|&(s, _)| s > 0)
            .collect();
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut decisions = Vec::new();
        let mut upload_left = self.config.max_upload_bytes_per_step;
        for (score, key) in candidates {
            let bytes = self.entry(key)?.bytes;
            let refuse = |reason| Decision::Refuse { key, reason };
            if score < self.config.admit_min_score {
                decisions.push(refuse(Refusal::Cold));
                continue;
            }
            if bytes > self.config.capacity_bytes {
                decisions.push(refuse(Refusal::TooLarge));
                continue;
            }
            if self.config.max_upload_bytes_per_step != 0 && bytes > upload_left {
                decisions.push(refuse(Refusal::UploadBudget));
                continue;
            }
            let free = self.config.capacity_bytes - self.used_bytes;
            let victims = if bytes <= free {
                Vec::new()
            } else {
                match self.victims_for(bytes - free, score) {
                    Some(v) => v,
                    None => {
                        decisions.push(refuse(Refusal::NoVictim));
                        continue;
                    }
                }
            };
            for victim in victims {
                let e = self.entry_mut(victim)?;
                e.state = None;
                let vb = e.bytes;
                self.used_bytes -= vb;
                self.telemetry.evictions += 1;
                decisions.push(Decision::Evict {
                    key: victim,
                    bytes: vb,
                });
            }
            let nonce = self.next_nonce()?;
            self.entry_mut(key)?.state = Some(SlotState::Pending(nonce));
            self.used_bytes += bytes;
            if self.config.max_upload_bytes_per_step != 0 {
                upload_left -= bytes;
            }
            self.telemetry.admissions += 1;
            decisions.push(Decision::Admit {
                key,
                bytes,
                ticket: AdmitTicket { key, nonce },
            });
        }
        for d in &decisions {
            if let Decision::Refuse { reason, .. } = d {
                *self
                    .telemetry
                    .refusals
                    .entry(format!("{reason:?}"))
                    .or_insert(0) += 1;
            }
        }
        Ok(decisions)
    }

    /// Coldest evictable residents whose bytes cover `needed`, or `None`.
    fn victims_for(&self, needed: u64, candidate_score: u64) -> Option<Vec<ExpertKey>> {
        let (now, half_life) = (self.step, self.config.half_life_steps);
        let mut pool: Vec<(u64, ExpertKey, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                matches!(e.state, Some(SlotState::Resident)) && e.leases == 0 && e.pins == 0
            })
            .map(|(&k, e)| (decayed(e.score, e.last_step, now, half_life), k, e.bytes))
            .filter(|&(s, _, _)| {
                (s as u128) * (self.config.hysteresis_num as u128)
                    < (candidate_score as u128) * (self.config.hysteresis_den as u128)
            })
            .collect();
        pool.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut freed = 0u64;
        let mut victims = Vec::new();
        for (_, key, bytes) in pool {
            if freed >= needed {
                break;
            }
            freed += bytes;
            victims.push(key);
        }
        (freed >= needed).then_some(victims)
    }

    /// Report the outcome of an admitted upload.
    pub fn complete_upload(&mut self, ticket: AdmitTicket, ok: bool) -> Result<(), ResidencyError> {
        let e = self.entry_mut(ticket.key)?;
        if e.state != Some(SlotState::Pending(ticket.nonce)) {
            return Err(ResidencyError::InvalidTicket);
        }
        if ok {
            e.state = Some(SlotState::Resident);
            Ok(())
        } else {
            e.state = None;
            let bytes = e.bytes;
            self.used_bytes -= bytes;
            self.telemetry.failed_uploads += 1;
            Err(ResidencyError::UploadFailed)
        }
    }

    /// Lease a resident expert for an in-flight route; `None` on a miss.
    pub fn acquire(&mut self, key: ExpertKey) -> Result<Option<DynamicLease>, ResidencyError> {
        if !matches!(self.entry(key)?.state, Some(SlotState::Resident)) {
            return Ok(None);
        }
        let nonce = self.next_nonce()?;
        self.entry_mut(key)?.leases += 1;
        self.leases.insert(nonce, key);
        Ok(Some(DynamicLease { key, nonce }))
    }

    pub fn release(&mut self, lease: DynamicLease) -> Result<(), ResidencyError> {
        if self.leases.get(&lease.nonce) != Some(&lease.key) {
            return Err(ResidencyError::InvalidTicket);
        }
        self.leases.remove(&lease.nonce);
        self.entry_mut(lease.key)?.leases -= 1;
        Ok(())
    }

    pub fn active_leases(&self) -> usize {
        self.leases.len()
    }

    /// Pin an expert (resident or not) so it is never evicted while pinned.
    pub fn pin(&mut self, key: ExpertKey) -> Result<(), ResidencyError> {
        let e = self.entry_mut(key)?;
        e.pins = e.pins.checked_add(1).ok_or(ResidencyError::Overflow)?;
        Ok(())
    }

    pub fn unpin(&mut self, key: ExpertKey) -> Result<(), ResidencyError> {
        let e = self.entry_mut(key)?;
        e.pins = e.pins.checked_sub(1).ok_or(ResidencyError::InvalidTicket)?;
        Ok(())
    }

    /// Shrink capacity (elastic return). Evicts coldest unleased, unpinned residents
    /// until `used <= capacity`; returns the evictions, or `Busy` if the leased or
    /// pinned set alone exceeds the new capacity (nothing is changed then).
    pub fn shrink(&mut self, capacity_bytes: u64) -> Result<Vec<Decision>, ResidencyError> {
        if self.used_bytes <= capacity_bytes {
            self.config.capacity_bytes = capacity_bytes;
            return Ok(Vec::new());
        }
        let victims = self
            .victims_for(self.used_bytes - capacity_bytes, u64::MAX)
            .ok_or(ResidencyError::Busy)?;
        let mut out = Vec::new();
        for key in victims {
            let e = self.entry_mut(key)?;
            e.state = None;
            let bytes = e.bytes;
            self.used_bytes -= bytes;
            self.telemetry.evictions += 1;
            out.push(Decision::Evict { key, bytes });
        }
        self.config.capacity_bytes = capacity_bytes;
        Ok(out)
    }

    pub fn grow(&mut self, capacity_bytes: u64) {
        self.config.capacity_bytes = self.config.capacity_bytes.max(capacity_bytes);
    }

    pub fn report_measured_bytes(&mut self, bytes: u64) {
        self.telemetry.measured_bytes = Some(bytes);
    }

    pub fn telemetry(&self) -> Telemetry {
        let mut t = self.telemetry.clone();
        t.resident_slots = self
            .entries
            .values()
            .filter(|e| matches!(e.state, Some(SlotState::Resident)))
            .count();
        t.planned_bytes = self.used_bytes;
        t
    }

    /// Resident keys in key order (for receipts and determinism checks).
    pub fn resident_set(&self) -> Vec<ExpertKey> {
        self.entries
            .iter()
            .filter(|(_, e)| matches!(e.state, Some(SlotState::Resident)))
            .map(|(&k, _)| k)
            .collect()
    }
}
