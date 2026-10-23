//! One device envelope, arbitrated between components.
//!
//! `available = device - desktop_ambient - desktop_reserve - min_free`. Minima are
//! granted first (or the whole split is refused), then the remainder is handed
//! out in [`Component`] priority order up to each component's desired size. The
//! split is a pure function of its inputs and is recorded per step.

use crate::MemoryError;
use std::collections::BTreeMap;

/// Allocation priority, highest first: without workspace nothing runs, without KV
/// no request runs, the expert cache is an accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Component {
    Workspace,
    Kv,
    Experts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Envelope {
    pub device_bytes: u64,
    /// Measured desktop usage on the card; `None` is `UNKNOWN` and is refused.
    pub desktop_ambient: Option<u64>,
    /// Extra headroom kept for the desktop (Q-041 profile).
    pub desktop_reserve: u64,
    /// Whole-card free floor the program promises (M1: 1 GiB).
    pub min_free: u64,
}

impl Envelope {
    pub fn available(&self) -> Result<u64, MemoryError> {
        let ambient = self.desktop_ambient.ok_or(MemoryError::Unknown {
            what: "desktop_ambient",
        })?;
        let held_back = ambient
            .checked_add(self.desktop_reserve)
            .and_then(|v| v.checked_add(self.min_free))
            .ok_or(MemoryError::Overflow)?;
        match self.device_bytes.checked_sub(held_back) {
            Some(available) if available > 0 => Ok(available),
            _ => Err(MemoryError::EnvelopeTooSmall {
                device: self.device_bytes,
                held_back,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub component: Component,
    pub min: u64,
    pub desired: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub available: u64,
    pub grants: BTreeMap<Component, u64>,
    /// Bytes inside `available` that nobody asked for.
    pub unallocated: u64,
}

impl Split {
    pub fn grant(&self, component: Component) -> u64 {
        self.grants.get(&component).copied().unwrap_or(0)
    }
    pub fn granted(&self) -> u64 {
        self.grants.values().sum()
    }
}

pub fn arbitrate(envelope: &Envelope, requests: &[Request]) -> Result<Split, MemoryError> {
    let available = envelope.available()?;
    let mut by_component: BTreeMap<Component, Request> = BTreeMap::new();
    for r in requests {
        if r.desired < r.min || by_component.insert(r.component, *r).is_some() {
            return Err(MemoryError::InvalidRequest);
        }
    }
    let needed = by_component
        .values()
        .try_fold(0u64, |s, r| s.checked_add(r.min))
        .ok_or(MemoryError::Overflow)?;
    if needed > available {
        return Err(MemoryError::MinimaExceedEnvelope { needed, available });
    }
    let mut left = available - needed;
    let mut grants = BTreeMap::new();
    for (component, r) in &by_component {
        let extra = (r.desired - r.min).min(left);
        left -= extra;
        grants.insert(*component, r.min + extra);
    }
    Ok(Split {
        available,
        grants,
        unallocated: left,
    })
}

/// Per-step record of every split, kept for receipts.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ArbitrationLog {
    entries: Vec<(u64, Result<Split, MemoryError>)>,
}

impl ArbitrationLog {
    pub fn record(
        &mut self,
        step: u64,
        envelope: &Envelope,
        requests: &[Request],
    ) -> Result<Split, MemoryError> {
        let result = arbitrate(envelope, requests);
        self.entries.push((step, result.clone()));
        result
    }
    pub fn entries(&self) -> &[(u64, Result<Split, MemoryError>)] {
        &self.entries
    }
}
