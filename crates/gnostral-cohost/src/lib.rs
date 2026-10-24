//! Several models sharing one device envelope (Q-043).
//!
//! A language model, a speech-to-text model, a text-to-speech model and a small
//! draft model live on the same card. Each tenant declares:
//!
//! - `core_bytes`: what must be resident for it to run at all (dense weights,
//!   workspace, its KV floor);
//! - an optional elastic range: bytes it can use well but can live without, such
//!   as a MoE expert cache, where fewer resident experts cost speed, not
//!   correctness;
//! - a [`Class`] (who may take memory from whom) and whether it may be parked
//!   (unloaded while idle, at the cost of a cold start).
//!
//! The [`Broker`] turns admissions, releases and capacity changes into a
//! [`Plan`]: frees first (shrink, park, unload), then loads and grows. Executed in
//! order, a plan never puts more bytes on the card than the capacity, at any step.
//! A request that cannot be met is a typed refusal and changes nothing.
//!
//! Policy, highest priority first: `Realtime`, `Interactive`, `Background`.
//! Elastic surplus is taken from the lowest class first and never from a class
//! above the requester; only idle, parkable tenants of a strictly lower class
//! are parked. Freed bytes are handed back to elastic tenants by priority.
//!
//! Everything here is bookkeeping: no allocation, no device call, no
//! measurement. The capacity comes from [`gnostral_memory_plan::envelope`], so an
//! `UNKNOWN` desktop ambient is refused before any tenant is admitted.

use gnostral_memory_plan::envelope::Envelope;
use gnostral_memory_plan::MemoryError;
use std::collections::BTreeMap;

/// Who may take memory from whom. Earlier variants have higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Audio in or out while a person is talking or listening.
    Realtime,
    /// A chat answer someone is waiting for.
    Interactive,
    /// Batch work, indexing, prefetching.
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Language,
    SpeechToText,
    TextToSpeech,
    Draft,
    Embedding,
    Other,
}

/// Elastic bytes a tenant can use but can run without (e.g. a MoE expert cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elastic {
    /// Below this the tenant does not run acceptably; part of its admission need.
    pub min: u64,
    pub max: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantSpec {
    pub name: String,
    pub kind: Kind,
    pub class: Class,
    pub core_bytes: u64,
    pub elastic: Option<Elastic>,
    pub parkable: bool,
}

impl TenantSpec {
    fn need(&self) -> Option<u64> {
        self.core_bytes
            .checked_add(self.elastic.map_or(0, |e| e.min))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Core resident plus this many elastic bytes.
    Loaded { elastic: u64 },
    /// Nothing resident; must be woken (cold start) before it runs.
    Parked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Tenant {
    spec: TenantSpec,
    residency: Residency,
    busy: bool,
}

impl Tenant {
    fn bytes(&self) -> u64 {
        match self.residency {
            Residency::Loaded { elastic } => self.spec.core_bytes + elastic,
            Residency::Parked => 0,
        }
    }
    fn elastic(&self) -> u64 {
        match self.residency {
            Residency::Loaded { elastic } => elastic,
            Residency::Parked => 0,
        }
    }
    fn surplus(&self) -> u64 {
        match (self.residency, self.spec.elastic) {
            (Residency::Loaded { elastic }, Some(e)) => elastic - e.min,
            _ => 0,
        }
    }
    fn headroom(&self) -> u64 {
        match (self.residency, self.spec.elastic) {
            (Residency::Loaded { elastic }, Some(e)) => e.max - elastic,
            _ => 0,
        }
    }
}

/// One step of a plan. Byte counts are what the step adds or removes on the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Shrink {
        id: TenantId,
        from: u64,
        to: u64,
    },
    Park {
        id: TenantId,
        bytes: u64,
    },
    Unload {
        id: TenantId,
        bytes: u64,
    },
    Load {
        id: TenantId,
        core: u64,
        elastic: u64,
    },
    Grow {
        id: TenantId,
        from: u64,
        to: u64,
    },
}

impl Step {
    /// Signed change of resident bytes.
    pub fn delta(&self) -> i128 {
        match *self {
            Step::Shrink { from, to, .. } | Step::Grow { from, to, .. } => {
                i128::from(to) - i128::from(from)
            }
            Step::Park { bytes, .. } | Step::Unload { bytes, .. } => -i128::from(bytes),
            Step::Load { core, elastic, .. } => i128::from(core) + i128::from(elastic),
        }
    }
}

pub type Plan = Vec<Step>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohostError {
    Memory(MemoryError),
    /// Elastic range with `min > max`, zero-byte tenant or overflow.
    InvalidSpec,
    NotFound(TenantId),
    /// The tenant is in the wrong residency for the call (e.g. waking a loaded tenant).
    WrongState(TenantId),
    /// Not enough memory even after every allowed shrink and park.
    Refused {
        needed: u64,
        reclaimable: u64,
    },
}

impl std::fmt::Display for CohostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cohost: {self:?}")
    }
}

impl std::error::Error for CohostError {}

impl From<MemoryError> for CohostError {
    fn from(e: MemoryError) -> Self {
        CohostError::Memory(e)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Admit(String),
    Wake(TenantId),
    Release(TenantId),
    SetCapacity(u64),
}

/// One receipted broker decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub seq: u64,
    pub operation: Operation,
    pub result: Result<Plan, CohostError>,
    /// Resident bytes after the decision.
    pub used_after: u64,
    pub capacity: u64,
}

/// A tenant as seen from outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub id: TenantId,
    pub name: String,
    pub kind: Kind,
    pub class: Class,
    pub residency: Residency,
    pub bytes: u64,
    pub busy: bool,
}

#[derive(Debug, Clone)]
pub struct Broker {
    capacity: u64,
    tenants: BTreeMap<TenantId, Tenant>,
    next: u32,
    receipts: Vec<Receipt>,
}

impl Broker {
    /// Capacity is the envelope's `available` bytes (desktop ambient, reserve and free floor removed).
    pub fn new(envelope: &Envelope) -> Result<Self, CohostError> {
        Ok(Self::with_capacity(envelope.available()?))
    }

    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            capacity,
            tenants: BTreeMap::new(),
            next: 0,
            receipts: Vec::new(),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used(&self) -> u64 {
        self.tenants.values().map(Tenant::bytes).sum()
    }

    pub fn receipts(&self) -> &[Receipt] {
        &self.receipts
    }

    pub fn views(&self) -> Vec<View> {
        self.tenants
            .iter()
            .map(|(&id, t)| View {
                id,
                name: t.spec.name.clone(),
                kind: t.spec.kind,
                class: t.spec.class,
                residency: t.residency,
                bytes: t.bytes(),
                busy: t.busy,
            })
            .collect()
    }

    pub fn view(&self, id: TenantId) -> Option<View> {
        self.views().into_iter().find(|v| v.id == id)
    }

    /// A busy tenant is never parked.
    pub fn set_busy(&mut self, id: TenantId, busy: bool) -> Result<(), CohostError> {
        self.tenants
            .get_mut(&id)
            .ok_or(CohostError::NotFound(id))?
            .busy = busy;
        Ok(())
    }

    /// Load a new tenant, shrinking or parking others if the policy allows.
    pub fn admit(&mut self, spec: TenantSpec) -> Result<(TenantId, Plan), CohostError> {
        let name = spec.name.clone();
        let id = TenantId(self.next);
        let result = self.admit_inner(id, spec);
        if result.is_ok() {
            self.next += 1;
        }
        self.record(Operation::Admit(name), result.clone().map(|p| p.1));
        result
    }

    fn admit_inner(
        &mut self,
        id: TenantId,
        spec: TenantSpec,
    ) -> Result<(TenantId, Plan), CohostError> {
        validate(&spec)?;
        let mut tenant = Tenant {
            spec,
            residency: Residency::Parked,
            busy: false,
        };
        let plan = self.bring_in(id, &mut tenant)?;
        self.tenants.insert(id, tenant);
        Ok((id, plan))
    }

    /// Reload a parked tenant under the same policy as an admission.
    pub fn wake(&mut self, id: TenantId) -> Result<Plan, CohostError> {
        let result = self.wake_inner(id);
        self.record(Operation::Wake(id), result.clone());
        result
    }

    fn wake_inner(&mut self, id: TenantId) -> Result<Plan, CohostError> {
        let mut tenant = self.tenants.remove(&id).ok_or(CohostError::NotFound(id))?;
        if tenant.residency != Residency::Parked {
            self.tenants.insert(id, tenant);
            return Err(CohostError::WrongState(id));
        }
        let result = self.bring_in(id, &mut tenant);
        self.tenants.insert(id, tenant);
        result
    }

    /// Unload a tenant for good and hand its bytes back to elastic tenants.
    pub fn release(&mut self, id: TenantId) -> Result<Plan, CohostError> {
        let result = match self.tenants.remove(&id) {
            None => Err(CohostError::NotFound(id)),
            Some(t) => {
                let mut plan = Vec::new();
                if t.bytes() > 0 {
                    plan.push(Step::Unload {
                        id,
                        bytes: t.bytes(),
                    });
                }
                self.regrow(&mut plan);
                Ok(plan)
            }
        };
        self.record(Operation::Release(id), result.clone());
        result
    }

    /// New capacity, e.g. after a desktop-reserve release or re-acquire (Q-041).
    ///
    /// Shrinking takes elastic surplus from the lowest class first, then parks idle parkable
    /// tenants from the lowest class; if that is not enough the change is refused and nothing moves.
    pub fn set_capacity(&mut self, capacity: u64) -> Result<Plan, CohostError> {
        let result = self.set_capacity_inner(capacity);
        self.record(Operation::SetCapacity(capacity), result.clone());
        result
    }

    fn set_capacity_inner(&mut self, capacity: u64) -> Result<Plan, CohostError> {
        let used = self.used();
        let mut plan = Vec::new();
        if capacity < used {
            let deficit = used - capacity;
            let (steps, reclaimable) = self.reclaim(deficit, None);
            if reclaimable < deficit {
                return Err(CohostError::Refused {
                    needed: deficit,
                    reclaimable,
                });
            }
            self.apply(&steps);
            plan.extend(steps);
        }
        self.capacity = capacity;
        self.regrow(&mut plan);
        Ok(plan)
    }

    fn bring_in(&mut self, id: TenantId, tenant: &mut Tenant) -> Result<Plan, CohostError> {
        let need = tenant.spec.need().ok_or(CohostError::InvalidSpec)?;
        let free = self.capacity.saturating_sub(self.used());
        let mut plan = Vec::new();
        if need > free {
            let deficit = need - free;
            let (steps, reclaimable) = self.reclaim(deficit, Some(tenant.spec.class));
            if reclaimable < deficit {
                return Err(CohostError::Refused {
                    needed: deficit,
                    reclaimable,
                });
            }
            self.apply(&steps);
            plan.extend(steps);
        }
        let free = self.capacity - self.used();
        let elastic = tenant
            .spec
            .elastic
            .map_or(0, |e| e.max.min(free - tenant.spec.core_bytes));
        tenant.residency = Residency::Loaded { elastic };
        plan.push(Step::Load {
            id,
            core: tenant.spec.core_bytes,
            elastic,
        });
        Ok(plan)
    }

    /// Steps freeing at least `deficit` bytes if possible, and the total that could be freed.
    ///
    /// `requester` limits whom we may take from: elastic surplus from classes at or below it,
    /// parking only from classes strictly below it. `None` (a capacity drop) may take from anyone.
    fn reclaim(&self, deficit: u64, requester: Option<Class>) -> (Plan, u64) {
        let may_shrink = |c: Class| requester.is_none_or(|r| c >= r);
        let may_park = |c: Class| requester.is_none_or(|r| c > r);
        let mut shrinkable: Vec<(&TenantId, &Tenant)> = self
            .tenants
            .iter()
            .filter(|(_, t)| t.surplus() > 0 && may_shrink(t.spec.class))
            .collect();
        shrinkable.sort_by_key(|(id, t)| {
            (
                std::cmp::Reverse(t.spec.class),
                std::cmp::Reverse(t.surplus()),
                **id,
            )
        });
        let mut parkable: Vec<(&TenantId, &Tenant)> = self
            .tenants
            .iter()
            .filter(|(_, t)| t.spec.parkable && !t.busy && t.bytes() > 0 && may_park(t.spec.class))
            .collect();
        parkable.sort_by_key(|(id, t)| {
            (
                std::cmp::Reverse(t.spec.class),
                std::cmp::Reverse(t.bytes()),
                **id,
            )
        });

        let mut steps = Vec::new();
        let mut freed = 0u64;
        let mut shrunk: BTreeMap<TenantId, u64> = BTreeMap::new();
        for (&id, t) in shrinkable {
            if freed >= deficit {
                break;
            }
            let take = t.surplus().min(deficit - freed);
            let from = t.elastic();
            steps.push(Step::Shrink {
                id,
                from,
                to: from - take,
            });
            shrunk.insert(id, take);
            freed += take;
        }
        for (&id, t) in parkable {
            if freed >= deficit {
                break;
            }
            // Parking frees what is left after any shrink planned above.
            let bytes = t.bytes() - shrunk.get(&id).copied().unwrap_or(0);
            steps.push(Step::Park { id, bytes });
            freed += bytes;
        }
        if freed < deficit {
            let total = self.reclaimable(requester);
            return (Vec::new(), total);
        }
        (steps, freed)
    }

    fn reclaimable(&self, requester: Option<Class>) -> u64 {
        self.tenants
            .values()
            .map(|t| {
                let park = t.spec.parkable && !t.busy && requester.is_none_or(|r| t.spec.class > r);
                if park {
                    t.bytes()
                } else if requester.is_none_or(|r| t.spec.class >= r) {
                    t.surplus()
                } else {
                    0
                }
            })
            .sum()
    }

    fn apply(&mut self, steps: &[Step]) {
        for step in steps {
            match *step {
                Step::Shrink { id, to, .. } | Step::Grow { id, to, .. } => {
                    if let Some(t) = self.tenants.get_mut(&id) {
                        t.residency = Residency::Loaded { elastic: to };
                    }
                }
                Step::Park { id, .. } => {
                    if let Some(t) = self.tenants.get_mut(&id) {
                        t.residency = Residency::Parked;
                    }
                }
                Step::Unload { .. } | Step::Load { .. } => {}
            }
        }
    }

    /// Hand free bytes to loaded elastic tenants, highest class first, then by id.
    fn regrow(&mut self, plan: &mut Plan) {
        let mut free = self.capacity.saturating_sub(self.used());
        let mut order: Vec<(Class, TenantId)> = self
            .tenants
            .iter()
            .filter(|(_, t)| t.headroom() > 0)
            .map(|(&id, t)| (t.spec.class, id))
            .collect();
        order.sort();
        for (_, id) in order {
            if free == 0 {
                break;
            }
            let t = &self.tenants[&id];
            let add = t.headroom().min(free);
            let from = t.elastic();
            let step = Step::Grow {
                id,
                from,
                to: from + add,
            };
            self.apply(&[step]);
            plan.push(step);
            free -= add;
        }
    }

    fn record(&mut self, operation: Operation, result: Result<Plan, CohostError>) {
        let seq = self.receipts.len() as u64;
        self.receipts.push(Receipt {
            seq,
            operation,
            result,
            used_after: self.used(),
            capacity: self.capacity,
        });
    }
}

fn validate(spec: &TenantSpec) -> Result<(), CohostError> {
    if let Some(e) = spec.elastic {
        if e.min > e.max {
            return Err(CohostError::InvalidSpec);
        }
    }
    match spec.need() {
        Some(0) | None => Err(CohostError::InvalidSpec),
        Some(_) => Ok(()),
    }
}

/// Resident bytes after each step of `plan`, starting from `used`; `None` if any step overbooks.
pub fn replay(used: u64, capacity: u64, plan: &[Step]) -> Option<Vec<u64>> {
    let mut now = i128::from(used);
    let mut trace = Vec::with_capacity(plan.len());
    for step in plan {
        now += step.delta();
        if now < 0 || now > i128::from(capacity) {
            return None;
        }
        trace.push(u64::try_from(now).ok()?);
    }
    Some(trace)
}
