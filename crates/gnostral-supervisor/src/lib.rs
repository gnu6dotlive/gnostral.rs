//! Multi-engine supervisor (Q-040) over the EngineProvider v1 draft.
//!
//! Rules this crate enforces, each covered by a test:
//!
//! - **Qualification is per engine and per identity.** A provider is selectable
//!   only after *its own* conformance run passed for the identity it reports now.
//!   If the identity changes (new binary, new revision), the engine is disabled
//!   until it is requalified. No result is copied from a sibling engine.
//! - **Selection uses qualified facts only** (via the contract's registry) and
//!   every start returns the selection receipt.
//! - **Admission denies `UNKNOWN`.** Stale, unknown or merely declared facts,
//!   an unclean driver or too little free GPU memory refuse the start.
//! - **One engine owns the GPU.** Starting on another engine drains and unloads
//!   the active one first; every switch is logged (A to B to A receipts).
//! - **Lifecycle is enforced** with the contract's transition graph.
//! - **Crash isolation.** A provider fault marks the instance `Faulted`,
//!   unloads it and leaves every other engine usable.
//! - **No authority crosses the boundary.** The supervisor only calls the
//!   provider trait, which has no placement or eviction command.

use gnostral_engine_provider_contract::v1::{
    provider_conformance, validate_stream_events, AdmissionFacts, ConformanceReport, DriverHealth,
    EngineProviderStream, PollOutcome, ProviderEntry, ProviderId, ProviderRegistry, RequestId,
    SelectionReceipt, SelectionRequest, SinkControl, StreamEvent, TokenSink,
};
use gnostral_engine_provider_contract::{
    valid_transition, DeclaredCapability, EngineIdentity, EngineProvider, Fact, GenerationReceipt,
    GenerationRequest, InstanceState, LoadIntent, ModelArtifact, ModelHandle, ProviderError,
    QualifiedCapability, ReadinessLevel, RuntimeObservation, SemanticProbeReceipt, UnloadReceipt,
};
use std::collections::BTreeMap;

/// A provider with or without the streaming extension.
pub enum Provider {
    OneShot(Box<dyn EngineProvider>),
    Streaming(Box<dyn EngineProviderStream>),
}

impl Provider {
    fn base(&mut self) -> &mut dyn EngineProvider {
        match self {
            Provider::OneShot(p) => p.as_mut(),
            Provider::Streaming(p) => p.as_mut(),
        }
    }
    fn base_ref(&self) -> &dyn EngineProvider {
        match self {
            Provider::OneShot(p) => p.as_ref(),
            Provider::Streaming(p) => p.as_ref(),
        }
    }
}

/// Sized adapter so the contract's generic conformance harness can run on a trait object.
struct Dyn<'a>(&'a mut dyn EngineProvider);

impl EngineProvider for Dyn<'_> {
    fn identity(&self) -> EngineIdentity {
        self.0.identity()
    }
    fn declared_capabilities(&self) -> Vec<DeclaredCapability> {
        self.0.declared_capabilities()
    }
    fn qualified_capabilities(&self) -> Vec<QualifiedCapability> {
        self.0.qualified_capabilities()
    }
    fn load(&mut self, intent: LoadIntent) -> Result<ModelHandle, ProviderError> {
        self.0.load(intent)
    }
    fn state(&self, handle: &ModelHandle) -> Result<InstanceState, ProviderError> {
        self.0.state(handle)
    }
    fn semantic_probe(
        &mut self,
        handle: &ModelHandle,
        probe_id: &str,
    ) -> Result<SemanticProbeReceipt, ProviderError> {
        self.0.semantic_probe(handle, probe_id)
    }
    fn generate(
        &mut self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<GenerationReceipt, ProviderError> {
        self.0.generate(handle, request)
    }
    fn observe(&self, handle: &ModelHandle) -> Result<RuntimeObservation, ProviderError> {
        self.0.observe(handle)
    }
    fn drain(&mut self, handle: &ModelHandle) -> Result<(), ProviderError> {
        self.0.drain(handle)
    }
    fn unload(&mut self, handle: &ModelHandle) -> Result<UnloadReceipt, ProviderError> {
        self.0.unload(handle)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Denial {
    Stale {
        age_ms: u64,
        ttl_ms: u64,
    },
    /// The named fact is `Unknown`, or only `Declared` (not observed).
    Unobserved {
        fact: &'static str,
    },
    DriverUnhealthy(DriverHealth),
    InsufficientGpu {
        free: u64,
        required: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisorError {
    UnknownProvider(ProviderId),
    IdentityMismatch {
        expected: Box<EngineIdentity>,
        actual: Box<EngineIdentity>,
    },
    NotQualified(ProviderId),
    NoProvider(SelectionReceipt),
    Denied {
        provider: ProviderId,
        denial: Denial,
    },
    ProbeFailed {
        provider: ProviderId,
        probe_id: String,
    },
    Provider {
        provider: ProviderId,
        message: String,
    },
    InvalidTransition {
        provider: ProviderId,
        from: InstanceState,
        to: InstanceState,
    },
    NotStreaming(ProviderId),
    InvalidStream {
        provider: ProviderId,
        detail: String,
    },
    Registry(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Qualified {
        provider: ProviderId,
        passed: bool,
    },
    Disabled {
        provider: ProviderId,
        reason: String,
    },
    Selected {
        receipt: SelectionReceipt,
    },
    Denied {
        provider: ProviderId,
        denial: Denial,
    },
    Loaded {
        provider: ProviderId,
        handle: ModelHandle,
        model: String,
    },
    Unloaded {
        provider: ProviderId,
        receipt: Option<UnloadReceipt>,
    },
    Faulted {
        provider: ProviderId,
        message: String,
    },
    Transition {
        provider: ProviderId,
        from: InstanceState,
        to: InstanceState,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionPolicy {
    /// Free GPU bytes required before loading a model.
    pub min_gpu_free_bytes: u64,
}

struct Slot {
    provider: Provider,
    qualified_identity: Option<EngineIdentity>,
    conformance: Option<ConformanceReport>,
    state: InstanceState,
    handle: Option<ModelHandle>,
    model: Option<ModelArtifact>,
}

/// Everything a start needs besides the generation request.
pub struct StartContext<'a> {
    pub request: &'a SelectionRequest,
    pub intent: &'a LoadIntent,
    /// Host-observed admission facts for a candidate provider.
    pub facts: &'a dyn Fn(&ProviderId) -> AdmissionFacts,
    pub probe_id: &'a str,
    pub now_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartReceipt {
    pub selection: SelectionReceipt,
    pub provider: ProviderId,
    pub switched_from: Option<ProviderId>,
    pub loaded: bool,
    pub generation: GenerationReceipt,
}

pub struct Supervisor {
    registry: ProviderRegistry,
    slots: BTreeMap<ProviderId, Slot>,
    active: Option<ProviderId>,
    policy: AdmissionPolicy,
    log: Vec<Event>,
}

fn observed<T: Copy>(fact: &Fact<T>, name: &'static str) -> Result<T, Denial> {
    match fact {
        Fact::EngineReported { value, .. }
        | Fact::HostObserved { value, .. }
        | Fact::Derived { value, .. } => Ok(*value),
        Fact::Unknown { .. } | Fact::Declared { .. } => Err(Denial::Unobserved { fact: name }),
    }
}

impl Supervisor {
    pub fn new(policy: AdmissionPolicy) -> Self {
        Self {
            registry: ProviderRegistry::default(),
            slots: BTreeMap::new(),
            active: None,
            policy,
            log: Vec::new(),
        }
    }

    pub fn log(&self) -> &[Event] {
        &self.log
    }
    pub fn active(&self) -> Option<&ProviderId> {
        self.active.as_ref()
    }
    pub fn state(&self, id: &ProviderId) -> Option<InstanceState> {
        self.slots.get(id).map(|s| s.state)
    }
    pub fn conformance(&self, id: &ProviderId) -> Option<&ConformanceReport> {
        self.slots.get(id).and_then(|s| s.conformance.as_ref())
    }

    /// Register a provider under `entry`. The entry starts disabled until [`Self::qualify`].
    pub fn register(
        &mut self,
        mut entry: ProviderEntry,
        provider: Provider,
    ) -> Result<(), SupervisorError> {
        let actual = provider.base_ref().identity();
        if actual != entry.identity {
            return Err(SupervisorError::IdentityMismatch {
                expected: Box::new(entry.identity),
                actual: Box::new(actual),
            });
        }
        entry.enabled = false;
        let id = entry.id.clone();
        self.registry
            .register(entry)
            .map_err(|e| SupervisorError::Registry(format!("{e:?}")))?;
        self.slots.insert(
            id,
            Slot {
                provider,
                qualified_identity: None,
                conformance: None,
                state: InstanceState::Offline,
                handle: None,
                model: None,
            },
        );
        Ok(())
    }

    fn set_enabled(&mut self, id: &ProviderId, enabled: bool) {
        let entries: Vec<ProviderEntry> = self
            .registry
            .entries()
            .iter()
            .cloned()
            .map(|mut e| {
                if &e.id == id {
                    e.enabled = enabled;
                }
                e
            })
            .collect();
        self.registry = ProviderRegistry::new(entries).expect("ids were unique already");
    }

    /// Run the shared conformance group against this provider alone.
    pub fn qualify(&mut self, id: &ProviderId) -> Result<bool, SupervisorError> {
        let slot = self
            .slots
            .get_mut(id)
            .ok_or_else(|| SupervisorError::UnknownProvider(id.clone()))?;
        let identity = slot.provider.base_ref().identity();
        let report = provider_conformance(&mut Dyn(slot.provider.base()));
        let passed = report.passed;
        slot.conformance = Some(report);
        slot.qualified_identity = passed.then_some(identity);
        self.log.push(Event::Qualified {
            provider: id.clone(),
            passed,
        });
        self.set_enabled(id, passed);
        Ok(passed)
    }

    /// Disable every provider whose current identity is not the one it qualified with.
    fn revalidate_identities(&mut self) {
        let drifted: Vec<ProviderId> = self
            .slots
            .iter()
            .filter(|(_, s)| {
                s.qualified_identity
                    .as_ref()
                    .is_some_and(|q| *q != s.provider.base_ref().identity())
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in drifted {
            let slot = self.slots.get_mut(&id).expect("listed");
            slot.qualified_identity = None;
            self.log.push(Event::Disabled {
                provider: id.clone(),
                reason: "identity changed since qualification".into(),
            });
            self.set_enabled(&id, false);
        }
    }

    fn transition(&mut self, id: &ProviderId, to: InstanceState) -> Result<(), SupervisorError> {
        let slot = self.slots.get_mut(id).expect("caller checked");
        let from = slot.state;
        if from == to {
            return Ok(());
        }
        if !valid_transition(from, to) {
            return Err(SupervisorError::InvalidTransition {
                provider: id.clone(),
                from,
                to,
            });
        }
        slot.state = to;
        self.log.push(Event::Transition {
            provider: id.clone(),
            from,
            to,
        });
        Ok(())
    }

    fn admit(&self, facts: &AdmissionFacts, now_ms: u64, loading: bool) -> Result<(), Denial> {
        if !facts.is_fresh(now_ms) {
            return Err(Denial::Stale {
                age_ms: now_ms.saturating_sub(facts.sampled_at_ms),
                ttl_ms: facts.ttl_ms,
            });
        }
        match observed(&facts.driver_health, "driver_health")? {
            DriverHealth::NoXid => {}
            other => return Err(Denial::DriverUnhealthy(other)),
        }
        if loading {
            let free = observed(&facts.gpu_free_bytes, "gpu_free_bytes")?;
            if free < self.policy.min_gpu_free_bytes {
                return Err(Denial::InsufficientGpu {
                    free,
                    required: self.policy.min_gpu_free_bytes,
                });
            }
        }
        Ok(())
    }

    fn fault(&mut self, id: &ProviderId, message: String) -> SupervisorError {
        self.log.push(Event::Faulted {
            provider: id.clone(),
            message: message.clone(),
        });
        let _ = self.transition(id, InstanceState::Faulted);
        let slot = self.slots.get_mut(id).expect("caller checked");
        let receipt = slot
            .handle
            .take()
            .and_then(|h| slot.provider.base().unload(&h).ok());
        slot.model = None;
        let _ = self.transition(id, InstanceState::Unloading);
        let _ = self.transition(id, InstanceState::Stopped);
        self.log.push(Event::Unloaded {
            provider: id.clone(),
            receipt,
        });
        if self.active.as_ref() == Some(id) {
            self.active = None;
        }
        SupervisorError::Provider {
            provider: id.clone(),
            message,
        }
    }

    /// Drain and unload the active engine, if any.
    pub fn unload_active(&mut self) -> Result<Option<ProviderId>, SupervisorError> {
        let Some(id) = self.active.clone() else {
            return Ok(None);
        };
        let handle = self.slots[&id]
            .handle
            .clone()
            .expect("active slot has a handle");
        self.transition(&id, InstanceState::Draining)?;
        let slot = self.slots.get_mut(&id).expect("active");
        if let Err(e) = slot.provider.base().drain(&handle) {
            return Err(self.fault(&id, format!("drain failed: {}", e.message)));
        }
        self.transition(&id, InstanceState::Unloading)?;
        let slot = self.slots.get_mut(&id).expect("active");
        let receipt = match slot.provider.base().unload(&handle) {
            Ok(r) => r,
            Err(e) => return Err(self.fault(&id, format!("unload failed: {}", e.message))),
        };
        slot.handle = None;
        slot.model = None;
        self.transition(&id, InstanceState::Stopped)?;
        self.log.push(Event::Unloaded {
            provider: id.clone(),
            receipt: Some(receipt),
        });
        self.active = None;
        Ok(Some(id))
    }

    fn ensure_loaded(
        &mut self,
        id: &ProviderId,
        intent: &LoadIntent,
        probe_id: &str,
    ) -> Result<bool, SupervisorError> {
        let slot = &self.slots[id];
        if slot.handle.is_some() && slot.model.as_ref() == Some(&intent.model) {
            return Ok(false);
        }
        if self.active.is_some() {
            self.unload_active()?;
        }
        // A reload is a new instance.
        if self.slots[id].state == InstanceState::Stopped {
            self.slots.get_mut(id).expect("present").state = InstanceState::Offline;
        }
        self.transition(id, InstanceState::Loading)?;
        let slot = self.slots.get_mut(id).expect("present");
        let handle = match slot.provider.base().load(intent.clone()) {
            Ok(h) => h,
            Err(e) => return Err(self.fault(id, format!("load failed: {}", e.message))),
        };
        slot.handle = Some(handle.clone());
        slot.model = Some(intent.model.clone());
        self.active = Some(id.clone());
        self.transition(id, InstanceState::Ready(ReadinessLevel::Control))?;
        let slot = self.slots.get_mut(id).expect("present");
        let probe = slot.provider.base().semantic_probe(&handle, probe_id);
        match probe {
            Ok(r) if r.passed && r.handle == handle => {}
            Ok(_) => {
                let _ = self.fault(id, format!("semantic probe {probe_id} failed"));
                return Err(SupervisorError::ProbeFailed {
                    provider: id.clone(),
                    probe_id: probe_id.into(),
                });
            }
            Err(e) => return Err(self.fault(id, format!("probe error: {}", e.message))),
        }
        self.transition(id, InstanceState::Ready(ReadinessLevel::Semantic))?;
        self.log.push(Event::Loaded {
            provider: id.clone(),
            handle,
            model: intent.model.name.clone(),
        });
        Ok(true)
    }

    fn select_and_admit(
        &mut self,
        request: &SelectionRequest,
        facts: &dyn Fn(&ProviderId) -> AdmissionFacts,
        now_ms: u64,
    ) -> Result<(SelectionReceipt, ProviderId), SupervisorError> {
        self.revalidate_identities();
        let receipt = self.registry.select(request, now_ms);
        self.log.push(Event::Selected {
            receipt: receipt.clone(),
        });
        let Some(id) = receipt.chosen.clone() else {
            return Err(SupervisorError::NoProvider(receipt));
        };
        if self.slots[&id].qualified_identity.is_none() {
            return Err(SupervisorError::NotQualified(id));
        }
        let loading = self.slots[&id].model.as_ref() != Some(&request.model);
        if let Err(denial) = self.admit(&facts(&id), now_ms, loading) {
            self.log.push(Event::Denied {
                provider: id.clone(),
                denial: denial.clone(),
            });
            return Err(SupervisorError::Denied {
                provider: id,
                denial,
            });
        }
        Ok((receipt, id))
    }

    /// Select, admit, (re)load if needed, generate.
    pub fn start(
        &mut self,
        ctx: &StartContext<'_>,
        generation: GenerationRequest,
    ) -> Result<StartReceipt, SupervisorError> {
        let (selection, id) = self.select_and_admit(ctx.request, ctx.facts, ctx.now_ms)?;
        let previous = self.active.clone().filter(|a| a != &id);
        let loaded = self.ensure_loaded(&id, ctx.intent, ctx.probe_id)?;
        self.transition(&id, InstanceState::Serving)?;
        let slot = self.slots.get_mut(&id).expect("present");
        let handle = slot.handle.clone().expect("loaded");
        match slot.provider.base().generate(&handle, generation) {
            Ok(receipt) => Ok(StartReceipt {
                selection,
                provider: id,
                switched_from: previous,
                loaded,
                generation: receipt,
            }),
            Err(e) => Err(self.fault(&id, format!("generate failed: {}", e.message))),
        }
    }

    /// Streaming start: events are validated against the stream contract before returning.
    pub fn start_stream(
        &mut self,
        ctx: &StartContext<'_>,
        generation: GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<(ProviderId, RequestId), SupervisorError> {
        let (_, id) = self.select_and_admit(ctx.request, ctx.facts, ctx.now_ms)?;
        if !matches!(self.slots[&id].provider, Provider::Streaming(_)) {
            return Err(SupervisorError::NotStreaming(id));
        }
        self.ensure_loaded(&id, ctx.intent, ctx.probe_id)?;
        self.transition(&id, InstanceState::Serving)?;
        let slot = self.slots.get_mut(&id).expect("present");
        let handle = slot.handle.clone().expect("loaded");
        let Provider::Streaming(p) = &mut slot.provider else {
            unreachable!("checked above")
        };
        let mut checked = CheckedSink {
            inner: sink,
            events: Vec::new(),
        };
        let started = p.start_generate(&handle, generation, &mut checked);
        if let Err(detail) = validate_stream_events(&checked.events) {
            return Err(SupervisorError::InvalidStream {
                provider: id,
                detail,
            });
        }
        match started {
            Ok(rid) => Ok((id, rid)),
            Err(e) => Err(self.fault(&id, format!("start_generate failed: {}", e.message))),
        }
    }

    pub fn poll(
        &mut self,
        id: &ProviderId,
        rid: RequestId,
    ) -> Result<PollOutcome, SupervisorError> {
        let slot = self
            .slots
            .get_mut(id)
            .ok_or_else(|| SupervisorError::UnknownProvider(id.clone()))?;
        let Provider::Streaming(p) = &mut slot.provider else {
            return Err(SupervisorError::NotStreaming(id.clone()));
        };
        p.poll(rid).map_err(|e| SupervisorError::Provider {
            provider: id.clone(),
            message: e.message,
        })
    }

    pub fn cancel(&mut self, id: &ProviderId, rid: RequestId) -> Result<(), SupervisorError> {
        let slot = self
            .slots
            .get_mut(id)
            .ok_or_else(|| SupervisorError::UnknownProvider(id.clone()))?;
        let Provider::Streaming(p) = &mut slot.provider else {
            return Err(SupervisorError::NotStreaming(id.clone()));
        };
        p.cancel(rid).map_err(|e| SupervisorError::Provider {
            provider: id.clone(),
            message: e.message,
        })
    }
}

struct CheckedSink<'a> {
    inner: &'a mut dyn TokenSink,
    events: Vec<StreamEvent>,
}

impl TokenSink for CheckedSink<'_> {
    fn on_event(&mut self, event: StreamEvent) -> SinkControl {
        self.events.push(event.clone());
        self.inner.on_event(event)
    }
}
