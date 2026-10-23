use gnostral_engine_provider_contract::v1::{
    AdmissionFacts, CapabilityKey, CapabilityRequirement, CapabilityValue, DriverHealth,
    EngineProviderStream, PollOutcome, ProviderEntry, ProviderError as V1Error, ProviderId,
    RequestId, SelectionRequest, SinkControl, StopReason, StreamEvent, TokenSink,
};
use gnostral_engine_provider_contract::{
    DeclaredCapability, EngineIdentity, EngineProvider, EngineTransport, Fact, GenerationReceipt,
    GenerationRequest, InstanceState, LoadIntent, ModelArtifact, ModelHandle, ProviderError,
    QualificationStatus, QualifiedCapability, ReadinessLevel, ResourceEnvelope, RuntimeObservation,
    SemanticProbeReceipt, ServingMode, UnloadReceipt,
};
use gnostral_supervisor::{
    AdmissionPolicy, Denial, Event, Provider, StartContext, Supervisor, SupervisorError,
};
use std::cell::RefCell;
use std::rc::Rc;

const GIB: u64 = 1 << 30;

type FactsFn = dyn Fn(&ProviderId) -> AdmissionFacts;

#[derive(Default)]
struct Shared {
    revision: String,
    loads: u32,
    unloads: u32,
    fail_generate: bool,
    probe_passes: bool,
    loaded: Option<ModelHandle>,
}

struct Stub {
    name: String,
    shared: Rc<RefCell<Shared>>,
    streaming: bool,
    cancelled: bool,
}

fn identity(name: &str, revision: &str) -> EngineIdentity {
    EngineIdentity {
        name: name.into(),
        revision: Some(revision.into()),
        binary_sha256: Some(format!("{name}-sha")),
        transport: EngineTransport::LocalServer,
    }
}

fn qualified(key: CapabilityKey, value: &str) -> QualifiedCapability {
    QualifiedCapability {
        key: key.as_str().into(),
        value: Some(value.into()),
        status: QualificationStatus::Qualified,
        profile: "rtx3070".into(),
        evidence_ref: "evidence/runs/stub".into(),
    }
}

fn unknown<T>() -> Fact<T> {
    Fact::Unknown {
        reason: "stub".into(),
    }
}

fn gen_receipt(text: &str) -> GenerationReceipt {
    GenerationReceipt {
        text: text.into(),
        prompt_tokens: unknown(),
        completion_tokens: unknown(),
        ttft_ms: unknown(),
    }
}

impl EngineProvider for Stub {
    fn identity(&self) -> EngineIdentity {
        identity(&self.name, &self.shared.borrow().revision)
    }
    fn declared_capabilities(&self) -> Vec<DeclaredCapability> {
        vec![DeclaredCapability {
            key: "streaming".into(),
            value: self.streaming.to_string(),
            source: "stub".into(),
        }]
    }
    fn qualified_capabilities(&self) -> Vec<QualifiedCapability> {
        vec![qualified(CapabilityKey::MaxContextTokens, "8192")]
    }
    fn load(&mut self, intent: LoadIntent) -> Result<ModelHandle, ProviderError> {
        let mut s = self.shared.borrow_mut();
        s.loads += 1;
        let h = ModelHandle(format!("{}:{}:{}", self.name, intent.model.name, s.loads));
        s.loaded = Some(h.clone());
        Ok(h)
    }
    fn state(&self, _: &ModelHandle) -> Result<InstanceState, ProviderError> {
        Ok(InstanceState::Ready(ReadinessLevel::Semantic))
    }
    fn semantic_probe(
        &mut self,
        handle: &ModelHandle,
        probe_id: &str,
    ) -> Result<SemanticProbeReceipt, ProviderError> {
        Ok(SemanticProbeReceipt {
            handle: handle.clone(),
            probe_id: probe_id.into(),
            passed: self.shared.borrow().probe_passes,
            observed_output: "4".into(),
            engine: self.identity(),
            model: model("m"),
            completed_at_ms: 0,
        })
    }
    fn generate(
        &mut self,
        _: &ModelHandle,
        r: GenerationRequest,
    ) -> Result<GenerationReceipt, ProviderError> {
        if self.shared.borrow().fail_generate {
            return Err(ProviderError {
                message: "engine crashed".into(),
            });
        }
        Ok(gen_receipt(&format!("{}:{}", self.name, r.prompt)))
    }
    fn observe(&self, handle: &ModelHandle) -> Result<RuntimeObservation, ProviderError> {
        Ok(RuntimeObservation {
            handle: handle.clone(),
            gpu_resident_bytes: unknown(),
            host_resident_bytes: unknown(),
            kv_bytes: unknown(),
            queue_depth: unknown(),
            active_requests: unknown(),
            driver_health: unknown(),
            placement_summary: unknown(),
        })
    }
    fn drain(&mut self, _: &ModelHandle) -> Result<(), ProviderError> {
        Ok(())
    }
    fn unload(&mut self, handle: &ModelHandle) -> Result<UnloadReceipt, ProviderError> {
        let mut s = self.shared.borrow_mut();
        s.unloads += 1;
        s.loaded = None;
        Ok(UnloadReceipt {
            handle: handle.clone(),
            engine_acknowledged: true,
            completed_at_ms: 0,
        })
    }
}

impl EngineProviderStream for Stub {
    fn start_generate(
        &mut self,
        _: &ModelHandle,
        r: GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<RequestId, V1Error> {
        for (i, ch) in r.prompt.chars().enumerate() {
            if sink.on_event(StreamEvent::Token {
                index: i as u32,
                token_id: None,
                text_delta: ch.to_string(),
                logprob: None,
            }) == SinkControl::Pause
            {
                break;
            }
        }
        Ok(RequestId(1))
    }
    fn poll(&mut self, _: RequestId) -> Result<PollOutcome, V1Error> {
        Ok(if self.cancelled {
            PollOutcome::Cancelled
        } else {
            PollOutcome::Pending
        })
    }
    fn cancel(&mut self, _: RequestId) -> Result<(), V1Error> {
        self.cancelled = true;
        Ok(())
    }
}

fn model(name: &str) -> ModelArtifact {
    ModelArtifact {
        name: name.into(),
        sha256: Some("model-sha".into()),
        format: "GGUF".into(),
        bytes: None,
    }
}

fn intent(name: &str) -> LoadIntent {
    LoadIntent {
        model: model(name),
        requested_context_tokens: Some(4096),
        requested_concurrency: Some(1),
        resources: ResourceEnvelope::default(),
        serving_mode: ServingMode::Persistent,
    }
}

fn entry(id: &str, ctx: &str, revision: &str) -> ProviderEntry {
    ProviderEntry {
        id: ProviderId(id.into()),
        identity: identity(id, revision),
        profile: "rtx3070".into(),
        qualified: vec![qualified(CapabilityKey::MaxContextTokens, ctx)],
        declared: Vec::new(),
        enabled: true,
    }
}

fn selection(name: &str, min_ctx: u64) -> SelectionRequest {
    SelectionRequest {
        required: vec![CapabilityRequirement::new(CapabilityKey::MaxContextTokens)
            .at_least(CapabilityValue::UInt(min_ctx))],
        preferred: Vec::new(),
        model: model(name),
        profile: "rtx3070".into(),
    }
}

fn healthy(_: &ProviderId) -> AdmissionFacts {
    AdmissionFacts {
        handle: None,
        sampled_at_ms: 100,
        ttl_ms: 1000,
        gpu_free_bytes: Fact::HostObserved {
            value: 6 * GIB,
            source: "nvml".into(),
            observed_at_ms: 100,
        },
        host_free_bytes: unknown(),
        driver_health: Fact::HostObserved {
            value: DriverHealth::NoXid,
            source: "dmesg".into(),
            observed_at_ms: 100,
        },
        admitted_context_tokens: unknown(),
        admitted_concurrency: unknown(),
    }
}

fn ctx<'a>(
    request: &'a SelectionRequest,
    intent: &'a LoadIntent,
    facts: &'a dyn Fn(&ProviderId) -> AdmissionFacts,
) -> StartContext<'a> {
    StartContext {
        request,
        intent,
        facts,
        probe_id: "p",
        now_ms: 100,
    }
}

fn gen(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        max_tokens: 4,
        priority: 0,
        deadline_ms: None,
    }
}

fn stub(name: &str, streaming: bool) -> (Provider, Rc<RefCell<Shared>>) {
    let shared = Rc::new(RefCell::new(Shared {
        revision: "r1".into(),
        probe_passes: true,
        ..Shared::default()
    }));
    let s = Stub {
        name: name.into(),
        shared: shared.clone(),
        streaming,
        cancelled: false,
    };
    let p = if streaming {
        Provider::Streaming(Box::new(s))
    } else {
        Provider::OneShot(Box::new(s))
    };
    (p, shared)
}

fn supervisor() -> (Supervisor, Rc<RefCell<Shared>>, Rc<RefCell<Shared>>) {
    let mut sup = Supervisor::new(AdmissionPolicy {
        min_gpu_free_bytes: 4 * GIB,
    });
    let (a, sa) = stub("a", false);
    let (b, sb) = stub("b", true);
    sup.register(entry("a", "8192", "r1"), a).unwrap();
    sup.register(entry("b", "32768", "r1"), b).unwrap();
    (sup, sa, sb)
}

#[test]
fn unqualified_engines_are_never_selected() {
    let (mut sup, _, _) = supervisor();
    match sup.start(
        &ctx(&selection("m", 1024), &intent("m"), &healthy),
        gen("x"),
    ) {
        Err(SupervisorError::NoProvider(receipt)) => assert_eq!(receipt.rejected.len(), 2),
        other => panic!("{other:?}"),
    }
    assert!(sup.qualify(&ProviderId("a".into())).unwrap());
    let r = sup
        .start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("x"),
        )
        .unwrap();
    assert_eq!(r.provider, ProviderId("a".into()));
    assert_eq!(r.generation.text, "a:x");
}

#[test]
fn capability_is_not_inherited_from_a_sibling() {
    let (mut sup, _, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    // Only b is qualified for 32K, and b has not passed its own conformance run.
    assert!(matches!(
        sup.start(
            &ctx(&selection("m", 32768), &intent("m"), &healthy),
            gen("x")
        ),
        Err(SupervisorError::NoProvider(_))
    ));
    sup.qualify(&ProviderId("b".into())).unwrap();
    let r = sup
        .start(
            &ctx(&selection("m", 32768), &intent("m"), &healthy),
            gen("x"),
        )
        .unwrap();
    assert_eq!(r.provider, ProviderId("b".into()));
}

#[test]
fn identity_drift_disables_until_requalified() {
    let (mut sup, sa, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    sa.borrow_mut().revision = "r2".into();
    assert!(matches!(
        sup.start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("x")
        ),
        Err(SupervisorError::NoProvider(_))
    ));
    assert!(sup
        .log()
        .iter()
        .any(|e| matches!(e, Event::Disabled { .. })));
    sup.qualify(&ProviderId("a".into())).unwrap();
    assert!(sup
        .start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("x")
        )
        .is_ok());
}

#[test]
fn admission_denies_unknown_stale_declared_and_unhealthy() {
    let (mut sup, _, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    let cases: Vec<(Box<FactsFn>, Denial)> = vec![
        (
            Box::new(|id| AdmissionFacts {
                gpu_free_bytes: unknown(),
                ..healthy(id)
            }),
            Denial::Unobserved {
                fact: "gpu_free_bytes",
            },
        ),
        (
            Box::new(|id| AdmissionFacts {
                gpu_free_bytes: Fact::Declared {
                    value: 8 * GIB,
                    source: "spec sheet".into(),
                },
                ..healthy(id)
            }),
            Denial::Unobserved {
                fact: "gpu_free_bytes",
            },
        ),
        (
            Box::new(|id| AdmissionFacts {
                sampled_at_ms: 0,
                ttl_ms: 10,
                ..healthy(id)
            }),
            Denial::Stale {
                age_ms: 100,
                ttl_ms: 10,
            },
        ),
        (
            Box::new(|id| AdmissionFacts {
                driver_health: Fact::HostObserved {
                    value: DriverHealth::Xid(79),
                    source: "dmesg".into(),
                    observed_at_ms: 100,
                },
                ..healthy(id)
            }),
            Denial::DriverUnhealthy(DriverHealth::Xid(79)),
        ),
        (
            Box::new(|id| AdmissionFacts {
                gpu_free_bytes: Fact::HostObserved {
                    value: GIB,
                    source: "nvml".into(),
                    observed_at_ms: 100,
                },
                ..healthy(id)
            }),
            Denial::InsufficientGpu {
                free: GIB,
                required: 4 * GIB,
            },
        ),
    ];
    for (facts, expected) in cases {
        match sup.start(
            &ctx(&selection("m", 1024), &intent("m"), facts.as_ref()),
            gen("x"),
        ) {
            Err(SupervisorError::Denied { denial, .. }) => assert_eq!(denial, expected),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_b_a_switching_unloads_before_loading_and_logs_every_step() {
    let (mut sup, sa, sb) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    sup.qualify(&ProviderId("b".into())).unwrap();
    let r1 = sup
        .start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("1"),
        )
        .unwrap();
    assert_eq!(
        (r1.provider.0.as_str(), r1.loaded, r1.switched_from.clone()),
        ("a", true, None)
    );
    let r2 = sup
        .start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("2"),
        )
        .unwrap();
    assert!(!r2.loaded, "same model on the active engine is reused");
    let r3 = sup
        .start(
            &ctx(&selection("m", 16384), &intent("m"), &healthy),
            gen("3"),
        )
        .unwrap();
    assert_eq!(r3.provider, ProviderId("b".into()));
    assert_eq!(r3.switched_from, Some(ProviderId("a".into())));
    assert!(sa.borrow().loaded.is_none(), "A unloaded before B loaded");
    let r4 = sup
        .start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("4"),
        )
        .unwrap();
    assert_eq!(r4.provider, ProviderId("a".into()));
    assert_eq!((sa.borrow().loads, sa.borrow().unloads), (2, 1));
    assert_eq!((sb.borrow().loads, sb.borrow().unloads), (1, 1));
    let unloads = sup
        .log()
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::Unloaded {
                    receipt: Some(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(unloads, 2);
    assert_eq!(
        sup.state(&ProviderId("b".into())),
        Some(InstanceState::Stopped)
    );
    assert_eq!(
        sup.state(&ProviderId("a".into())),
        Some(InstanceState::Serving)
    );
}

#[test]
fn a_crashing_engine_is_isolated_and_others_keep_serving() {
    let (mut sup, sa, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    sup.qualify(&ProviderId("b".into())).unwrap();
    sa.borrow_mut().fail_generate = true;
    match sup.start(
        &ctx(&selection("m", 1024), &intent("m"), &healthy),
        gen("x"),
    ) {
        Err(SupervisorError::Provider { provider, .. }) => assert_eq!(provider.0, "a"),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        sup.state(&ProviderId("a".into())),
        Some(InstanceState::Stopped)
    );
    assert!(
        sa.borrow().loaded.is_none(),
        "faulted instance was unloaded, no orphan"
    );
    assert!(sup.active().is_none());
    assert!(sup.log().iter().any(|e| matches!(e, Event::Faulted { .. })));
    let r = sup
        .start(
            &ctx(&selection("m", 16384), &intent("m"), &healthy),
            gen("y"),
        )
        .unwrap();
    assert_eq!(r.provider.0, "b");
}

#[test]
fn failed_semantic_probe_never_serves() {
    let (mut sup, sa, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    sa.borrow_mut().probe_passes = false;
    assert!(matches!(
        sup.start(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("x")
        ),
        Err(SupervisorError::ProbeFailed { .. })
    ));
    assert!(sa.borrow().loaded.is_none());
}

#[derive(Default)]
struct Collect(Vec<StreamEvent>);
impl TokenSink for Collect {
    fn on_event(&mut self, e: StreamEvent) -> SinkControl {
        self.0.push(e);
        SinkControl::Continue
    }
}

#[test]
fn streaming_and_cancel_go_through_the_selected_engine_only() {
    let (mut sup, _, _) = supervisor();
    sup.qualify(&ProviderId("a".into())).unwrap();
    sup.qualify(&ProviderId("b".into())).unwrap();
    let mut sink = Collect::default();
    assert!(matches!(
        sup.start_stream(
            &ctx(&selection("m", 1024), &intent("m"), &healthy),
            gen("hi"),
            &mut sink
        ),
        Err(SupervisorError::NotStreaming(_))
    ));
    let (id, rid) = sup
        .start_stream(
            &ctx(&selection("m", 16384), &intent("m"), &healthy),
            gen("hey"),
            &mut sink,
        )
        .unwrap();
    assert_eq!(id.0, "b");
    assert_eq!(sink.0.len(), 3);
    assert_eq!(sup.poll(&id, rid).unwrap(), PollOutcome::Pending);
    sup.cancel(&id, rid).unwrap();
    sup.cancel(&id, rid).unwrap();
    assert_eq!(sup.poll(&id, rid).unwrap(), PollOutcome::Cancelled);
    let _ = StopReason::Cancelled;
}

#[test]
fn registration_rejects_identity_mismatch() {
    let mut sup = Supervisor::new(AdmissionPolicy {
        min_gpu_free_bytes: 0,
    });
    let (a, _) = stub("a", false);
    assert!(matches!(
        sup.register(entry("a", "8192", "other-revision"), a),
        Err(SupervisorError::IdentityMismatch { .. })
    ));
}
