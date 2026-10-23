//! Q-024a: EngineProvider v1 draft is additive and mechanically testable.

use gnostral_engine_provider_contract::v1::*;
use gnostral_engine_provider_contract::{
    DeclaredCapability, EngineIdentity, EngineProvider, EngineTransport, Fact, GenerationReceipt,
    GenerationRequest, InstanceState, ModelArtifact, ModelHandle, QualificationStatus,
    QualifiedCapability, ReadinessLevel, RuntimeObservation, SemanticProbeReceipt, UnloadReceipt,
};

const SHA: &str = "64eec222ccefff4cd4c0067bd83348f1d22eebda27d0d857e352ee090cf19723";

fn identity(name: &str) -> EngineIdentity {
    EngineIdentity {
        name: name.into(),
        revision: Some("0bed3b4ce".into()),
        binary_sha256: Some(SHA.into()),
        transport: EngineTransport::LocalServer,
    }
}

fn qualified(key: CapabilityKey, value: &str) -> QualifiedCapability {
    QualifiedCapability {
        key: key.as_str().into(),
        value: Some(value.into()),
        status: QualificationStatus::Qualified,
        profile: "p".into(),
        evidence_ref: "q013".into(),
    }
}

fn entry(id: &str, profile: &str, capabilities: Vec<QualifiedCapability>) -> ProviderEntry {
    ProviderEntry {
        id: ProviderId(id.into()),
        identity: identity("engine"),
        profile: profile.into(),
        qualified: capabilities,
        declared: Vec::new(),
        enabled: true,
    }
}

fn model() -> ModelArtifact {
    ModelArtifact {
        name: "test".into(),
        sha256: Some(SHA.into()),
        format: "GGUF".into(),
        bytes: None,
    }
}

fn request(profile: &str, required: Vec<CapabilityRequirement>) -> SelectionRequest {
    SelectionRequest {
        required,
        preferred: Vec::new(),
        model: model(),
        profile: profile.into(),
    }
}

fn unknown_fact<T>() -> Fact<T> {
    Fact::Unknown {
        reason: "test".into(),
    }
}

#[test]
fn select_uses_qualified_not_declared_capabilities() {
    let mut declared_only = entry("declared", "p", Vec::new());
    declared_only.declared = vec![DeclaredCapability {
        key: CapabilityKey::MaxContextTokens.as_str().into(),
        value: "8192".into(),
        source: "vendor claim".into(),
    }];
    let qualified_entry = entry(
        "qualified",
        "p",
        vec![qualified(CapabilityKey::MaxContextTokens, "4096")],
    );
    let registry = ProviderRegistry::new(vec![declared_only, qualified_entry]).unwrap();
    let receipt = registry.select(
        &request(
            "p",
            vec![CapabilityRequirement::new(CapabilityKey::MaxContextTokens)
                .at_least(CapabilityValue::UInt(2048))],
        ),
        1,
    );
    assert_eq!(receipt.chosen, Some(ProviderId("qualified".into())));
    assert!(receipt
        .rejected
        .iter()
        .any(|(id, reason)| id.0 == "declared"
            && matches!(
                reason,
                SelectionRejection::Capabilities(Negotiation::Unknown(_))
            )));
}

#[test]
fn negotiation_is_typed_and_numeric() {
    let provider = entry(
        "p",
        "prof",
        vec![qualified(CapabilityKey::MaxContextTokens, "4096")],
    );
    let accepted = negotiate(
        &provider,
        &[CapabilityRequirement::new(CapabilityKey::MaxContextTokens)
            .at_least(CapabilityValue::UInt(4096))],
    );
    assert_eq!(accepted, Negotiation::Accepted);

    let unsatisfied = negotiate(
        &provider,
        &[CapabilityRequirement::new(CapabilityKey::MaxContextTokens)
            .at_least(CapabilityValue::UInt(4097))],
    );
    assert!(matches!(unsatisfied, Negotiation::Unsatisfied(_)));

    let conflict = negotiate(
        &entry(
            std::string::String::new().as_str(),
            "prof",
            vec![qualified(CapabilityKey::MaxContextTokens, "4096 tokens")],
        ),
        &[CapabilityRequirement::new(CapabilityKey::MaxContextTokens)
            .at_least(CapabilityValue::UInt(1))],
    );
    assert!(matches!(conflict, Negotiation::Conflict(_)));
}

#[test]
fn unknown_is_a_distinct_deny_state() {
    let entry = entry("p", "prof", Vec::new());
    let outcome = negotiate(
        &entry,
        &[CapabilityRequirement::new(CapabilityKey::Streaming).exact(CapabilityValue::Bool(true))],
    );
    assert_eq!(
        outcome,
        Negotiation::Unknown(vec![CapabilityKey::Streaming])
    );
    assert!(!outcome.satisfied());
}

#[test]
fn profile_mismatch_and_disabled_are_rejections() {
    let wrong_profile = entry("wrong", "other", Vec::new());
    let mut disabled = entry("disabled", "p", Vec::new());
    disabled.enabled = false;
    let registry = ProviderRegistry::new(vec![wrong_profile, disabled]).unwrap();
    let receipt = registry.select(&request("p", Vec::new()), 2);
    assert_eq!(receipt.chosen, None);
    assert!(receipt
        .rejected
        .iter()
        .any(|(id, reason)| id.0 == "wrong" && *reason == SelectionRejection::ProfileMismatch));
    assert!(receipt
        .rejected
        .iter()
        .any(|(id, reason)| id.0 == "disabled" && *reason == SelectionRejection::Disabled));
}

#[test]
fn preferred_capability_breaks_ties_deterministically() {
    let plain = entry("plain", "p", Vec::new());
    let streamy = entry(
        "streamy",
        "p",
        vec![qualified(CapabilityKey::Streaming, "true")],
    );
    let registry = ProviderRegistry::new(vec![plain, streamy]).unwrap();
    let mut req = request("p", Vec::new());
    req.preferred = vec![CapabilityKey::Streaming];
    assert_eq!(
        registry.select(&req, 3).chosen,
        Some(ProviderId("streamy".into()))
    );
    // No preference: registry order, never timing.
    req.preferred.clear();
    assert_eq!(
        registry.select(&req, 4).chosen,
        Some(ProviderId("plain".into()))
    );
}

#[test]
fn registry_rejects_duplicate_and_empty_ids() {
    let one = entry("x", "p", Vec::new());
    let mut registry = ProviderRegistry::default();
    registry.register(one.clone()).unwrap();
    assert_eq!(
        registry.register(one),
        Err(RegistryError::DuplicateProviderId(ProviderId("x".into())))
    );
    let empty = entry("", "p", Vec::new());
    assert_eq!(
        registry.register(empty),
        Err(RegistryError::EmptyProviderId)
    );
}

#[test]
fn error_taxonomy_has_no_other_and_explicit_retryability() {
    assert_eq!(ProviderErrorKind::ALL.len(), 13);
    let labels: std::collections::BTreeSet<&str> = ProviderErrorKind::ALL
        .iter()
        .map(|kind| kind.label())
        .collect();
    assert_eq!(labels.len(), 13);
    assert!(!labels.contains("other"));
    let error = ProviderError::new(ProviderErrorKind::TransportFault, "socket closed", true);
    assert_eq!(error.kind, ProviderErrorKind::TransportFault);
    assert!(error.retryable);
    assert!(error.retry_after_ms.is_unknown());
}

#[test]
fn stream_events_are_monotonic_and_terminated_once() {
    let receipt = receipt();
    let good = vec![
        StreamEvent::Token {
            index: 0,
            token_id: Some(7),
            text_delta: "a".into(),
            logprob: None,
        },
        StreamEvent::Token {
            index: 1,
            token_id: Some(8),
            text_delta: "b".into(),
            logprob: None,
        },
        StreamEvent::Finished {
            stop_reason: StopReason::Eos,
            receipt,
        },
    ];
    assert!(validate_stream_events(&good).is_ok());

    let mut skipped = good.clone();
    skipped[1] = StreamEvent::Token {
        index: 2,
        token_id: Some(8),
        text_delta: "b".into(),
        logprob: None,
    };
    assert!(validate_stream_events(&skipped).is_err());

    let mut after_terminal = good.clone();
    after_terminal.push(StreamEvent::Token {
        index: 2,
        token_id: Some(9),
        text_delta: "c".into(),
        logprob: None,
    });
    assert!(validate_stream_events(&after_terminal).is_err());
}

#[test]
fn admission_facts_are_freshness_stamped() {
    let facts = AdmissionFacts {
        handle: None,
        sampled_at_ms: 1_000,
        ttl_ms: 500,
        gpu_free_bytes: unknown_fact(),
        host_free_bytes: Fact::HostObserved {
            value: 6_000,
            source: "host".into(),
            observed_at_ms: 1_000,
        },
        driver_health: Fact::HostObserved {
            value: DriverHealth::NoXid,
            source: "kernel log".into(),
            observed_at_ms: 1_000,
        },
        admitted_context_tokens: unknown_fact(),
        admitted_concurrency: unknown_fact(),
    };
    assert!(facts.is_fresh(1_400));
    assert!(!facts.is_fresh(1_600));
    assert!(facts.gpu_free_bytes.is_unknown());
    assert!(facts.handle.is_none());
}

// ---------------------------------------------------------------------------
// A minimal provider used only to exercise the streaming trait and conformance.
// ---------------------------------------------------------------------------

fn receipt() -> GenerationReceipt {
    GenerationReceipt {
        text: "ok".into(),
        prompt_tokens: Fact::Unknown {
            reason: "test".into(),
        },
        completion_tokens: Fact::Unknown {
            reason: "test".into(),
        },
        ttft_ms: Fact::Unknown {
            reason: "test".into(),
        },
    }
}

#[derive(Default)]
struct CollectSink {
    events: Vec<StreamEvent>,
}

impl TokenSink for CollectSink {
    fn on_event(&mut self, event: StreamEvent) -> SinkControl {
        self.events.push(event);
        SinkControl::Continue
    }
}

struct MockProvider {
    identity: EngineIdentity,
    state: InstanceState,
    handle: ModelHandle,
    finished: bool,
}

impl MockProvider {
    fn new() -> Self {
        Self {
            identity: identity("mock"),
            state: InstanceState::Offline,
            handle: ModelHandle("mock-0".into()),
            finished: false,
        }
    }
}

impl EngineProvider for MockProvider {
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }

    fn declared_capabilities(&self) -> Vec<DeclaredCapability> {
        vec![DeclaredCapability {
            key: "streaming".into(),
            value: "true".into(),
            source: "unit test".into(),
        }]
    }

    fn qualified_capabilities(&self) -> Vec<QualifiedCapability> {
        vec![qualified(CapabilityKey::Streaming, "true")]
    }

    fn load(
        &mut self,
        _intent: gnostral_engine_provider_contract::LoadIntent,
    ) -> Result<ModelHandle, gnostral_engine_provider_contract::ProviderError> {
        self.state = InstanceState::Ready(ReadinessLevel::Semantic);
        Ok(self.handle.clone())
    }

    fn state(
        &self,
        _handle: &ModelHandle,
    ) -> Result<InstanceState, gnostral_engine_provider_contract::ProviderError> {
        Ok(self.state)
    }

    fn semantic_probe(
        &mut self,
        _handle: &ModelHandle,
        _probe_id: &str,
    ) -> Result<SemanticProbeReceipt, gnostral_engine_provider_contract::ProviderError> {
        Ok(SemanticProbeReceipt {
            handle: self.handle.clone(),
            probe_id: "probe".into(),
            passed: true,
            observed_output: "ok".into(),
            engine: self.identity.clone(),
            model: model(),
            completed_at_ms: 5,
        })
    }

    fn generate(
        &mut self,
        _handle: &ModelHandle,
        _request: GenerationRequest,
    ) -> Result<GenerationReceipt, gnostral_engine_provider_contract::ProviderError> {
        Ok(receipt())
    }

    fn observe(
        &self,
        handle: &ModelHandle,
    ) -> Result<RuntimeObservation, gnostral_engine_provider_contract::ProviderError> {
        Ok(RuntimeObservation {
            handle: handle.clone(),
            gpu_resident_bytes: unknown_fact(),
            host_resident_bytes: unknown_fact(),
            kv_bytes: unknown_fact(),
            queue_depth: unknown_fact(),
            active_requests: unknown_fact(),
            driver_health: unknown_fact(),
            placement_summary: unknown_fact(),
        })
    }

    fn drain(
        &mut self,
        _handle: &ModelHandle,
    ) -> Result<(), gnostral_engine_provider_contract::ProviderError> {
        self.state = InstanceState::Draining;
        Ok(())
    }

    fn unload(
        &mut self,
        handle: &ModelHandle,
    ) -> Result<UnloadReceipt, gnostral_engine_provider_contract::ProviderError> {
        Ok(UnloadReceipt {
            handle: handle.clone(),
            engine_acknowledged: true,
            completed_at_ms: 6,
        })
    }
}

impl EngineProviderStream for MockProvider {
    fn start_generate(
        &mut self,
        _handle: &ModelHandle,
        _request: GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<RequestId, ProviderError> {
        let _ = sink.on_event(StreamEvent::Token {
            index: 0,
            token_id: Some(1),
            text_delta: "o".into(),
            logprob: None,
        });
        let _ = sink.on_event(StreamEvent::Finished {
            stop_reason: StopReason::Eos,
            receipt: receipt(),
        });
        self.finished = true;
        Ok(RequestId(7))
    }

    fn poll(&mut self, _id: RequestId) -> Result<PollOutcome, ProviderError> {
        Ok(if self.finished {
            PollOutcome::Finished(receipt())
        } else {
            PollOutcome::Pending
        })
    }

    fn cancel(&mut self, _id: RequestId) -> Result<(), ProviderError> {
        // Idempotent and never faults the instance.
        Ok(())
    }
}

#[test]
fn streaming_start_poll_cancel_contract() {
    let mut provider = MockProvider::new();
    let mut sink = CollectSink::default();
    let id = provider
        .start_generate(
            &provider.handle.clone(),
            GenerationRequest {
                prompt: "hi".into(),
                max_tokens: 2,
                priority: 0,
                deadline_ms: None,
            },
            &mut sink,
        )
        .unwrap();
    assert_eq!(id, RequestId(7));
    assert_eq!(sink.events.len(), 2);
    assert!(validate_stream_events(&sink.events).is_ok());
    assert_eq!(provider.poll(id).unwrap(), PollOutcome::Finished(receipt()));
    provider.cancel(RequestId(99)).unwrap();
}

#[test]
fn conformance_passes_a_well_behaved_provider_and_fails_an_anonymous_one() {
    let mut good = MockProvider::new();
    let report = provider_conformance(&mut good);
    assert!(
        report.passed,
        "{:?}",
        report
            .checks
            .iter()
            .filter(|c| !c.passed)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.checks.len(), 7);

    struct Anonymous;
    impl EngineProvider for Anonymous {
        fn identity(&self) -> EngineIdentity {
            EngineIdentity {
                name: String::new(),
                revision: None,
                binary_sha256: None,
                transport: EngineTransport::InProcess,
            }
        }
        fn declared_capabilities(&self) -> Vec<DeclaredCapability> {
            Vec::new()
        }
        fn qualified_capabilities(&self) -> Vec<QualifiedCapability> {
            Vec::new()
        }
        fn load(
            &mut self,
            _intent: gnostral_engine_provider_contract::LoadIntent,
        ) -> Result<ModelHandle, gnostral_engine_provider_contract::ProviderError> {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
        fn state(
            &self,
            _handle: &ModelHandle,
        ) -> Result<InstanceState, gnostral_engine_provider_contract::ProviderError> {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
        fn semantic_probe(
            &mut self,
            _handle: &ModelHandle,
            _probe_id: &str,
        ) -> Result<SemanticProbeReceipt, gnostral_engine_provider_contract::ProviderError>
        {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
        fn generate(
            &mut self,
            _handle: &ModelHandle,
            _request: GenerationRequest,
        ) -> Result<GenerationReceipt, gnostral_engine_provider_contract::ProviderError> {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
        fn observe(
            &self,
            _handle: &ModelHandle,
        ) -> Result<RuntimeObservation, gnostral_engine_provider_contract::ProviderError> {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
        fn drain(
            &mut self,
            _handle: &ModelHandle,
        ) -> Result<(), gnostral_engine_provider_contract::ProviderError> {
            Ok(())
        }
        fn unload(
            &mut self,
            _handle: &ModelHandle,
        ) -> Result<UnloadReceipt, gnostral_engine_provider_contract::ProviderError> {
            Err(gnostral_engine_provider_contract::ProviderError {
                message: "unused".into(),
            })
        }
    }
    let report = provider_conformance(&mut Anonymous);
    assert!(!report.passed);
    assert!(report
        .checks
        .iter()
        .any(|check| check.id == "identity_stable" && !check.passed));
}

#[test]
fn transport_and_revision_capabilities_are_built_from_identity() {
    let capabilities = transport_and_revision(&identity("engine"));
    assert!(capabilities
        .iter()
        .any(|c| c.key == "transport" && c.value.as_deref() == Some("local_server")));
    assert!(capabilities
        .iter()
        .any(|c| c.key == "engine_revision" && c.value.as_deref() == Some("0bed3b4ce")));
}
