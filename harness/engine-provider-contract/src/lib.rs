//! Executable reference model for gnostral.rs EngineProvider v0.
//!
//! This crate is intentionally dependency-free. It models authority and
//! evidence boundaries; it is not an inference runtime.

pub mod q013;
pub mod sha256;
pub mod v1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fact<T> {
    Unknown {
        reason: String,
    },
    Declared {
        value: T,
        source: String,
    },
    EngineReported {
        value: T,
        source: String,
        observed_at_ms: u64,
    },
    HostObserved {
        value: T,
        source: String,
        observed_at_ms: u64,
    },
    Derived {
        value: T,
        method: String,
        observed_at_ms: u64,
    },
}

impl<T> Fact<T> {
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineTransport {
    InProcess,
    LocalServer,
    LocalCli,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineIdentity {
    pub name: String,
    pub revision: Option<String>,
    pub binary_sha256: Option<String>,
    pub transport: EngineTransport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    pub name: String,
    pub sha256: Option<String>,
    pub format: String,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredCapability {
    pub key: String,
    pub value: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationStatus {
    Qualified,
    Rejected,
    Blocked,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedCapability {
    pub key: String,
    pub value: Option<String>,
    pub status: QualificationStatus,
    pub profile: String,
    pub evidence_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingMode {
    Persistent,
    OneShot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessLevel {
    Control,
    Registered,
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceState {
    Offline,
    Loading,
    Ready(ReadinessLevel),
    Serving,
    Draining,
    Unloading,
    Stopped,
    Faulted,
}

pub fn valid_transition(from: InstanceState, to: InstanceState) -> bool {
    use InstanceState::*;
    use ReadinessLevel::*;

    if to == Faulted {
        return from != Stopped;
    }

    matches!(
        (from, to),
        (Offline, Loading)
            | (Loading, Ready(Control))
            | (Loading, Ready(Registered))
            | (Loading, Ready(Semantic))
            | (Ready(Control), Ready(Registered))
            | (Ready(Control), Ready(Semantic))
            | (Ready(Registered), Ready(Semantic))
            | (Ready(Semantic), Serving)
            | (Ready(Semantic), Draining)
            | (Serving, Draining)
            | (Draining, Unloading)
            | (Ready(Control), Unloading)
            | (Ready(Registered), Unloading)
            | (Ready(Semantic), Unloading)
            | (Loading, Unloading)
            | (Unloading, Stopped)
            | (Faulted, Unloading)
            | (Faulted, Stopped)
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResourceEnvelope {
    pub max_gpu_bytes: Option<u64>,
    pub min_gpu_reserve_bytes: Option<u64>,
    pub max_host_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadIntent {
    pub model: ModelArtifact,
    pub requested_context_tokens: Option<u64>,
    pub requested_concurrency: Option<u32>,
    pub resources: ResourceEnvelope,
    pub serving_mode: ServingMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelHandle(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticProbeReceipt {
    pub handle: ModelHandle,
    pub probe_id: String,
    pub passed: bool,
    pub observed_output: String,
    pub engine: EngineIdentity,
    pub model: ModelArtifact,
    pub completed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeObservation {
    pub handle: ModelHandle,
    pub gpu_resident_bytes: Fact<u64>,
    pub host_resident_bytes: Fact<u64>,
    pub kv_bytes: Fact<u64>,
    pub queue_depth: Fact<u64>,
    pub active_requests: Fact<u64>,
    pub driver_health: Fact<String>,
    /// Human/machine-readable engine-selected placement summary.
    /// Observation only: this field cannot command placement.
    pub placement_summary: Fact<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnloadReceipt {
    pub handle: ModelHandle,
    pub engine_acknowledged: bool,
    pub completed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReclaimReceipt {
    pub handle: ModelHandle,
    pub ambient_gpu_bytes: Fact<u64>,
    pub final_gpu_bytes: Fact<u64>,
    pub within_ambient_tolerance: Fact<bool>,
    pub reclaim_latency_ms: Fact<u64>,
    pub driver_health: Fact<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthSnapshot {
    pub state: InstanceState,
    pub semantic_ready: bool,
    pub observation: RuntimeObservation,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub priority: u8,
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationReceipt {
    pub text: String,
    pub prompt_tokens: Fact<u64>,
    pub completion_tokens: Fact<u64>,
    pub ttft_ms: Fact<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    pub message: String,
}

/// Reference boundary only. Async/stream transport bindings are deliberately
/// left to concrete adapters.
pub trait EngineProvider {
    fn identity(&self) -> EngineIdentity;
    fn declared_capabilities(&self) -> Vec<DeclaredCapability>;
    fn qualified_capabilities(&self) -> Vec<QualifiedCapability>;

    fn load(&mut self, intent: LoadIntent) -> Result<ModelHandle, ProviderError>;
    fn state(&self, handle: &ModelHandle) -> Result<InstanceState, ProviderError>;
    fn semantic_probe(
        &mut self,
        handle: &ModelHandle,
        probe_id: &str,
    ) -> Result<SemanticProbeReceipt, ProviderError>;
    fn generate(
        &mut self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<GenerationReceipt, ProviderError>;
    fn observe(&self, handle: &ModelHandle) -> Result<RuntimeObservation, ProviderError>;
    fn drain(&mut self, handle: &ModelHandle) -> Result<(), ProviderError>;
    fn unload(&mut self, handle: &ModelHandle) -> Result<UnloadReceipt, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> ModelArtifact {
        ModelArtifact {
            name: name.into(),
            sha256: Some("sha256".into()),
            format: "GGUF".into(),
            bytes: None,
        }
    }

    #[test]
    fn mistral_bonsai_declared_and_qualified_capacity_can_disagree_without_data_loss() {
        let declared = DeclaredCapability {
            key: "context_tokens".into(),
            value: "4096".into(),
            source: "engine_config".into(),
        };
        let qualified = QualifiedCapability {
            key: "context_tokens".into(),
            value: Some("384".into()),
            status: QualificationStatus::Qualified,
            profile: "CONTROL".into(),
            evidence_ref: "q002".into(),
        };

        assert_eq!(declared.value, "4096");
        assert_eq!(qualified.value.as_deref(), Some("384"));
    }

    #[test]
    fn one_shot_cli_can_skip_control_and_registration_readiness() {
        assert!(valid_transition(
            InstanceState::Loading,
            InstanceState::Ready(ReadinessLevel::Semantic)
        ));
        assert!(valid_transition(
            InstanceState::Ready(ReadinessLevel::Semantic),
            InstanceState::Draining
        ));
    }

    #[test]
    fn unknown_observation_is_first_class() {
        let fact: Fact<u64> = Fact::Unknown {
            reason: "engine exposes no KV byte telemetry".into(),
        };
        assert!(fact.is_unknown());
    }

    #[test]
    fn pulsar_tiered_residency_fits_without_generic_placement_commands() {
        let observation = RuntimeObservation {
            handle: ModelHandle("pulsar-qwen".into()),
            gpu_resident_bytes: Fact::HostObserved {
                value: 7_143 * 1024 * 1024,
                source: "nvidia-smi".into(),
                observed_at_ms: 1,
            },
            host_resident_bytes: Fact::Unknown {
                reason: "host cache percentage is engine-specific".into(),
            },
            kv_bytes: Fact::Unknown {
                reason: "not exposed".into(),
            },
            queue_depth: Fact::Unknown {
                reason: "one-shot CLI".into(),
            },
            active_requests: Fact::Declared {
                value: 1,
                source: "one-shot workload".into(),
            },
            driver_health: Fact::HostObserved {
                value: "NO_XID".into(),
                source: "kernel_log".into(),
                observed_at_ms: 1,
            },
            placement_summary: Fact::EngineReported {
                value: "VRAM cache + host cache".into(),
                source: "Pulsar profile".into(),
                observed_at_ms: 1,
            },
        };
        assert!(!observation.gpu_resident_bytes.is_unknown());
    }

    #[test]
    fn zllm_declared_auto_placement_can_remain_blocked() {
        let declared = DeclaredCapability {
            key: "cpu_cuda_auto_placement".into(),
            value: "qwen3.6/qwen3.8".into(),
            source: "zllm pinned source".into(),
        };
        let qualified = QualifiedCapability {
            key: "cpu_cuda_auto_placement".into(),
            value: None,
            status: QualificationStatus::Blocked,
            profile: "CONTROL/DESKTOP-PRESSURE".into(),
            evidence_ref: "q005".into(),
        };
        assert_eq!(declared.key, qualified.key);
        assert_eq!(qualified.status, QualificationStatus::Blocked);
    }

    #[test]
    fn unload_ack_and_physical_reclaim_are_distinct_receipts() {
        let handle = ModelHandle("bonsai".into());
        let unload = UnloadReceipt {
            handle: handle.clone(),
            engine_acknowledged: true,
            completed_at_ms: 10,
        };
        let reclaim = ReclaimReceipt {
            handle,
            ambient_gpu_bytes: Fact::HostObserved {
                value: 1_200,
                source: "gpu sampler".into(),
                observed_at_ms: 0,
            },
            final_gpu_bytes: Fact::HostObserved {
                value: 1_210,
                source: "gpu sampler".into(),
                observed_at_ms: 20,
            },
            within_ambient_tolerance: Fact::HostObserved {
                value: true,
                source: "qualification harness".into(),
                observed_at_ms: 20,
            },
            reclaim_latency_ms: Fact::Derived {
                value: 238,
                method: "exit_to_ambient".into(),
                observed_at_ms: 20,
            },
            driver_health: Fact::HostObserved {
                value: "NO_XID".into(),
                source: "kernel_log".into(),
                observed_at_ms: 20,
            },
        };

        assert!(unload.engine_acknowledged);
        assert_ne!(unload.completed_at_ms, 20);
        assert!(matches!(
            reclaim.within_ambient_tolerance,
            Fact::HostObserved { value: true, .. }
        ));
    }

    #[test]
    fn load_intent_expresses_budget_not_layer_placement() {
        let intent = LoadIntent {
            model: model("example"),
            requested_context_tokens: Some(4096),
            requested_concurrency: Some(4),
            resources: ResourceEnvelope {
                max_gpu_bytes: Some(7_000_000_000),
                min_gpu_reserve_bytes: Some(512_000_000),
                max_host_bytes: None,
            },
            serving_mode: ServingMode::Persistent,
        };

        assert_eq!(intent.requested_concurrency, Some(4));
        assert_eq!(intent.resources.min_gpu_reserve_bytes, Some(512_000_000));
    }
}
