//! EngineProvider v1 — draft, additive to v0 (Q-024a).
//!
//! v0 remains the qualified surface. This module adds only what one supervisor
//! needs to drive **several engines** without gaining materialization authority:
//! a selection-only registry, typed capability negotiation, start/poll streaming,
//! an explicit error taxonomy, freshness-stamped admission facts and a mechanical
//! conformance harness.
//!
//! Deliberately absent (the authority test): any placement, KV, batching, kernel
//! or expert-selection method; any discovery or provider-initiated registration;
//! any async runtime. Callers wrap start/poll in their own runtime.
//!
//! No type here is an implementation or a qualification. A provider that
//! implements this module is still unqualified until evidence says otherwise.

use crate::{
    valid_transition, DeclaredCapability, EngineProvider, EngineTransport, Fact, GenerationReceipt,
    GenerationRequest, InstanceState, ModelArtifact, ModelHandle, QualificationStatus,
    QualifiedCapability,
};

/// Stable identity of a provider entry. Supervisor-owned; never provider-registered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderId(pub String);

/// Typed capability keys. Strings are deliberately not accepted: a typo must fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CapabilityKey {
    Transport,
    ServingMode,
    Streaming,
    Cancellation,
    ConstrainedDecoding,
    MultiChoiceN,
    PromptLogprobs,
    TokenLogprobs,
    MaxContextTokens,
    MaxConcurrency,
    KvBytes,
    EngineRevision,
}

impl CapabilityKey {
    pub const ALL: [CapabilityKey; 12] = [
        CapabilityKey::Transport,
        CapabilityKey::ServingMode,
        CapabilityKey::Streaming,
        CapabilityKey::Cancellation,
        CapabilityKey::ConstrainedDecoding,
        CapabilityKey::MultiChoiceN,
        CapabilityKey::PromptLogprobs,
        CapabilityKey::TokenLogprobs,
        CapabilityKey::MaxContextTokens,
        CapabilityKey::MaxConcurrency,
        CapabilityKey::KvBytes,
        CapabilityKey::EngineRevision,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            CapabilityKey::Transport => "transport",
            CapabilityKey::ServingMode => "serving_mode",
            CapabilityKey::Streaming => "streaming",
            CapabilityKey::Cancellation => "cancellation",
            CapabilityKey::ConstrainedDecoding => "constrained_decoding",
            CapabilityKey::MultiChoiceN => "multi_choice_n",
            CapabilityKey::PromptLogprobs => "prompt_logprobs",
            CapabilityKey::TokenLogprobs => "token_logprobs",
            CapabilityKey::MaxContextTokens => "max_context_tokens",
            CapabilityKey::MaxConcurrency => "max_concurrency",
            CapabilityKey::KvBytes => "kv_bytes",
            CapabilityKey::EngineRevision => "engine_revision",
        }
    }
}

impl std::fmt::Display for CapabilityKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A typed capability value. Numeric comparisons are numeric; no string capacities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityValue {
    Bool(bool),
    UInt(u64),
    Text(String),
    Enum(u16),
}

impl CapabilityValue {
    fn parse(raw: &str) -> Option<Self> {
        if raw == "true" {
            return Some(Self::Bool(true));
        }
        if raw == "false" {
            return Some(Self::Bool(false));
        }
        if let Ok(value) = raw.parse::<u64>() {
            return Some(Self::UInt(value));
        }
        if raw.is_empty() {
            return None;
        }
        Some(Self::Text(raw.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRequirement {
    pub key: CapabilityKey,
    /// Numeric minimum; only meaningful for `UInt` requirements.
    pub at_least: Option<CapabilityValue>,
    /// Exact value for booleans, enums and textual facts.
    pub exact: Option<CapabilityValue>,
}

impl CapabilityRequirement {
    pub fn new(key: CapabilityKey) -> Self {
        Self {
            key,
            at_least: None,
            exact: None,
        }
    }

    pub fn at_least(mut self, value: CapabilityValue) -> Self {
        self.at_least = Some(value);
        self
    }

    pub fn exact(mut self, value: CapabilityValue) -> Self {
        self.exact = Some(value);
        self
    }
}

/// Result of comparing a selection request with one provider's **qualified** facts.
/// `Unknown` is distinct and never satisfies a requirement (v0 `Fact::Unknown` rule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Negotiation {
    Accepted,
    Unsatisfied(Vec<CapabilityKey>),
    Unknown(Vec<CapabilityKey>),
    /// A declared/qualified record has the wrong type or the two disagree where use is forbidden.
    Conflict(Vec<CapabilityKey>),
}

impl Negotiation {
    pub fn satisfied(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

/// One supervisor-owned registry entry: identity, qualification profile, evidence-backed facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderEntry {
    pub id: ProviderId,
    pub identity: crate::EngineIdentity,
    /// Qualification profile the entries below were produced under.
    pub profile: String,
    pub qualified: Vec<QualifiedCapability>,
    pub declared: Vec<DeclaredCapability>,
    pub enabled: bool,
}

impl ProviderEntry {
    fn qualified_value(&self, key: CapabilityKey) -> Option<&QualifiedCapability> {
        self.qualified
            .iter()
            .find(|capability| capability.key == key.as_str())
    }
}

/// Static selection surface. No discovery, no dynamic registration, no provider-to-provider calls.
#[derive(Debug, Default, Clone)]
pub struct ProviderRegistry {
    entries: Vec<ProviderEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateProviderId(ProviderId),
    EmptyProviderId,
}

impl ProviderRegistry {
    pub fn new(entries: Vec<ProviderEntry>) -> Result<Self, RegistryError> {
        let mut registry = Self {
            entries: Vec::with_capacity(entries.len()),
        };
        for entry in entries {
            registry.register(entry)?;
        }
        Ok(registry)
    }

    pub fn register(&mut self, entry: ProviderEntry) -> Result<(), RegistryError> {
        if entry.id.0.trim().is_empty() {
            return Err(RegistryError::EmptyProviderId);
        }
        if self.entries.iter().any(|known| known.id == entry.id) {
            return Err(RegistryError::DuplicateProviderId(entry.id));
        }
        self.entries.push(entry);
        Ok(())
    }

    pub fn entries(&self) -> &[ProviderEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Select a provider for an intent. Never loads, never falls back silently, never
    /// mutates a provider. The receipt is a decision record, not a load grant.
    pub fn select(&self, request: &SelectionRequest, decided_at_ms: u64) -> SelectionReceipt {
        let mut candidates: Vec<(&ProviderEntry, Negotiation)> = Vec::new();
        let mut rejected: Vec<(ProviderId, SelectionRejection)> = Vec::new();

        for entry in &self.entries {
            if !entry.enabled {
                rejected.push((entry.id.clone(), SelectionRejection::Disabled));
                continue;
            }
            if entry.profile != request.profile {
                rejected.push((entry.id.clone(), SelectionRejection::ProfileMismatch));
                continue;
            }
            match negotiate(entry, &request.required) {
                Negotiation::Accepted => candidates.push((entry, Negotiation::Accepted)),
                other => rejected.push((entry.id.clone(), SelectionRejection::Capabilities(other))),
            }
        }

        // Deterministic preference: required order is already fixed; `preferred` breaks ties
        // by which candidate holds the earlier preferred capability. Registry order is final.
        candidates.sort_by_key(|(entry, _)| {
            request
                .preferred
                .iter()
                .position(|key| entry.qualified_value(*key).is_some())
                .unwrap_or(usize::MAX)
        });

        let chosen = candidates.first().map(|(entry, _)| entry.id.clone());
        SelectionReceipt {
            chosen,
            rejected,
            decided_at_ms,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionRequest {
    pub required: Vec<CapabilityRequirement>,
    pub preferred: Vec<CapabilityKey>,
    pub model: ModelArtifact,
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionReceipt {
    pub chosen: Option<ProviderId>,
    pub rejected: Vec<(ProviderId, SelectionRejection)>,
    pub decided_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionRejection {
    Disabled,
    ProfileMismatch,
    Capabilities(Negotiation),
}

/// Compare a request against **qualified** facts only. Declared facts are never read here.
pub fn negotiate(entry: &ProviderEntry, required: &[CapabilityRequirement]) -> Negotiation {
    let mut unsatisfied = Vec::new();
    let mut unknown = Vec::new();
    let mut conflict = Vec::new();

    for requirement in required {
        let Some(capability) = entry.qualified_value(requirement.key) else {
            unknown.push(requirement.key);
            continue;
        };
        if capability.status != QualificationStatus::Qualified {
            unsatisfied.push(requirement.key);
            continue;
        }
        let Some(raw) = capability.value.as_deref() else {
            unsatisfied.push(requirement.key);
            continue;
        };
        let Some(value) = CapabilityValue::parse(raw) else {
            conflict.push(requirement.key);
            continue;
        };

        if let Some(exact) = &requirement.exact {
            if &value != exact {
                unsatisfied.push(requirement.key);
                continue;
            }
        }
        if let Some(at_least) = &requirement.at_least {
            match (at_least, &value) {
                (CapabilityValue::UInt(minimum), CapabilityValue::UInt(actual)) => {
                    if actual < minimum {
                        unsatisfied.push(requirement.key);
                        continue;
                    }
                }
                _ => {
                    conflict.push(requirement.key);
                    continue;
                }
            }
        }
    }

    if !conflict.is_empty() {
        Negotiation::Conflict(conflict)
    } else if !unknown.is_empty() {
        Negotiation::Unknown(unknown)
    } else if !unsatisfied.is_empty() {
        Negotiation::Unsatisfied(unsatisfied)
    } else {
        Negotiation::Accepted
    }
}

// ---------------------------------------------------------------------------
// Error taxonomy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderErrorKind {
    IdentityMismatch,
    UnknownModel,
    AdmissionRejected,
    CapabilityUnsatisfied,
    NotReady,
    Busy,
    Cancelled,
    DeadlineExceeded,
    EngineFault,
    DriverFault,
    TransportFault,
    ProtocolViolation,
    Internal,
}

impl ProviderErrorKind {
    /// There is no `Other`: an unclassifiable adapter must say `Internal` and look unqualified.
    pub const ALL: [ProviderErrorKind; 13] = [
        ProviderErrorKind::IdentityMismatch,
        ProviderErrorKind::UnknownModel,
        ProviderErrorKind::AdmissionRejected,
        ProviderErrorKind::CapabilityUnsatisfied,
        ProviderErrorKind::NotReady,
        ProviderErrorKind::Busy,
        ProviderErrorKind::Cancelled,
        ProviderErrorKind::DeadlineExceeded,
        ProviderErrorKind::EngineFault,
        ProviderErrorKind::DriverFault,
        ProviderErrorKind::TransportFault,
        ProviderErrorKind::ProtocolViolation,
        ProviderErrorKind::Internal,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ProviderErrorKind::IdentityMismatch => "identity_mismatch",
            ProviderErrorKind::UnknownModel => "unknown_model",
            ProviderErrorKind::AdmissionRejected => "admission_rejected",
            ProviderErrorKind::CapabilityUnsatisfied => "capability_unsatisfied",
            ProviderErrorKind::NotReady => "not_ready",
            ProviderErrorKind::Busy => "busy",
            ProviderErrorKind::Cancelled => "cancelled",
            ProviderErrorKind::DeadlineExceeded => "deadline_exceeded",
            ProviderErrorKind::EngineFault => "engine_fault",
            ProviderErrorKind::DriverFault => "driver_fault",
            ProviderErrorKind::TransportFault => "transport_fault",
            ProviderErrorKind::ProtocolViolation => "protocol_violation",
            ProviderErrorKind::Internal => "internal",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    /// Explicit; callers must not infer retry policy from `kind` or `message`.
    pub retryable: bool,
    pub retry_after_ms: Fact<u64>,
    pub observed_at_ms: u64,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable,
            retry_after_ms: Fact::Unknown {
                reason: "no retry estimate reported".into(),
            },
            observed_at_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming (start/poll; object-safe; no async runtime)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    Length,
    StopSequence,
    Cancelled,
    Deadline,
    EngineFault,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Token {
        index: u32,
        token_id: Option<u32>,
        text_delta: String,
        logprob: Option<f32>,
    },
    Usage {
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
    },
    Finished {
        stop_reason: StopReason,
        receipt: GenerationReceipt,
    },
    Failed {
        error: ProviderError,
    },
}

impl StreamEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished { .. } | Self::Failed { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkControl {
    Continue,
    Pause,
}

pub trait TokenSink {
    fn on_event(&mut self, event: StreamEvent) -> SinkControl;
}

#[derive(Debug, Clone, PartialEq)]
pub enum PollOutcome {
    Pending,
    Finished(GenerationReceipt),
    Failed(ProviderError),
    Cancelled,
}

/// Streaming extension. Implementing it is optional; a one-shot CLI remains a valid v1 provider.
pub trait EngineProviderStream: EngineProvider {
    fn start_generate(
        &mut self,
        handle: &ModelHandle,
        request: GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<RequestId, ProviderError>;

    /// Must not block. A transport that cannot poll non-blockingly does not qualify `Streaming`.
    fn poll(&mut self, id: RequestId) -> Result<PollOutcome, ProviderError>;

    /// Idempotent; never faults the instance when the id is unknown/finished.
    fn cancel(&mut self, id: RequestId) -> Result<(), ProviderError>;
}

/// Check the wire contract of a stream: monotonic token indices from 0, no events after
/// a terminal event, and a token index that never exceeds `u32`.
pub fn validate_stream_events(events: &[StreamEvent]) -> Result<(), String> {
    let mut expected_index: u32 = 0;
    let mut terminal = false;
    for event in events {
        if terminal {
            return Err("event after a terminal Finished/Failed".to_string());
        }
        match event {
            StreamEvent::Token { index, .. } => {
                if *index != expected_index {
                    return Err(format!(
                        "non-monotonic token index: expected {expected_index}, saw {index}"
                    ));
                }
                expected_index = expected_index
                    .checked_add(1)
                    .ok_or_else(|| "token index overflow".to_string())?;
            }
            StreamEvent::Usage { .. } => {}
            StreamEvent::Finished { .. } | StreamEvent::Failed { .. } => terminal = true,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Admission facts (provider reports; supervisor decides)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverHealth {
    NoXid,
    Xid(u32),
    Reset,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionFacts {
    /// None before load: a valid, deny-by-default state.
    pub handle: Option<ModelHandle>,
    pub sampled_at_ms: u64,
    pub ttl_ms: u64,
    pub gpu_free_bytes: Fact<u64>,
    pub host_free_bytes: Fact<u64>,
    pub driver_health: Fact<DriverHealth>,
    pub admitted_context_tokens: Fact<u64>,
    pub admitted_concurrency: Fact<u32>,
}

impl AdmissionFacts {
    /// A fact sampled more than `ttl_ms` ago must be treated as `Unknown` by the supervisor.
    pub fn is_fresh(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.sampled_at_ms) <= self.ttl_ms
    }
}

// ---------------------------------------------------------------------------
// Mechanical conformance harness
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceCheck {
    pub id: &'static str,
    pub passed: bool,
    pub detail: String,
}

impl ConformanceCheck {
    fn new(id: &'static str, passed: bool, detail: impl Into<String>) -> Self {
        Self {
            id,
            passed,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    pub checks: Vec<ConformanceCheck>,
    pub passed: bool,
}

/// Mechanical checks only: no model, no GPU, no load. Profile qualification
/// (semantic readiness, reclaim, driver health) stays where v0 put it: evidence.
pub fn provider_conformance<P: EngineProvider>(provider: &mut P) -> ConformanceReport {
    let mut checks = Vec::new();

    let first = provider.identity();
    let second = provider.identity();
    checks.push(ConformanceCheck::new(
        "identity_stable",
        first == second && !first.name.trim().is_empty(),
        "identity() must be stable across calls and name its engine",
    ));

    let declared = provider.declared_capabilities();
    let qualified = provider.qualified_capabilities();
    let declared_ok = declared
        .iter()
        .all(|capability| !capability.source.trim().is_empty());
    let qualified_ok = qualified.iter().all(|capability| {
        !capability.evidence_ref.trim().is_empty() && !capability.profile.trim().is_empty()
    });
    checks.push(ConformanceCheck::new(
        "declared_and_qualified_stay_separate",
        declared_ok && qualified_ok,
        "declared facts need a source; qualified facts need evidence_ref and profile",
    ));

    let transition_model_ok = valid_transition(InstanceState::Offline, InstanceState::Loading)
        && !valid_transition(InstanceState::Stopped, InstanceState::Serving)
        && !valid_transition(InstanceState::Offline, InstanceState::Serving);
    checks.push(ConformanceCheck::new(
        "lifecycle_model_is_enforced",
        transition_model_ok,
        "Serving is unreachable except through the documented transition graph",
    ));

    let unknown: Fact<u64> = Fact::Unknown {
        reason: "conformance probe".into(),
    };
    checks.push(ConformanceCheck::new(
        "unknown_survives_as_fact",
        unknown.is_unknown(),
        "a missing observation must not become a default",
    ));

    let labels: Vec<&str> = ProviderErrorKind::ALL
        .iter()
        .map(|kind| kind.label())
        .collect();
    let unique = labels
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        == labels.len();
    checks.push(ConformanceCheck::new(
        "error_taxonomy_is_classifiable",
        ProviderErrorKind::ALL.len() == 13 && unique && !labels.contains(&"other"),
        "13 explicit kinds, unique labels, no Other",
    ));

    let synthetic = vec![
        StreamEvent::Token {
            index: 0,
            token_id: None,
            text_delta: "a".into(),
            logprob: None,
        },
        StreamEvent::Finished {
            stop_reason: StopReason::Eos,
            receipt: GenerationReceipt {
                text: "a".into(),
                prompt_tokens: Fact::Unknown {
                    reason: "probe".into(),
                },
                completion_tokens: Fact::Unknown {
                    reason: "probe".into(),
                },
                ttft_ms: Fact::Unknown {
                    reason: "probe".into(),
                },
            },
        },
    ];
    let stream_ok = validate_stream_events(&synthetic).is_ok()
        && validate_stream_events(&[
            StreamEvent::Token {
                index: 1,
                token_id: None,
                text_delta: "b".into(),
                logprob: None,
            },
            synthetic[1].clone(),
        ])
        .is_err();
    checks.push(ConformanceCheck::new(
        "stream_contract_is_checkable",
        stream_ok,
        "monotonic indices and exactly-one terminal event",
    ));

    let mut registry = ProviderRegistry::default();
    let entry = ProviderEntry {
        id: ProviderId("probe".into()),
        identity: first,
        profile: "conformance".into(),
        qualified: Vec::new(),
        declared: Vec::new(),
        enabled: true,
    };
    let duplicate_rejected =
        registry.register(entry.clone()).is_ok() && registry.register(entry).is_err();
    checks.push(ConformanceCheck::new(
        "registry_rejects_duplicate_ids",
        duplicate_rejected,
        "selection surface must not contain ambiguous identities",
    ));

    let passed = checks.iter().all(|check| check.passed);
    ConformanceReport { checks, passed }
}

/// Capability values derived from a v0 identity, for registry construction.
pub fn transport_and_revision(identity: &crate::EngineIdentity) -> Vec<QualifiedCapability> {
    let transport = match identity.transport {
        EngineTransport::InProcess => "in_process",
        EngineTransport::LocalServer => "local_server",
        EngineTransport::LocalCli => "local_cli",
        EngineTransport::Remote => "remote",
    };
    let mut capabilities = vec![QualifiedCapability {
        key: CapabilityKey::Transport.as_str().into(),
        value: Some(transport.into()),
        status: QualificationStatus::Qualified,
        profile: "identity".into(),
        evidence_ref: "identity".into(),
    }];
    if let Some(revision) = &identity.revision {
        capabilities.push(QualifiedCapability {
            key: CapabilityKey::EngineRevision.as_str().into(),
            value: Some(revision.clone()),
            status: QualificationStatus::Qualified,
            profile: "identity".into(),
            evidence_ref: "identity".into(),
        });
    }
    capabilities
}
