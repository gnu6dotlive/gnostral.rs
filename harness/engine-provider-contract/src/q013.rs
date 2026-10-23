//! Q-013-A: evidence-bound probe admission and lifecycle *reference model*.
//! No process spawning, privilege grant, model loading, or network operations.
use crate::sha256::sha256_hex;
use crate::{
    valid_transition, EngineIdentity, Fact, InstanceState, ModelArtifact, ModelHandle,
    QualificationStatus, ReadinessLevel, ReclaimReceipt, SemanticProbeReceipt, ServingMode,
    UnloadReceipt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeKind {
    DenseSentinel,
    EmbeddingTriple1024,
    MoeExpertSentence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeQualification {
    pub engine_sha256: String,
    pub model_sha256: String,
    pub probe: ProbeKind,
    pub evidence_ref: String,
    pub profile: String,
    pub status: QualificationStatus,
    /// Descriptive observation, NOT a hard admission budget.
    pub observed_card_gpu_peak_mib: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeIntent {
    pub engine_sha256: String,
    pub model_sha256: String,
    pub probe: ProbeKind,
    pub evidence_ref: String,
    pub serving_mode: ServingMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidIdentity,
    UnsupportedMode,
    NoQualifiedEvidence,
    AmbiguousEvidence,
    InvalidEvidence,
    WrongPhase,
    ReceiptIdentityMismatch,
    SemanticProofMissing,
    WitnessDigestMismatch,
    UnloadNotAcknowledged,
    ReclaimNotHostVerified,
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn valid_evidence_ref(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(valid_sha)
}

/// SHA-256 of the exact witness output bytes, lowercase hex.
///
/// A witness must carry this value; a digest-shaped string is not evidence.
pub fn witness_digest(observed_output: &str) -> String {
    sha256_hex(observed_output.as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbePlan {
    pub(crate) binding: ProbeQualification,
}

impl ProbePlan {
    pub fn binding(&self) -> &ProbeQualification {
        &self.binding
    }
    /// This is an informational planning result, **not** permission to execute.
    pub fn execution_authorized(&self) -> bool {
        false
    }
}

/// Exact evidence binding only: no fallback by friendly model name, partial digest,
/// advertised capability, or other provider; rejects conflicting duplicate receipts.
pub fn select_probe(
    candidates: &[ProbeQualification],
    intent: &ProbeIntent,
) -> Result<ProbePlan, AdmissionError> {
    if !valid_sha(&intent.engine_sha256)
        || !valid_sha(&intent.model_sha256)
        || !valid_evidence_ref(&intent.evidence_ref)
    {
        return Err(AdmissionError::InvalidIdentity);
    }
    if intent.serving_mode != ServingMode::OneShot {
        return Err(AdmissionError::UnsupportedMode);
    }
    let matching: Vec<&ProbeQualification> = candidates
        .iter()
        .filter(|q| {
            q.engine_sha256 == intent.engine_sha256
                && q.model_sha256 == intent.model_sha256
                && q.probe == intent.probe
                && q.evidence_ref == intent.evidence_ref
        })
        .collect();
    if matching.len() > 1 {
        return Err(AdmissionError::AmbiguousEvidence);
    }
    let q = matching
        .first()
        .ok_or(AdmissionError::NoQualifiedEvidence)?;
    if q.status != QualificationStatus::Qualified
        || q.profile != "Q012_SEQUENTIAL_FUNCTIONAL"
        || !valid_evidence_ref(&q.evidence_ref)
        || q.observed_card_gpu_peak_mib == 0
        || !valid_sha(&q.engine_sha256)
        || !valid_sha(&q.model_sha256)
    {
        return Err(AdmissionError::InvalidEvidence);
    }
    Ok(ProbePlan {
        binding: (*q).clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependentWitness {
    pub probe: ProbeKind,
    pub evidence_ref: String,
    pub verifier: String,
    /// Exact bytes interpreted by the independent harness; no syntactic digest implies trust.
    pub observed_output: String,
    pub observed_digest_sha256: String,
    pub passed: bool,
    pub checked_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ProbeLifecycle {
    plan: ProbePlan,
    engine: EngineIdentity,
    model: ModelArtifact,
    handle: ModelHandle,
    state: InstanceState,
    created_at_ms: u64,
    unloaded_at_ms: Option<u64>,
    reclaim_verified: bool,
}

impl ProbeLifecycle {
    pub fn new(
        plan: ProbePlan,
        engine: EngineIdentity,
        model: ModelArtifact,
        handle: ModelHandle,
        created_at_ms: u64,
    ) -> Result<Self, AdmissionError> {
        if created_at_ms == 0
            || handle.0.trim().is_empty()
            || engine.binary_sha256.as_deref() != Some(&plan.binding.engine_sha256)
            || model.sha256.as_deref() != Some(&plan.binding.model_sha256)
        {
            return Err(AdmissionError::ReceiptIdentityMismatch);
        }
        Ok(Self {
            plan,
            engine,
            model,
            handle,
            state: InstanceState::Offline,
            created_at_ms,
            unloaded_at_ms: None,
            reclaim_verified: false,
        })
    }
    pub fn state(&self) -> InstanceState {
        self.state
    }
    pub fn reclaimed(&self) -> bool {
        self.reclaim_verified
    }
    pub fn can_route_user_requests(&self) -> bool {
        false
    }

    pub fn loading(&mut self) -> Result<(), AdmissionError> {
        self.transition(InstanceState::Loading)
    }
    fn transition(&mut self, next: InstanceState) -> Result<(), AdmissionError> {
        if !valid_transition(self.state, next) {
            return Err(AdmissionError::WrongPhase);
        }
        self.state = next;
        Ok(())
    }
    pub fn semantic_ready(
        &mut self,
        receipt: &SemanticProbeReceipt,
        witness: &IndependentWitness,
    ) -> Result<(), AdmissionError> {
        if !matches!(
            self.state,
            InstanceState::Loading
                | InstanceState::Ready(ReadinessLevel::Control)
                | InstanceState::Ready(ReadinessLevel::Registered)
        ) {
            return Err(AdmissionError::WrongPhase);
        }
        if receipt.handle != self.handle
            || receipt.engine != self.engine
            || receipt.model != self.model
            || receipt.completed_at_ms < self.created_at_ms
            || receipt.probe_id != witness.evidence_ref
        {
            return Err(AdmissionError::ReceiptIdentityMismatch);
        }
        if !receipt.passed
            || receipt.observed_output.is_empty()
            || witness.probe != self.plan.binding.probe
            || witness.evidence_ref != self.plan.binding.evidence_ref
            || witness.evidence_ref != receipt.probe_id
            || witness.verifier.trim().is_empty()
            || witness.verifier == self.engine.name
            || witness.observed_output != receipt.observed_output
            || !witness.passed
            || witness.checked_at_ms < receipt.completed_at_ms
        {
            return Err(AdmissionError::SemanticProofMissing);
        }
        // The digest must be the digest *of the bytes above*, not a shape-valid token.
        if !valid_sha(&witness.observed_digest_sha256)
            || witness_digest(&witness.observed_output) != witness.observed_digest_sha256
        {
            return Err(AdmissionError::WitnessDigestMismatch);
        }
        self.transition(InstanceState::Ready(ReadinessLevel::Semantic))
    }
    /// Marks only the start of the *qualified probe*. Never delegates arbitrary prompts.
    pub fn serving_probe(&mut self) -> Result<(), AdmissionError> {
        self.transition(InstanceState::Serving)
    }
    pub fn drain(&mut self) -> Result<(), AdmissionError> {
        self.transition(InstanceState::Draining)
    }
    pub fn unloading(&mut self) -> Result<(), AdmissionError> {
        self.transition(InstanceState::Unloading)
    }
    pub fn unloaded(&mut self, receipt: &UnloadReceipt) -> Result<(), AdmissionError> {
        if self.state != InstanceState::Unloading {
            return Err(AdmissionError::WrongPhase);
        }
        if receipt.handle != self.handle || receipt.completed_at_ms < self.created_at_ms {
            return Err(AdmissionError::ReceiptIdentityMismatch);
        }
        if !receipt.engine_acknowledged {
            return Err(AdmissionError::UnloadNotAcknowledged);
        }
        self.transition(InstanceState::Stopped)?;
        self.unloaded_at_ms = Some(receipt.completed_at_ms);
        Ok(())
    }
    pub fn verify_reclaim(&mut self, receipt: &ReclaimReceipt) -> Result<(), AdmissionError> {
        if self.state != InstanceState::Stopped
            || self.unloaded_at_ms.is_none()
            || self.reclaim_verified
        {
            return Err(AdmissionError::WrongPhase);
        }
        if receipt.handle != self.handle {
            return Err(AdmissionError::ReceiptIdentityMismatch);
        }
        let since = self.unloaded_at_ms.expect("checked above");
        let host_true = matches!(&receipt.within_ambient_tolerance,
            Fact::HostObserved { value: true, source, observed_at_ms }
                if !source.trim().is_empty() && *observed_at_ms >= since);
        let host_ambient = matches!(&receipt.ambient_gpu_bytes,
            Fact::HostObserved { source, observed_at_ms, .. }
                if !source.trim().is_empty() && *observed_at_ms >= self.created_at_ms);
        let host_final = matches!(&receipt.final_gpu_bytes,
            Fact::HostObserved { source, observed_at_ms, .. }
                if !source.trim().is_empty() && *observed_at_ms >= since);
        let host_driver = matches!(&receipt.driver_health,
            Fact::HostObserved { value, source, observed_at_ms }
                if value == "NO_XID" && !source.trim().is_empty() && *observed_at_ms >= since);
        if !(host_true && host_ambient && host_final && host_driver) {
            return Err(AdmissionError::ReclaimNotHostVerified);
        }
        self.reclaim_verified = true;
        Ok(())
    }
}
