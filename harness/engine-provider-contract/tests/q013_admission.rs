use gnostral_engine_provider_contract::q013::*;
use gnostral_engine_provider_contract::*;
const BIN: &str = "64eec222ccefff4cd4c0067bd83348f1d22eebda27d0d857e352ee090cf19723";
const MOE: &str = "db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170";
const EMB: &str = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd";
const DEN: &str = "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8";
const REF: &str = "sha256:48241e35df171941e5d80a8f3832a55642de43d96abad04c0a000208e1cd3164";

fn qualified(model: &str, probe: ProbeKind) -> ProbeQualification {
    ProbeQualification {
        engine_sha256: BIN.into(),
        model_sha256: model.into(),
        probe,
        evidence_ref: REF.into(),
        profile: "Q012_SEQUENTIAL_FUNCTIONAL".into(),
        status: QualificationStatus::Qualified,
        observed_card_gpu_peak_mib: 2427,
    }
}
fn intent(model: &str, probe: ProbeKind) -> ProbeIntent {
    ProbeIntent {
        engine_sha256: BIN.into(),
        model_sha256: model.into(),
        probe,
        evidence_ref: REF.into(),
        serving_mode: ServingMode::OneShot,
    }
}
fn plan() -> ProbePlan {
    select_probe(
        &[qualified(MOE, ProbeKind::MoeExpertSentence)],
        &intent(MOE, ProbeKind::MoeExpertSentence),
    )
    .unwrap()
}
fn engine() -> EngineIdentity {
    EngineIdentity {
        name: "mistral.rs".into(),
        revision: Some("117badf".into()),
        binary_sha256: Some(BIN.into()),
        transport: EngineTransport::LocalServer,
    }
}
fn model() -> ModelArtifact {
    ModelArtifact {
        name: "Qwen3-30B-A3B Q2_K".into(),
        sha256: Some(MOE.into()),
        format: "GGUF".into(),
        bytes: Some(11_258_610_240),
    }
}
fn session() -> ProbeLifecycle {
    ProbeLifecycle::new(
        plan(),
        engine(),
        model(),
        ModelHandle("q013-moe".into()),
        100,
    )
    .unwrap()
}
fn receipt() -> SemanticProbeReceipt {
    SemanticProbeReceipt {
        handle: ModelHandle("q013-moe".into()),
        probe_id: REF.into(),
        passed: true,
        observed_output: "MoE experts moved to CPU RAM".into(),
        engine: engine(),
        model: model(),
        completed_at_ms: 102,
    }
}
fn witness() -> IndependentWitness {
    IndependentWitness {
        probe: ProbeKind::MoeExpertSentence,
        evidence_ref: REF.into(),
        verifier: "independent-harness".into(),
        observed_output: "MoE experts moved to CPU RAM".into(),
        observed_digest_sha256: witness_digest("MoE experts moved to CPU RAM"),
        passed: true,
        checked_at_ms: 103,
    }
}
fn observed<T>(value: T, at: u64) -> Fact<T> {
    Fact::HostObserved {
        value,
        source: "host-sampler".into(),
        observed_at_ms: at,
    }
}
fn reclaim() -> ReclaimReceipt {
    ReclaimReceipt {
        handle: ModelHandle("q013-moe".into()),
        ambient_gpu_bytes: observed(1200u64, 105),
        final_gpu_bytes: observed(1200u64, 106),
        within_ambient_tolerance: observed(true, 106),
        reclaim_latency_ms: Fact::Derived {
            value: 12,
            method: "reclaim".into(),
            observed_at_ms: 106,
        },
        driver_health: observed("NO_XID".to_string(), 106),
    }
}

#[test]
fn three_q012_profiles_are_plannable_but_not_execution_authorizations() {
    let qs = [
        qualified(DEN, ProbeKind::DenseSentinel),
        qualified(EMB, ProbeKind::EmbeddingTriple1024),
        qualified(MOE, ProbeKind::MoeExpertSentence),
    ];
    for (model, kind) in [
        (DEN, ProbeKind::DenseSentinel),
        (EMB, ProbeKind::EmbeddingTriple1024),
        (MOE, ProbeKind::MoeExpertSentence),
    ] {
        let selected = select_probe(&qs, &intent(model, kind)).unwrap();
        assert_eq!(selected.binding().model_sha256, model);
        assert!(!selected.execution_authorized());
    }
}
#[test]
fn mismatched_model_operation_or_evidence_cannot_fallback() {
    let qs = [qualified(MOE, ProbeKind::MoeExpertSentence)];
    assert_eq!(
        select_probe(&qs, &intent(EMB, ProbeKind::MoeExpertSentence)),
        Err(AdmissionError::NoQualifiedEvidence)
    );
    assert_eq!(
        select_probe(&qs, &intent(MOE, ProbeKind::DenseSentinel)),
        Err(AdmissionError::NoQualifiedEvidence)
    );
    let mut i = intent(MOE, ProbeKind::MoeExpertSentence);
    i.evidence_ref = "unsupported".into();
    assert_eq!(select_probe(&qs, &i), Err(AdmissionError::InvalidIdentity));
    i.evidence_ref = format!("sha256:{}", "0".repeat(64));
    assert_eq!(
        select_probe(&qs, &i),
        Err(AdmissionError::NoQualifiedEvidence)
    );
}
#[test]
fn persistent_service_and_invalid_or_old_sha_rejected() {
    let qs = [qualified(MOE, ProbeKind::MoeExpertSentence)];
    let mut i = intent(MOE, ProbeKind::MoeExpertSentence);
    i.serving_mode = ServingMode::Persistent;
    assert_eq!(select_probe(&qs, &i), Err(AdmissionError::UnsupportedMode));
    i.serving_mode = ServingMode::OneShot;
    i.engine_sha256 = "wrong".into();
    assert_eq!(select_probe(&qs, &i), Err(AdmissionError::InvalidIdentity));
    i.engine_sha256 = "0".repeat(64);
    assert_eq!(
        select_probe(&qs, &i),
        Err(AdmissionError::NoQualifiedEvidence)
    );
}
#[test]
fn rejected_or_duplicate_evidence_cannot_be_selected() {
    let mut q = qualified(MOE, ProbeKind::MoeExpertSentence);
    q.status = QualificationStatus::Blocked;
    assert_eq!(
        select_probe(&[q], &intent(MOE, ProbeKind::MoeExpertSentence)),
        Err(AdmissionError::InvalidEvidence)
    );
    let q = qualified(MOE, ProbeKind::MoeExpertSentence);
    assert_eq!(
        select_probe(&[q.clone(), q], &intent(MOE, ProbeKind::MoeExpertSentence)),
        Err(AdmissionError::AmbiguousEvidence)
    );
}
#[test]
fn semantic_witness_must_precede_probe_serving() {
    let mut s = session();
    assert_eq!(s.serving_probe(), Err(AdmissionError::WrongPhase));
    s.loading().unwrap();
    assert_eq!(s.serving_probe(), Err(AdmissionError::WrongPhase));
    let mut w = witness();
    w.verifier = "mistral.rs".into();
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::SemanticProofMissing)
    );
    s.semantic_ready(&receipt(), &witness()).unwrap();
    assert_eq!(s.state(), InstanceState::Ready(ReadinessLevel::Semantic));
    assert!(!s.can_route_user_requests());
    s.serving_probe().unwrap();
}
#[test]
fn incorrect_receipt_can_never_advance_lifecycle() {
    let mut s = session();
    s.loading().unwrap();
    let mut r = receipt();
    r.handle = ModelHandle("other".into());
    assert_eq!(
        s.semantic_ready(&r, &witness()),
        Err(AdmissionError::ReceiptIdentityMismatch)
    );
    let mut r = receipt();
    r.passed = false;
    assert_eq!(
        s.semantic_ready(&r, &witness()),
        Err(AdmissionError::SemanticProofMissing)
    );
    let mut r = receipt();
    r.engine.binary_sha256 = Some("0".repeat(64));
    assert_eq!(
        s.semantic_ready(&r, &witness()),
        Err(AdmissionError::ReceiptIdentityMismatch)
    );
    assert_eq!(s.state(), InstanceState::Loading);
}

#[test]
fn unload_acknowledgement_and_host_reclaim_are_distinct_gates() {
    let mut s = session();
    s.loading().unwrap();
    s.semantic_ready(&receipt(), &witness()).unwrap();
    s.serving_probe().unwrap();
    assert_eq!(
        s.verify_reclaim(&reclaim()),
        Err(AdmissionError::WrongPhase)
    );
    s.drain().unwrap();
    s.unloading().unwrap();
    let mut u = UnloadReceipt {
        handle: ModelHandle("q013-moe".into()),
        engine_acknowledged: false,
        completed_at_ms: 104,
    };
    assert_eq!(s.unloaded(&u), Err(AdmissionError::UnloadNotAcknowledged));
    assert!(!s.reclaimed());
    u.engine_acknowledged = true;
    s.unloaded(&u).unwrap();
    assert_eq!(s.state(), InstanceState::Stopped);
    assert!(!s.reclaimed());
    let mut r = reclaim();
    r.within_ambient_tolerance = Fact::Unknown {
        reason: "not observed".into(),
    };
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
    r.within_ambient_tolerance = Fact::Declared {
        value: true,
        source: "engine".into(),
    };
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
    s.verify_reclaim(&reclaim()).unwrap();
    assert!(s.reclaimed());
    assert_eq!(s.serving_probe(), Err(AdmissionError::WrongPhase));
    assert_eq!(
        s.verify_reclaim(&reclaim()),
        Err(AdmissionError::WrongPhase)
    );
}
#[test]
fn stale_driver_health_and_reclaim_samples_fail_closed() {
    let mut s = session();
    s.loading().unwrap();
    s.semantic_ready(&receipt(), &witness()).unwrap();
    s.drain().unwrap();
    s.unloading().unwrap();
    s.unloaded(&UnloadReceipt {
        handle: ModelHandle("q013-moe".into()),
        engine_acknowledged: true,
        completed_at_ms: 104,
    })
    .unwrap();
    let mut r = reclaim();
    r.driver_health = observed("XID".to_string(), 106);
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
    r = reclaim();
    r.final_gpu_bytes = observed(1200, 103);
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
}
#[test]
fn initializing_with_different_artifact_cannot_reuse_plan() {
    let mut m = model();
    m.sha256 = Some(EMB.into());
    assert_eq!(
        ProbeLifecycle::new(plan(), engine(), m, ModelHandle("q013-moe".into()), 100).err(),
        Some(AdmissionError::ReceiptIdentityMismatch)
    );
}

#[test]
fn independent_witness_must_match_actual_output_and_time() {
    let mut s = session();
    s.loading().unwrap();
    let mut w = witness();
    w.observed_output = "different response".into();
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::SemanticProofMissing)
    );
    w = witness();
    w.checked_at_ms = 101; // earlier than the semantic receipt
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::SemanticProofMissing)
    );
    w = witness();
    w.observed_digest_sha256 = "0".repeat(64);
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::WitnessDigestMismatch)
    );
    assert_eq!(s.state(), InstanceState::Loading);

    // A valid-looking digest of different bytes is not evidence.
    w = witness();
    w.observed_digest_sha256 = witness_digest("another response");
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::WitnessDigestMismatch)
    );

    // A non-hex digest is rejected before any byte comparison.
    w = witness();
    w.observed_digest_sha256 = "z".repeat(64);
    assert_eq!(
        s.semantic_ready(&receipt(), &w),
        Err(AdmissionError::WitnessDigestMismatch)
    );
    assert_eq!(s.state(), InstanceState::Loading);
}

#[test]
fn witness_digest_helper_matches_the_published_nist_vector() {
    assert_eq!(
        witness_digest("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn advertised_or_unknown_ambient_memory_is_not_reclaim_proof() {
    let mut s = session();
    s.loading().unwrap();
    s.semantic_ready(&receipt(), &witness()).unwrap();
    s.drain().unwrap();
    s.unloading().unwrap();
    s.unloaded(&UnloadReceipt {
        handle: ModelHandle("q013-moe".into()),
        engine_acknowledged: true,
        completed_at_ms: 104,
    })
    .unwrap();
    let mut r = reclaim();
    r.ambient_gpu_bytes = Fact::Declared {
        value: 1200,
        source: "runtime".into(),
    };
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
    r = reclaim();
    r.ambient_gpu_bytes = Fact::Unknown {
        reason: "sampler unavailable".into(),
    };
    assert_eq!(
        s.verify_reclaim(&r),
        Err(AdmissionError::ReclaimNotHostVerified)
    );
}
