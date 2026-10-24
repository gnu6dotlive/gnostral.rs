# gnostral.rs · research-to-runtime roadmap

**As of 2026-10-10. Technical sequencing, not a delivery schedule.**
[QUESTLOG](../QUESTLOG.md) · [Capability truth table](CAPABILITY_MATRIX.md) · [How to reproduce](REPRODUCIBILITY.md)

> **North star / cible :** an inspectable, Rust-first, **local** inference control plane for constrained hardware: engine-aware model selection, explicit resource envelopes, verifiable readiness, safe unloading and accountable performance. Kernels, KV and expert placement remain owned by each execution engine.

## The dependency graph

```mermaid
flowchart LR
    A[Q-002/Q-003 baseline & residency ledger] --> B[Q-007/Q-008 provider contract & authority]
    B --> C[Q-010/Q-011 MoE hot path & placement]
    C --> D[Q-012 dense + embed + MoE functional proofs]
    D --> E[Q-013A evidence-bound plans]
    E --> F[Q-013B isolated process fixture]
    F --> G[Q-013C real dense supervised pilot]
    G --> H[Q-013D authenticated runtime adapter]
    H --> I[Q-014 sequential model A-B-A lifecycle]
    I --> J[Q-015 resource admission]
    J --> K[Q-016 client SLO qualification]
    I --> L[Future multi-model resident serving]
    K --> L
    R[Burn Remote v5 research] -. interface oracle .-> J
    S[WakeKV / engine-specific residency research] -. independent evidence .-> I
    T[Ferrum SLO research] -. measurement donor .-> K
```

The dashed research links are **candidate inputs**, not dependencies, integrations or verified technology transfers.

## Lane A — harden actual local serving

### M0 · Proven baseline — CLOSED

Q010 CPU MoE A/B, Q011 load-time placement observation, Q012 three-family sequential functional qualification, Q013A typed evidence plans, Q013B CPU fixture process isolation and Q013C one genuine dense server with independent exact-oracle audit. See [matrix](CAPABILITY_MATRIX.md).

### M1 · Q-013D: prove identity **during** serving — NEXT, PROPOSED

**Goal:** turn the single dense experimental supervisor into a **narrow** validated Rust `EngineProvider` adapter, still refusing arbitrary programs/models.

Work packages:
1. Bind localhost listener to the **owned process** and session (PID reuse, port takeover, unexpected existing listener, early exit and stale health responses must be rejected).
2. Recheck binary+model identity at the relevant use boundary; define the trust model for file replacement/TOCTOU and prove that a snapshot cannot be swapped by an untrusted actor within scope.
3. Make semantic witness provenance verifiable, not just a caller-supplied `passed=true` or digest-shaped string.
4. Fresh host-sourced GPU/driver/cgroup observations during service; timeout, negative OOM/Xid path, cancellation and process-group reclaim.
5. Replay with a real dense model at strict memory headroom; exact result, explicit state transitions and independent post-run audit.

**Exit:** pinned identity + listener identity + semantic result + process-tree shutdown + bounded host/GPU reclaim + no desktop session loss, across a frozen repetition protocol with negative controls. One server only.

### M2 · Q-014: serialized A → B → A — PROPOSED

**Goal:** before claiming *hot swap*, prove **cold sequential lifecycle** with two different real model families.

Experiment:
- Process A dense → unload → confirm host RAM and GPU return.
- Process B embeddings (or safely supported MoE within RAM envelope) → unload → confirm return.
- Process A again → re-verify semantic output and no cumulative resource drift.
- Repeat with canceled startup, timeout mid-load, failure after semantic readiness and unavailable GPU reserve.

**Exit:** reproducible bounded load/serve/drain/unload/reclaim, correct outputs and negative gates at every transition. **This does not yet prove simultaneous serving, hot switching without interruption, or one-process model residency.**

### M3 · Q-015: resource admission — PROPOSED

Define a typed envelope for **max RAM, VRAM headroom, CPU threads, GPU/driver health and admission TTL**; enforce through systemd cgroups + sampled GPU data, rejecting stale/UNKNOWN records. Prove cold-start admission under synthetic VRAM reservation without evicting or starving the desktop. Engine retains precise layer/expert/KV mapping authority.

**Exit:** no over-claim of hard NVIDIA VRAM caps; explain TOCTOU limitations and actual failure path. No arbitrary modeld allocation controls.

### M4 · Q-016: quality-of-service instrumentation — PROPOSED

Record client-side TTFT/TPOT/visible ITL, P50/P95/P99, throughput, request failure and output correctness at fixed tokenizer/prompts/concurrency. Only then compare static prefill vs optional adapted scheduling, borrowing experimental ideas from Ferrum. Disable adaptation by default until causal gains survive paired-run tests.

**Exit:** confidence-bounded throughput/tail trade-offs on target hardware, not estimates displayed by `/health` alone.

### M5 · A real multi-model service — CONDITIONAL FUTURE

Possible only after M1–M4 and additional authority/security review: admission for >1 family, concurrency/isolation, secure API transport and identity, per-model health, failure/recovery, inspectable scheduling, packaging, distribution rights and a production operations plan.

## Lane B — cutting-edge research; **not** production adoption

| Priority | Program | First falsifiable experiment | Blocker |
| --- | --- | --- | --- |
| **P1** | [BURN-REMOTE-01](../research/runtime/RUST-INFERENCE-WATCH-20261010.md) · quantized tensor transport | Two local loopback peers; protocol v5 acceptance/v4 refusal; byte-for-byte schemes/scales/values in Q8/Q4/FP8/FP4; shape/layout changes and exact provenance | No local compatible Burn test matrix frozen yet |
| **P2** | [FERRUM-SLO-01](../research/runtime/RUST-INFERENCE-WATCH-20261010.md) · client tail metrics | Offline request trace and P99 TTFT/TPOT/ITL oracle, then same-model static vs opt-in SLO A/B | Upstream reported no established ShareGPT throughput benefit |
| **P2** | [WakeKV](../research/runtime/WAKEKV-LEAD-20261009.md) · reversible KV | Supported attention model, matched-VRAM static vs reversible CPU reservoir; correctness on wake/re-promotion and PCIe stalls | Paper read; local implementation/weights not verified |
| **P3** | [CubeCL](../research/runtime/RUST-INFERENCE-WATCH-20261010.md) · collective ordering | Concurrent transfers + collectives with explicit timeouts, independent GPU telemetry | Second physical GPU absent |
| **Ongoing** | Bonsai ternary PTQ1 / Q2K-Q3K MoE | Kernel layout, signs/LUT, actual prefill/decode, memory bandwidth and unsafe-code correctness tests | Distinct models/protocols must never be conflated |

Parallel research may be executed in independent worktrees; **no research intake is authorized to silently alter the Q012-pinned CUDA runtime**.

## Engine program — PROPOSED

A separate, gated program for the engine itself: **run a 30B-class MoE model on an 8 GiB consumer GPU at quality parity, faster than anything that fits that card, with reusable measurements.** Program of record: [ENGINE_PROGRAM.md](ENGINE_PROGRAM.md) — thesis, scorecard, five invariants, workstreams W1–W7, milestone ladder M0–M7, decision gates D1–D4, and the evidence policy ("no number without a receipt").

Lanes A and B above are unchanged by it. The engine program does not replace **Q-013D → Q-014 → Q-015 → Q-016**, does not merge engines, and does not touch **BURN-REMOTE-01**. At present **no engine speed claim exists**: offline, CPU-only pieces of Q-026 to Q-031 and Q-038 to Q-042 are implemented and tested, none is integrated into the engine and none has a receipt.

| Workstream | Quests (all **PROPOSED**) |
| --- | --- |
| W1 · MoE executor + CPU kernels | [Q-026](../quests/Q-026-MOE-EXECUTOR-02.md) MOE-EXECUTOR-02 · [Q-029](../quests/Q-029-CPU-KERNEL-Q8-01.md) CPU-KERNEL-Q8-01 · [Q-037](../quests/Q-037-PREFETCH-EXPERT-01.md) PREFETCH-EXPERT-01 (stretch) |
| W2 · Memory hierarchy | [Q-027](../quests/Q-027-RESIDENCY-DYNAMIC-01.md) RESIDENCY-DYNAMIC-01 · [Q-028](../quests/Q-028-PAGED-OFFLOAD-01.md) PAGED-OFFLOAD-01 · [Q-038](../quests/Q-038-KV-STREAM-01.md) KV-STREAM-01 |
| W3 · Scheduler + speculation | [Q-039](../quests/Q-039-SLO-SCHED-01.md) SLO-SCHED-01 |
| W4 · Supervisor + provider | [Q-040](../quests/Q-040-MULTI-ENGINE-01.md) MULTI-ENGINE-01 · [Q-041](../quests/Q-041-DESKTOP-RESERVE-01.md) DESKTOP-RESERVE-01 |
| W5 · Evidence + quality | [Q-030](../quests/Q-030-QUALITY-PARITY-01.md) QUALITY-PARITY-01 · [Q-031](../quests/Q-031-ENGINE-PERF-01.md) ENGINE-PERF-01 · [Q-042](../quests/Q-042-PROFILE-SPARK-01.md) PROFILE-SPARK-01 |
| W6 · Coverage + packaging | [Q-033](../quests/Q-033-MODEL-MATRIX-01.md) MODEL-MATRIX-01 · [Q-034](../quests/Q-034-PACKAGING-01.md) PACKAGING-01 |
| W7 · Assistant engine | [Q-043](../quests/Q-043-COHOST-01.md) COHOST-01 · [Q-044](../quests/Q-044-VOICE-TURN-01.md) VOICE-TURN-01 · [Q-045](../quests/Q-045-GPU-SHARE-01.md) GPU-SHARE-01 · [Q-046](../quests/Q-046-DRAFT-SPEC-01.md) DRAFT-SPEC-01 · [Q-047](../quests/Q-047-SPEECH-MODELS-01.md) SPEECH-MODELS-01 |
| Unallocated | [Q-032](../quests/Q-032-RESERVED.md) · [Q-035](../quests/Q-035-RESERVED.md) · [Q-036](../quests/Q-036-RESERVED.md) — reserved numbers, no scope |

Order of work: M0 oracle + baseline (Q-025, Q-030, Q-031) → M1 executor (Q-026, Q-029) → M2 memory (Q-027, Q-028, Q-038) → M3 scheduler (Q-018…Q-021, Q-039) → M4 supervisor (Q-024b, Q-013D→Q-016, Q-040, Q-041) → M5 coverage (Q-033, Q-034) → M6 stretch (Q-037, AMD port) → M7 assistant engine (Q-043…Q-047: language model, speech-to-text, text-to-speech and a draft model sharing one desktop card; policy work runs in parallel now). D2 at the end of M1 can legitimately end the Rust-executor bet; the negative result then stays public.

## Stop and release gates

- **Host:** use `gnu6-lab-run --exclusive` for inference, CUDA or heavy builds; read live RAM, GPU and graphical-session health before each attempt; do not disable systemd-oomd.
- **Safety:** fail closed on unknown identities, missing licenses, unverified semantic outputs, Xid events, driver instability, unreclaimed memory or unexpected processes.
- **Comparability:** preserve GPU card-usage vs per-process allocations, RSS vs PSS, host file cache vs engine warmness, original upstream vs patched engine vs lab-owned code.
- **Truth:** failed runs remain first-class evidence and never disappear when later retries pass.
- **Publication:** no release tag, public hosted API, production-performance assertion or broad portability claim until its own dedicated gate passes.

## The near-term order

1. **Now:** publish complete evidence navigation, maintain public QUESTLOG, and protect original local unpublished work.
2. **Next engineering slice:** Q013D listener/process identity + bounded negative tests (prefer CPU stubs first).
3. **Parallel low-resource research:** BURN-REMOTE-01 protocol/wire-format oracle; freeze a test plan before downloading/building.
4. **Then:** Q014 real sequential A→B→A pilot; only after Q013D permits reliable startup/stop under pressure.
5. **Later:** Q015 resource admission; Q016 trace-based SLO benchmarking; cross-vendor/multi-GPU by physical opportunity.

**Definition of progress:** one more independently auditable boundary established, not one more feature added to a roadmap.
