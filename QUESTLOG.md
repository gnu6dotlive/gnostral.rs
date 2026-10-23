# QUESTLOG — gnostral.rs

> Evidence-led execution index · 2026-10-10. Reference GPU: RTX 3070 8 GiB; host RAM: 32 GiB.
> CLOSED means the *specified bounded gate*, not product release.
>
> [Start here](docs/START_HERE.md) · [Capabilities](docs/CAPABILITY_MATRIX.md) · [Roadmap](docs/ROADMAP.md) · [Evidence](docs/EVIDENCE_MODEL.md)

## 1. Closed foundational research (Q-001–Q-008)

| ID | Decision | Established scope |
| --- | --- | --- |
| [Q-001](quests/Q-001-FG03-TRUTH-01.md) | CLOSED — claim corrected | FG-02 35.2 tok/s retained; unsupported FG-03 40 tok/s claim not published |
| [Q-002](quests/Q-002-RUNTIME-BASELINE-01.md) | CLOSED — qualified with limits | Bonsai 27B PTQ1 control and desktop-pressure baseline, semantic readiness, lifecycle, reclaim |
| [Q-003](quests/Q-003-ELASTIC-LEDGER-02.md) | CLOSED — observability | Residency/reclaim ledger over distinct mechanisms; no generic allocator |
| [Q-004](quests/Q-004-XINFER-SPIKE-01.md) | CLOSED — research | xInfer candidate scheduling/KV analysis; no general adoption |
| [Q-005](quests/Q-005-ZLLM-PLACEMENT-01.md) | CLOSED — research | zLLM placement/intake; no universal hot-swap proof |
| [Q-006](quests/Q-006-UNILLM-DONOR-01.md) | CLOSED — donor research | Engine abstraction patterns assessed, not production implementations |
| [Q-007](quests/Q-007-ENGINE-PROVIDER-01.md) | CLOSED — typed contract | Provider lifecycle, HTTP vs semantic readiness and unload/reclaim states |
| [Q-008](quests/Q-008-MODELD-PROMOTION-GATE.md) | CLOSED — authority boundary | Observability qualified as contract; generic KV/expert materialization stays engine-owned |

## 2. Native runtime proof chain (Q-009–Q-013)

| ID | Current verdict | What is established — and what is not |
| --- | --- | --- |
| [Q-009](quests/Q-009-STRATA-NATIVE-RESIDENCY-01.md) | PARTIAL / separate experimental track | Strata engine-owned residency research; [Oct 9 intake](quests/Q-009-STRATA-NATIVE-RESIDENCY-01.md) is a dated snapshot superseded in part by later Q-010/Q-011 results; no product gate |
| Q-010 · CPU MoE hot path | CLOSED — bounded A/B | [Strata CPU Q2_K/Q3_K borrowed weights](research/runtime/GNOSTRAL-Q010-Q011-PUBLIC-HANDOFF-20261010.md): median 9.51 → 20.18 tok/s (2.12×) on six-check probe, N=3 per lane; failed first reclaim attempt retained; not proven superior to llama.cpp |
| Q-011 · placement evidence | CLOSED — observed on load | [GPU mapping](research/runtime/GNOSTRAL-Q010-Q011-PUBLIC-HANDOFF-20261010.md): 22/21/21 layers baseline vs 13/12/13 under artificial VRAM reservation; no online model migration or compute contender |
| Q-012 · three families | CLOSED — functional 3/3 | [One exact binary](research/runtime/GNOSTRAL-Q012-NATIVE-CAPABILITIES-20261010.md), sequential dense, Qwen3 embeddings, quantized Qwen3 MoE; independent hashes and oracle; negative missing-Cargo-feature regression preserved |
| [Q-013-A](quests/Q-013-EVIDENCE-BOUND-ADMISSION.md) · plans | CLOSED — contract only | Three SHA-pinned evidence-bound *plans*, no executable grant |
| [Q-013-B](quests/Q-013B-ISOLATED-FIXTURE.md) · subprocess | CLOSED — CPU fixture only | Cgroup checks, bounded process, cancellation/timeout/group kill, CPU reclaim; no model or GPU reclaim claims |
| [Q-013-C](quests/Q-013C-REAL-DENSE-PILOT.md) · dense supervisor | CLOSED — single real server | [Pinned Qwen2.5-Coder-1.5B](evidence/runs/gnostral-q013c-real-dense-single-server-20261010.json), GNOSTRAL exact oracle, 16.829s lifecycle, +8 MiB ambient GPU post-stop; not persistent or multi-model |

## 3. Proposed execution gates — NOT QUALIFIED

| Candidate | State | Required proof before promotion |
| --- | --- | --- |
| **Q-013-D** · authenticated real-engine adapter | NEXT — PROPOSED | Bind listener to owned PID/engine/model identity; independent semantic validation with verifiable provenance; fresh VRAM/driver evidence; timeout/crash/reclaim; narrow Rust EngineProvider integration |
| **Q-014** · serialized model lifecycle | PROPOSED | Two real models, separate processes; A→B→A load/use/unload; semantic parity; bounded host/GPU reclaim; desktop survives; no concurrency claim |
| **Q-015** · resource-aware admission | PROPOSED | Expiring host/VRAM observations, denial on UNKNOWN, resource envelopes/cgroup enforcement, cancellation under pressure; never assign expert/KV placement authority to modeld |
| **Q-016** · client SLO qualification | PROPOSED | TTFT/TPOT/ITL P95/P99 from frozen workload and client events, controlled static-vs-SLO A/B, throughput trade-offs and negative gates |
| Multiple engines / AMD / Intel / multi-GPU | RESEARCH ONLY | Independent physical devices and driver/backend tests; typed traits alone do not qualify support |
| **Q-025** · engine-direction decision | PROPOSED — protocol frozen | [Three-way benchmark](quests/Q-025-ENGINE-DIRECTION.md): llama.cpp CPU-MoE vs mistralrs-strata vs +residency on the same GGUF/host; decision rules pre-declared; no roadmap change before the receipt |

## 4. Independent research radar

- **P1** [Burn Remote v5 / Flex](research/runtime/RUST-INFERENCE-WATCH-20261010.md): protocol-v4 refusal and byte-preserving quantized transfer oracle Q8/Q4/FP8/FP4, isolated loopback peers.
- **P2** [Ferrum SLO](research/runtime/RUST-INFERENCE-WATCH-20261010.md): measure client-visible TTFT/TPOT/ITL and throughput before adopting an opt-in scheduler.
- **P2** [WakeKV](research/runtime/WAKEKV-LEAD-20261009.md): reversible KV residency, paper review only; no local implementation or numbered quest.
- **P3** [CubeCL](research/runtime/RUST-INFERENCE-WATCH-20261010.md): concurrent inter-GPU transfers/collectives, pending physical multi-GPU hardware.
- **Ongoing** Bonsai PTQ1, ternary/low-bit CUDA kernel research and engine-owned expert/KV residency, each gated independently.

## Publication and safety contract

The GitHub repository is a **research workbench with executable fixtures**, not a production serving service. Claims must retain model/binary SHA, workload, host profile, source changes, negative results, audit and explicit nonclaims. Source support ≠ local observation ≠ functional PASS ≠ production readiness. Do not publish raw responses, secrets, local machine configuration or model weights.

The Oct 9 Q-009/WakeKV intake is preserved from a locally signed research commit. Its historical worktree status is not silently represented as Oct 10 ground truth. The workstation's root `main` and its uncommitted files are left unchanged during public consolidation.

## 5. Donor intake

| ID | Quest | State | Boundary |
| --- | --- | --- | --- |
| [Q-017](quests/Q-017-DEEPINFER-DONOR-01.md) · DEEPINFER-DONOR-01 | REFERENCE_PINNED | Content-pinned reference donor only; no build, import, dependency or inference. Five engine-lane experiments (Q-018…Q-023) and the supervisor contract (Q-024) are gated separately. |
