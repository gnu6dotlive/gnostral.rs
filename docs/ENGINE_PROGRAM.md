# Engine program — the best local inference engine for constrained hardware

Program of record · **2026-10-23**, revised **2026-10-24** (W7) · State: **PROPOSED**
[QUESTLOG](../QUESTLOG.md) · [Roadmap](ROADMAP.md) · [Capability matrix](CAPABILITY_MATRIX.md) ·
[Benchmark protocol](BENCHMARK_PROTOCOL.md) · [Evidence model](EVIDENCE_MODEL.md)

> **Source:** `audit/ENGINE-ROADMAP-2026-10-23.md` (program workspace, outside this repository).
> This page is the in-repository program of record. Quests live in `quests/Q-026…Q-047` and are
> **PROPOSED** until their own gate produces a receipt.

**No engine speed claim is made on this page.** Every target below is a gate to be measured, not a
result. The only measured engine figures quoted are the already-retained Q-009/Q-010 results,
attributed where they appear.

## Thesis

The engine should not try to be everything. The retained evidence points at exactly one place where
it can be the best in the world: **running a 30B-class MoE model on an 8 GiB consumer GPU, at quality
parity, faster than anything that fits that card, with reusable measurements.** That is a
memory-hierarchy and execution-path problem, not a kernel-authoring problem. The plan therefore
freezes the oracle first, then replaces the per-route expert loop with a grouped CPU/GPU executor and
a quantized CPU kernel, then makes residency dynamic and KV-aware, turns on paged attention and
batching for the offload configuration, adds speculation and scheduling, proves quality and
determinism, and only then widens coverage. Each step has a numeric gate; nothing is called "best"
without a receipt.

## What "best" means

"Best" is not one number. A change is an improvement only if it wins on at least one axis **without
regressing quality, peak VRAM, or reclaim** beyond tolerance. Scored on a Pareto scorecard over a
frozen workload.

### Reference workload (frozen in [Q-025](../quests/Q-025-ENGINE-DIRECTION.md))

| Field | Frozen value |
| --- | --- |
| Host | RTX 3070 8 GiB, 32 GiB RAM, 8 CPU threads, graphical desktop present |
| Model | Qwen3-30B-A3B Q2_K GGUF, exact SHA; Qwen3.8-Flash-Next (Strata target) when available |
| Context | 1K fixed prompt set, then 8K / 32K long-context tier |
| Sampling | temperature 0, fixed seed, fixed `n_predict`; semantic oracle before timing |
| Protocol | 1 warmup, 3 measured runs per lane, interleaved; every sample retained |

### Scorecard (targets, not measurements)

| Axis | Target | Anti-regression rule |
| --- | --- | --- |
| Quality | greedy token agreement ≥ 99.5 % vs a reference run on ≥ 32 fixed prompts; KL ≤ 1e-3 where logits are comparable; every divergence explained | **hard gate** — a speed win with an unexplained quality change is rejected |
| Decode (C1) | ≥ 35 tok/s (llama.cpp parity 34.54), stretch 45 with accepted MTP | no quality/VRAM regression |
| Prefill / TTFT | ≥ 2× the current candidate on 1K; ≥ 1.5× on 8K | no decode regression > 3 % |
| Peak VRAM | ≤ 7.0 GiB whole-card **including** desktop ambient, with ≥ 1.0 GiB left free | hard envelope; desktop must survive |
| Host working set | ≤ 24 GiB RSS/PSS | no unbounded growth across requests |
| Long context | 32K tokens on 8 GiB via KV streaming, decode ≥ 15 tok/s | exact same tokens as short-context baseline |
| Concurrency | C4 aggregate ≥ 1.5× C1; TPOT(C4) ≤ 1.5× TPOT(C1) | no starvation, per-request fairness receipt |
| Determinism | same seed, batch neighbours and speculation on/off ⇒ identical token ids and logprobs | batch-invariance gate |
| Lifecycle | A→B→A load/serve/unload/reclaim; reclaim within ±128 MiB; no orphan | driver health `NO_XID` |
| Versatility | ≥ 3 local engines behind one provider contract; selection receipt, no capability inheritance | conformance + no authority leak |
| Evidence | every number has a receipt and a negative-result section | no receipt, no claim |

### Non-goals

Training or fine-tuning; multi-node clusters; public multi-tenant serving; cloud provider parity.
Beating datacenter engines (vLLM / DeepInfer / SGLang) on their hardware — they are donors, not
targets. New CUDA kernel families before the execution path is fixed; new model architectures before
quality gates exist; benchmark-specific quantization changes; GPU work in CI.

## The five invariants

1. **One dispatch per layer, not one per route.** Routes are partitioned by target (GPU / CPU /
   upload) and each partition executes as a batch. No per-route `device.synchronize()`, no per-route
   `to_vec2`.
2. **Expert compute is quantized end-to-end.** Activations are quantized to Q8_0 once per layer and
   multiplied against Q2_K/Q3_K blocks in a vectorized CPU kernel with a thread pool; F32 is for
   accumulation only.
3. **Transfers are batched, pinned and leased.** Uploads run on a copy stream, in ≤ segment-sized
   batches, with an explicit ticket/lease so no slot is published before completion or released
   while referenced.
4. **Memory is a budget, not an assumption.** Expert cache, KV cache, workspace and desktop reserve
   share one envelope; admission rejects or degrades explicitly; elastic segments can return VRAM.
5. **Quality is a gate, not a regression note.** Every engine change runs the frozen quality oracle
   (token parity + KL) before any performance number is recorded.

## Workstreams

Numbering follows the source audit. **Q-018…Q-024** are the DeepInfer/provider follow-ons from the
[Q-017](../quests/Q-017-DEEPINFER-DONOR-01.md) donor plan and are not re-issued here; **Q-013-D**
through **Q-016** remain the repository's own supervisor gates and are not replaced by this program.

### W1 — MoE executor and CPU kernels (the main bet)

Files: `mistralrs-core/src/moe/experts/{residency.rs,backends.rs,forward.rs,mod.rs}`,
`mistralrs-core/src/models/qwen3_moe.rs`, `mistralrs-quant/`, the vendored Candle quant kernels,
and the testable policy crate `crates/gnostral-moe-exec/`.

| Quest | Scope |
| --- | --- |
| [Q-026](../quests/Q-026-MOE-EXECUTOR-02.md) · MOE-EXECUTOR-02 | Route partitioner, grouped CPU executor, grouped GPU reuse, scatter-add; CPU parity tests, then engine integration |
| [Q-029](../quests/Q-029-CPU-KERNEL-Q8-01.md) · CPU-KERNEL-Q8-01 | Q8_0 activation quantization, fused gate/up, batched quantized GEMM, AVX2/AVX-512 dispatch, thread pool |
| [Q-037](../quests/Q-037-PREFETCH-EXPERT-01.md) · PREFETCH-EXPERT-01 (stretch) | Router-logit next-step expert prefetch for batch decode, only if profiling shows upload stalls dominate |

**Acceptance (M1):** decode ≥ 35 tok/s at 1K C1, prefill ≥ 2× current, ≤ 7.0 GiB whole-card peak,
quality gate green, every sample retained. M1 is the milestone that decides the engine's future (D2).

### W2 — Memory: dynamic residency, KV and budget arbitration

Files: `crates/gnostral-expert-residency/` (policy), `mistralrs-core/src/moe/experts/residency.rs`
(integration), `mistralrs-core/src/paged_attention/` (KV), the vendored KV layout, and a future
`crates/gnostral-memory-plan/`.

| Quest | Scope |
| --- | --- |
| [Q-027](../quests/Q-027-RESIDENCY-DYNAMIC-01.md) · RESIDENCY-DYNAMIC-01 | Online hit histogram → admission/eviction with leases and pin counts, segmented allocator with elastic return, cross-budget arbitration, honest `UNKNOWN` telemetry |
| [Q-028](../quests/Q-028-PAGED-OFFLOAD-01.md) · PAGED-OFFLOAD-01 | PagedAttention + continuous batching when the device map is all-GPU and only experts are host-resident; hybrid KV fallback |
| [Q-038](../quests/Q-038-KV-STREAM-01.md) · KV-STREAM-01 | Long-context KV streaming to pinned host with prefetch, exact-token parity, measured stall budget |

**Acceptance (M2):** 32K context on 8 GiB, dynamic cache hit-rate receipt, no OOM, reclaim ±128 MiB,
desktop reserve preserved. Strata's own KV_PREFETCH ABBA showed ~0 gain in its first test; prefetch
stays a hypothesis, not a win.

### W3 — Scheduler, speculation and determinism

Files: `mistralrs-core/src/paged_attention/scheduler.rs`, `speculative/`, `sampler.rs`,
`sequence.rs`, `pipeline/sampling.rs`.

| Quest | Scope |
| --- | --- |
| [Q-039](../quests/Q-039-SLO-SCHED-01.md) · SLO-SCHED-01 | Deadline/priority-aware admission, fairness receipt, C1/C4 tail metrics (feeds Q-016) |

The MTP-width policy (Q-019), ZOS-lite (Q-020), sampler determinism (Q-018) and
structured-output-under-speculation (Q-021) are the donor follow-ons; **Q-018 must land before
Q-019** and before any dynamic verification length.

**Acceptance (M3):** ≥ 5 % TPOT from MTP or a recorded negative, C4 aggregate ≥ 1.5× C1,
determinism gates green, no quality change.

### W4 — Supervisor, provider and lifecycle

Files: `harness/engine-provider-contract/src/v1.rs` (draft landed), a future
`crates/gnostral-supervisor/`, `harness/q013b-local-adapter/`, `contracts/`.

| Quest | Scope |
| --- | --- |
| [Q-040](../quests/Q-040-MULTI-ENGINE-01.md) · MULTI-ENGINE-01 | mistralrs-strata in-process, llama.cpp CLI/server and the C++ Strata server behind one supervisor; per-engine qualification profile; no capability inheritance |
| [Q-041](../quests/Q-041-DESKTOP-RESERVE-01.md) · DESKTOP-RESERVE-01 | Elastic VRAM reserve handshake with a desktop-pressure observer, only after the memory manager exists |

`Q-013-D → Q-014 → Q-015 → Q-016` stay the repository's existing gates, and the provider v1
implementation (Q-024b) sits on that lane. The engine program neither replaces nor weakens them.

**Acceptance (M4):** two engines serve through the same provider API; A→B→A receipt; admission
denies `UNKNOWN`; stream/cancel/crash tests; no authority leak (v1 authority test).

### W5 — Evidence, quality and benchmark infrastructure

Files: a future `harness/bench/` (orchestrator + schemas), `harness/experiments/`, `schemas/`,
`evidence/runs/`.

| Quest | Scope |
| --- | --- |
| [Q-030](../quests/Q-030-QUALITY-PARITY-01.md) · QUALITY-PARITY-01 | Token parity, KL/perplexity harness, batch-invariance and sampler gates, long-context parity — the gate every engine PR passes |
| [Q-031](../quests/Q-031-ENGINE-PERF-01.md) · ENGINE-PERF-01 | One command runs the frozen Q-025 protocol across N lanes, normalizes receipts, renders the Pareto table, refuses un-pinned identities |
| [Q-042](../quests/Q-042-PROFILE-SPARK-01.md) · PROFILE-SPARK-01 | Per-phase timers behind an env flag; single-request timeline exported as evidence |

**Acceptance:** a benchmark run produces a receipt a second person can recompute; the quality gate
blocks any PR that changes tokens without explanation.

### W6 — Coverage and packaging (after the engine is fast and correct)

Files: a future `crates/gnostral-serve/` (or fork of server-core), CLI, docs, packaging.

| Quest | Scope |
| --- | --- |
| [Q-033](../quests/Q-033-MODEL-MATRIX-01.md) · MODEL-MATRIX-01 | Dense, embeddings, Qwen3 MoE, Qwen3.8-Flash-Next/GDN, Q4/IQ2/IQ3; multimodal only if it does not compromise the core |
| [Q-034](../quests/Q-034-PACKAGING-01.md) · PACKAGING-01 | Single binary `gnostral serve` (OpenAI-compatible), model manager, profiles, desktop reserve, reproducible builds |

**Acceptance (M5):** model-matrix receipts, one-command install/server, docs, release.

### W7 — Assistant engine: several models on one desktop card

Program revision 2026-10-24, at the owner's direction. The engine's purpose widens from "one
model, fast" to "one assistant on one desktop card": a MoE language model, speech-to-text,
text-to-speech and a small draft model resident together. The MoE expert cache is what makes it
possible: it is the elastic part of the card that shrinks when speech arrives and grows back when
it leaves, at a speed cost and never a correctness cost. W7 builds on W2 (elastic expert budget),
W3 (scheduler, speculation) and W4 (supervisor, desktop reserve); it does not weaken any of their
gates, and the W1 speed gate (M1) stays the program's first test.

Files: `crates/gnostral-cohost/`, `crates/gnostral-assistant/`, `crates/gnostral-speculative/`,
`harness/quality/speech_oracle.py`; engine seams in the supervisor (hosting instead of switching),
the residency budget and the scheduler step cap.

| Quest | Scope |
| --- | --- |
| [Q-043](../quests/Q-043-COHOST-01.md) · COHOST-01 | Co-residency broker: core + elastic budgets, class rules, shrink/park plans that never overbook, typed refusals |
| [Q-044](../quests/Q-044-VOICE-TURN-01.md) · VOICE-TURN-01 | Voice turn state machine with barge-in, speakable chunking, first-audio budget |
| [Q-045](../quests/Q-045-GPU-SHARE-01.md) · GPU-SHARE-01 | Realtime-first GPU sharing with a non-preemptive EDF admission test and a language-model step cap |
| [Q-046](../quests/Q-046-DRAFT-SPEC-01.md) · DRAFT-SPEC-01 | Draft-model speculation gated by a greedy oracle; draft length and keep/park policy |
| [Q-047](../quests/Q-047-SPEECH-MODELS-01.md) · SPEECH-MODELS-01 | Speech model choice, footprint, and WER gates (STT direct, TTS by round trip) |

**Acceptance (M7):** language model plus speech-to-text plus text-to-speech resident within the
envelope (whole-card peak, desktop reserve, `NO_XID`); zero audio deadline misses over the frozen
voice protocol; first-audio p95 within target; language-model tokens identical to solo at every
cache size; speech WER within frozen thresholds; throughput cost of co-hosting reported, all
samples kept.

### Unallocated slots

`Q-032`, `Q-035` and `Q-036` are reserved by the source audit's numbering but **no workstream,
deliverable or gate is assigned to them**. They exist so the numbering stays stable. Their quest
files say so and authorize nothing; a scope may only be added by a later program revision.

## Milestone ladder

Targets are reviewed after M0's baseline; they are goals, not claims. Effort is the source audit's
rough estimate, not a delivery schedule.

| # | Milestone | Scope | Gate (all must hold) | Rough effort |
| --- | --- | --- | --- | --- |
| **M0** | Oracle + baseline | release applied; Q-025 three-way benchmark; Q-031 harness; Q-030 scaffold | 3 lanes pinned and measured; Pareto table; quality oracle reproducible; D1 decided | 1–2 weeks |
| **M1** | MoE executor v2 | Q-029 kernel; Q-026 dispatcher; integration; profiling | decode ≥ 35 tok/s C1 @1K; prefill ≥ 2× baseline; whole-card peak ≤ 7.0 GiB with ≥ 1 GiB free; quality gate green; all samples kept; D2 decided | 4–6 weeks |
| **M2** | Memory hierarchy | Q-027; Q-028; Q-038 | 32K context on 8 GiB; C4 aggregate ≥ 1.5× C1; cache/eviction receipts; reclaim ±128 MiB | 4–6 weeks |
| **M3** | Scheduler + speculation | Q-019; Q-020; Q-018; Q-039 | ≥ 5 % TPOT or negative; batch-invariance + PIT/KS green; C4 tail receipt | 4–6 weeks |
| **M4** | Supervisor + provider | Q-024b, Q-013-D, Q-014, Q-015, Q-016, Q-040 | two engines, A→B→A, admission denies UNKNOWN, stream/cancel, SLO receipt | 6–8 weeks (parallel) |
| **M5** | Coverage + packaging | Q-033, Q-034 | model matrix receipts, one-command install/server, docs, release | 4–6 weeks |
| **M6** | Stretch | Q-037, AMD port, multi-GPU | only if earlier gates are met | ongoing |
| **M7** | Assistant engine | Q-043, Q-044, Q-045, Q-046, Q-047 | co-resident LM + STT + TTS within the envelope; zero audio misses; first-audio p95 target; tokens identical at every cache size; WER gates; `NO_XID` | after M2 and M4 hooks; policy work in parallel now |

M1 and M2 are the make-or-break phase. M3–M5 can run in parallel once the executor API is stable:
the supervisor lane does not depend on engine internals.

## Decision gates (pre-declared, no retrofitting)

**D1 — substrate (end of M0).** After Q-025:

1. If the current Rust candidate is within 10 % of llama.cpp decode and the gap is attributable to
   the serial route loop: continue the fork (recommended default).
2. If llama.cpp wins by > 25 % and the Rust candidate has no credible path to parity inside M1:
   pivot to "llama.cpp/Strata as the execution engine, Rust owns residency patches + supervisor",
   keeping the Rust executor as research. This is a legitimate outcome and must not be hidden.
3. Never choose a substrate on donor claims; only on the frozen receipt.

**D2 — executor viability (end of M1).** If after Q-026/Q-029 the Rust engine is still below 0.9×
llama.cpp, choose one: (a) port the missing ggml CPU kernel/threadpool pieces (one quarter),
(b) adopt llama.cpp as the execution engine, or (c) narrow the product claim to "best *fits* memory
envelope" and stop claiming raw speed. Record the decision and the negative results.

**D3 — context ceiling (end of M2).** If 32K on 8 GiB is not reachable without quality loss, publish
honest context tiers (for example 8K on 8 GiB, 32K on 12 GiB) and claim no more.

**D4 — speculation (end of M3).** If the MTP width policy or ZOS-lite does not produce ≥ 2 % median
improvement, keep them off by default and publish the negative results.

## What is executable without a GPU today

Cloud and CI sessions have no NVIDIA driver; all GPU runs happen on the lab host (RTX 3070 8 GiB).
Anything that needs a real CUDA forward pass is GPU-gated. A CPU forward pass of the real engine on
the synthetic fixture runs anywhere (`harness/synthetic-moe/`).

| Quest | Testable here (CPU) | GPU-gated on the lab host |
| --- | --- | --- |
| Q-026 | Plan determinism, conservation, parity against a naive per-route oracle, `Upload` rejection | Engine integration, grouped GPU dispatch, Nsight-style profiling |
| Q-027 | Admission/eviction policy, lease and pin invariants, segmented-budget arithmetic, `UNKNOWN` propagation | Real cache hit rate, eviction under pressure, reclaim |
| Q-028 | Scheduler-side unit tests only | Any PagedAttention/batching measurement |
| Q-029 | Q8_0 quantize/dequantize parity, kernel correctness, synthetic-block microbench | Throughput inside a real forward pass |
| Q-030 | Prompt-set freeze, oracle plumbing, receipt schema; reference tokens must be captured once on a GPU host | Reference capture, long-context parity |
| Q-031 | Orchestrator against synthetic receipts; un-pinned identity refusal | One real multi-lane run |
| Q-033 | Loader/format support, tokenizer parity, CPU dense paths | Every functional receipt |
| Q-034 | Build, CLI, config, packaging, reproducibility | Server smoke test |
| Q-037 | None | Upload-stall profile, prefetch gain |
| Q-038 | KV layout arithmetic | Streaming parity, stall budget, 32K on 8 GiB |
| Q-039 | Admission/fairness policy simulation | C1/C4 tails with a real engine |
| Q-040 | Contract conformance against stubs | Two engines actually serving |
| Q-041 | Reserve arithmetic | Desktop-pressure handshake, `NO_XID` |
| Q-042 | Timer compilation, trace schema | Measured timeline of a real request |

## Evidence policy

- **No number without a receipt.** A claim cites a file under `evidence/runs/` that a second person
  can replay from the recorded command line, binary and model hashes, and frozen inputs.
- Receipts keep **all** samples, not just the median; the median is reported alongside them.
- Failed runs, lost samples and negative results are first-class evidence and stay published.
- An identity that cannot be pinned is `BLOCKED` or `UNKNOWN`, never approximated.
- Benchmarks use 1 warmup and ≥ 3 measured repetitions per lane, interleaved, per
  [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).
- Quality is checked **before** timing is counted.
- No GPU work in CI. GitHub Actions never decides whether a result exists.
- A quest's result is a verdict with a definition of done, not a feature count.

## Current state

Strictly what is true at the head of `feat/engine-program` (revised 2026-10-24; the reasoning behind each change is in [FINDINGS.md](FINDINGS.md)). Uncommitted files in a working
tree are not program state: a quest changes state only when its commit lands and, later, when its
receipt exists.

| Item | State |
| --- | --- |
| Release `v0.9.0-rc1` | Exists as a tag (`b2578a0`); identity and gates in [RELEASE.md](RELEASE.md) |
| `crates/gnostral-moe-exec/` (Q-026 policy half) | Implemented and tested offline (13 tests): deterministic grouping, conservation, placement isolation, reference executor, and `gather_group` / `scatter_ordered`, whose `(token, ordinal)` scatter makes a grouped executor bit-equal to the per-route engine loop for any input permutation and group cap |
| Q-026/Q-029 engine integration | Patch [`patches/engine-program/0001`](../patches/engine-program/README.md) over the 20261023 engine tree: grouped executor (default) with the per-route loop kept behind `GNOSTRAL_MOE_EXECUTOR=per-route`, and the gnostral CPU kernel behind `GNOSTRAL_CPU_EXPERT_KERNEL=gnostral`. Compiled and tested on CPU only: 53 MoE tests pass, grouped output bit-equal to per-route. Patch `0002` adds Q-042 timers, an all-CPU residency opt-in and the `gnostral_capture` tool; on a synthetic Qwen3-MoE GGUF ([`harness/synthetic-moe/`](../harness/synthetic-moe/README.md)) `grouped` is token-identical to `per-route` on 32/32 frozen prompts and tracing is token-neutral. Patch `0004` makes the `gnostral` CPU kernel bit-equal to candle, also 32/32 token-identical ([receipt](../evidence/runs/synthetic-moe-e2e-20261023.json)). Patch `0005` adds capture diagnostics (top-N logprobs, prompt subset, prefix cache off). **On the lab host:** CUDA build; synthetic CUDA runs with comparison accepted ([receipt](../evidence/runs/synthetic-moe-e2e-gpu-20261023.json), F-006, F-007); Qwen3-30B-A3B Q2_K loads and runs on CUDA, but the first M1 run stopped on the parity gate because the automatic device map placed 15 or 16 of 48 layers on the card depending on free VRAM (F-009 to F-011). Patches `0006`/`0007` fix the sizing; `0007` is the one on the real load path and awaits the lab's placement report (F-012, F-013). **No qualified real-model measurement** |
| `crates/gnostral-cpu-kernels/` (Q-029) | Implemented and tested offline (48 tests). Q8_0 reference path, plus the K-quant path the engine needs: GGML-layout `Q8_K`/`Q2_K`/`Q3_K` blocks (zero-copy view of GGUF bytes), scalar, AVX2 and AVX-512BW integer cores that are **bit-identical** to each other, a persistent thread pool, `gemm_q8k` and `grouped_expert_ffn` (bit-equal across 1-8 threads and across grouping). Parity against a transcription of the vendored candle `35d7ae7` loops: `Q8_K` quantisation, `Q2_K`/`Q3_K` dequantisation and the `Q2_K` dot are bit-equal; the native `Q3_K` dot is within a tolerance frozen in the test, and `Numerics::CandleCompat` (used by the engine) is bit-equal to transcriptions of candle's AVX2+FMA `Q2_K`/`Q3_K` intrinsics and of `vec_dot_unopt`, on every ISA. `kquant_bench` run on a cloud VM only, next to ggml's `test-backend-ops` (F-021 to F-023); the `CandleCompat` combination is vectorised and stays bit-equal (F-024: per expert layer 5.53 to 2.24 ms at 2 threads on that VM, ggml about 1.02 ms). Covers `Q2_K`/`Q3_K` experts only (F-026). **Not measured on the lab host** |
| `crates/gnostral-expert-residency/` dynamic module (Q-027 policy) | Implemented and tested offline (11 new tests): decayed histogram, ticketed admission, typed refusals, lease/pin-safe eviction with hysteresis, shrink/grow, `UNKNOWN` device bytes. Engine side: patch `0003` records observed expert demand as a loadable static profile (`GNOSTRAL_EXPERT_PROFILE_OUT`), exercised on the synthetic fixture. Dynamic admission is not wired into the engine; no hit-rate receipt |
| `crates/gnostral-memory-plan/` (Q-027/Q-028/Q-038/Q-041 arithmetic) | Implemented and tested offline (13 tests): envelope arbitration refusing an `UNKNOWN` desktop ambient, segmented allocator with elastic return, paged KV blocks with copy-on-write forks, KV tiering that never drops a block, ticketed block moves, desktop-reserve state machine. No device code, no measurement |
| `crates/gnostral-scheduler/` (Q-028/Q-039 policy) | Implemented and tested offline (7 tests): continuous batching with paged-KV admission and recompute preemption, FIFO default and an SLO policy with aging and explicit deadline rejection, deterministic simulation with per-request receipts. Not wired into the engine scheduler; no C1/C4 measurement |
| `crates/gnostral-supervisor/` (Q-040) | Implemented and tested against stub engines (9 tests): per-identity qualification, no capability inheritance, deny-`UNKNOWN` admission, exclusive GPU switching (A to B to A), lifecycle enforcement, crash isolation, streaming and cancel. No real engine has been run through it |
| `crates/gnostral-trace/` (Q-042) | Implemented and tested offline (7 tests): off-by-default per-phase timers, allocation-free recording, wall-time reconciliation (interval union) with explicit `UNKNOWN`, measured overhead, JSON receipt. Wired into the engine by patch `0002` (schedule, forward, sample, cpu_expert); exercised only on the synthetic CPU fixture; no profile of a real model exists |
| `harness/quality/` (Q-030) | Implemented and tested offline (41 tests): parity, KL (CLI default now `1e-3`), PIT/KS, plus the frozen 32-prompt gate set `q030-32-20261023` (`c573a8d8...1787e`), frozen thresholds `q030-gates-20261023`, a divergence ledger and a full `gate` receipt (BLOCKED / FAIL / INCOMPLETE / PASS). Synthetic captures and real-model diagnostic captures exist (F-009, F-010); no gate receipt on a real model; Q-025 prompt-byte reconciliation is open |
| `harness/bench/` (Q-031) | Implemented and tested offline (42 tests): receipt validator, Pareto comparator, receipt encoder and the `run` lane driver (identity pins, 1 warmup, interleaved repetitions, every run kept, blocked lanes abort the comparison only). Run on synthetic CPU and CUDA fixtures, and once on the real model, where the parity stop ended the run before timing (F-009) |
| `crates/gnostral-cohost/` (Q-043) | Implemented and tested offline (7 tests): co-residency broker over the memory-plan envelope, core and elastic budgets, class rules, shrink/park/regrow plans checked step by step against capacity over 200 random sequences, typed refusals that change nothing, receipts. Branch `feat/assistant-engine`; no engine hook |
| `crates/gnostral-assistant/` (Q-044/Q-045) | Implemented and tested offline (9 tests): voice turn state machine with barge-in, lossless speakable chunking, first-audio budget; non-preemptive EDF admission (processor-demand test with blocking) validated by simulation on over 500 admitted random stream sets with zero misses. A first, simpler bound was rejected by that test. No audio, no model |
| `crates/gnostral-speculative/` (Q-046) | Implemented and tested offline (5 tests): greedy speculation oracle token-identical to plain greedy for any draft, acceptance tracker, draft-length and keep-draft policy. No draft model run |
| `harness/quality/speech_oracle.py` (Q-047) | Implemented and tested offline (4 tests): normalisation, WER alignment, corpus gate with `BLOCKED` rules. No clip set frozen, no speech model run |
| Q-037 prefetch | **Deliberately not started**: its gate forbids writing prefetch before a Q-042 profile shows upload stalls matter |
| Q-033, Q-034 | **Not started** (W6, after the engine is fast and correct) |
| Q-032, Q-035, Q-036 | Reserved; no scope assigned |
| `harness/lanes/` (Q-025/Q-031) | Shared lane plumbing and the llama.cpp lane (parity class `self`), tested against a fake server and run once against a real `llama-server` b11857 on the synthetic fixture: placement pinned, self-reproduction on every run ([receipt](../evidence/runs/synthetic-moe-four-lane-cpu-20261024/summary.json)). Not yet run on the real model |
| `harness/model-inventory/` | Header-only GGUF inventory: architecture, bytes per tensor class and type, and which engine path can run each layer's experts (F-026, F-027). Tested offline (12 tests); not yet run on a real model file |
| Architecture coverage | Expert residency, the Q-029 kernels and the `0007` sizing exist for `qwen3moe` only (F-027). Qwen3.5/3.6 MoE (`qwen35moe`), LFM2.5 MoE and gpt-oss load as plain mistral.rs models |
| `harness/diag/repro_capture.py` | Captures the same prompts twice per configuration and compares tokens, logprob bits and the logged device map; produced the F-010 matrix |
| [FINDINGS.md](FINDINGS.md) | Append-only journal of what the program learned, including rejected fixes and process errors |
| Engine speed claims | **None.** Synthetic-fixture timings exist and are labelled as such; no real-model benchmark qualifies |

These crates and harnesses are on branch `feat/engine-program`, after the release tag; they are **not**
part of the tagged `v0.9.0-rc1` tree.

### Branches

| Branch | Holds | Rule |
| --- | --- | --- |
| `feat/engine-program` | W1-W6 work, the engine patch series, the lab's harness and evidence | The lab host and cloud sessions both push here; pull before editing, never force-push |
| `feat/assistant-engine` | W7 (Q-043 to Q-047): `gnostral-cohost`, `gnostral-assistant`, `gnostral-speculative`, the speech oracle | Branched at `ecd2ddf`. `feat/engine-program` is merged **into** it whenever it moves (merge, not rebase: its history is already pushed). It merges back into `feat/engine-program` after the first qualified M1 receipt, so the measurement branch stays small while M1 runs. W7 touches no engine file and no file the lab edits |

Neither branch is merged into `main` or tagged without the owner's explicit decision.

## Non-claims

- No decode, prefill, TTFT, VRAM, context, concurrency or packaging result exists for this program.
  The scorecard is a set of gates.
- The 20.18 / 34.54 / 9.42 tok/s figures are Q-009/Q-010 observations on one host, one model and one
  quantisation; they justify the thesis, they do not predict any quest's outcome.
- Passing a CPU-only policy test proves bookkeeping, not engine performance and not GPU behaviour.
- Nothing here promotes a second engine into the supervisor, changes the Q-007/Q-008 authority
  boundary, changes the Q-012-pinned CUDA runtime, or authorizes GPU work in CI.
- No quest authorizes a model download, a quantization change for benchmarking, or a new CUDA kernel
  family before the execution path is fixed.
