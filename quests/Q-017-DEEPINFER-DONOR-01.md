# Q-017 — DEEPINFER-DONOR-01

Review date: 2026-10-23
State: **REFERENCE_PINNED** — intake and pinning only; no port, no build, no inference.
Authorization: donor *reading* and *specification extraction* only. This quest does not authorize
a DeepInfer runtime, a DeepInfer build, a torch/CUDA install, a new engine dependency, or any
change to the Q-007/Q-008 authority boundary. Any port that follows from this quest needs its own
gate (Q-018…Q-023) before it may touch a runtime.

## Question

Can any DeepInfer mechanism be re-implemented **inside the pinned Rust engine** so that the
result is faster or more predictable *on the RTX 3070 8 GiB / 32 GiB host*, without importing
DeepInfer's datacenter assumptions (Python 3.12 + torch, CUDA 13, SM100 / Ascend 950,
DeepSeek-V4.1-Flash, expert-parallel multi-rank), and without weakening model, routing,
quantization or activation semantics?

This is not an authorization to run DeepInfer, to vendor a buildable dependency, or to make a
datacenter framework the project's runtime.

## Source and current evidence

- Donor: DeepSeek-AI/DeepInfer, Apache-2.0. Public archive with **no Git history**: revision
  `UNRESOLVED`, pinned by archive sha256
  `a2f6f0a42e7bb213bb9e30fe997b6778a467829a4d71b94e783d0e6f6fa8af3d` and full-tree digest
  `30b8f3d319c86e34d14f1979aa288a8534400f754139b85da467d6aa04ee7fc0`.
- Local state: `REFERENCE_ONLY`; no tree vendored, nothing built, nothing imported.
  Manifest entry and inventory as proposed in `manifests/upstreams.toml` /
  `evidence/runs/deepinfer-reference-20261023.json`.
- Donor analysis: `docs/DEEPINFER_DONOR_ANALYSIS.md` (component map, portability, non-portable list).
- Six upstream git submodules (DeepGEMM, DeepEP, FlashMLA, DeepSelect, and the Ascend variants)
  are absent from the archive. No complete dependency provenance exists, and the archive cannot be
  imported even if a build were attempted.
- Local engine context: `vendor/gnostral-strata-worktree/vendor/mistralrs-strata/` already carries a
  paged-attention scheduler, block pool, KV cache manager, llguidance grammar path, CUDA graphs, a
  `speculative/` module (DFlash drafter, proposer, verifier, static `proposal_len`) and a
  windowed KV pool. DeepInfer is a **design reference**, not a missing feature set.
- The donor's own tests are not an offline gate: every file under `tests/` has an import chain that
  reaches `torch` (10 import it directly; `tests/engine/test_flops_estimator.py:14` loads torch before
  `deepinfer._C`, the compiled extension that cannot be built here), so nothing in `tests/` runs on
  this host.

## Execution sequence (bounded, in order)

1. **Pin the reference (done).** Record archive digest, tree digest, per-file hashes, exclusions,
   absent submodules, license bytes; state `REFERENCE_ONLY`.
2. **Extract specifications, not code.** For each candidate mechanism, write the algorithm's
   *observable contract* — inputs, outputs, invariants, the property a test can assert — with
   donor `file:line` citations. No copy-paste into a runtime.
3. **Write the local falsifiable test first.** The port must be expressible as a
   dependency-free Rust test before the port exists. If the property cannot be stated as a test,
   the slice stops at DESIGN_ONLY.
4. **Port only what survives step 3**, in this order (dependency order, not donor rank):
   Q-018 position-keyed randomness + the two determinism gates → Q-019 confidence-scheduled MTP
   verification → Q-020 ZOS-lite deferred post-processing → Q-021 structured-output × speculation →
   Q-022 residency invariants/`BlockLoadPlan` → Q-023 FLOPs accounting. Each is a separate quest,
   a separate branch, a separate gate.
   **Q-018 (a) must land before Q-019**: the local verifier draws one uniform per draft position from
   a stateful stream (`speculative/verifier.rs:480-495`), so dynamic verification length without
   position-keyed draws silently changes RNG streams.
5. **Only after a port's own tests pass**: measure on the frozen A/B protocol
   (`docs/BENCHMARK_PROTOCOL.md`), one warmup and ≥3 measured repetitions per lane, every sample
   retained, median reported, negative results kept.

Full optimistic ZOS scheduling (`csrc/engine/zos.cpp`, `deepinfer/engine/llm_engine.py`) is
**NOT_PORTABLE**: it is inseparable from torch streams, CUDA graph capture and the donor's Python
step loop. Its invariants may inform a future engine note; they do not become a quest. Only the
ported half — one-step-deferred post-processing with pinned staging — is authorized, as Q-020, and
it must measure the CPU gap *before* writing code.

## Required claims and stop conditions

- DeepInfer is a **reference donor**. It is not a dependency, not a second engine, not a fallback
  runtime, and never a provider in EngineProvider.
- No build, no `pip install`, no torch wheel, no CUDA/CANN toolchain change, no model download, no
  GPU run of donor code. A build attempt is a process violation, not a negative result.
- Ported code is gnostral-owned Rust carrying an explicit
  `Derived from deepinfer/<path>, Apache-2.0, © DeepSeek-AI` header; it is **not** relicensed
  under the root MIT license, and the donor's third-party notices (vLLM/EleutherAI/HuggingFace in
  `rotary_embedding.py`; vLLM in `setup.py`) are carried if those files inform a port.
- A port must not change model semantics to manufacture a gain: no change to vocab, tokenizer,
  top-k/top-p/temperature math, routing weights, expert count, quantization or activation
  transform. Batch-invariance is a *bit-identical result* property, not a "close enough" tolerance.
- Engine owns batching, paged KV, draft verification, kernel selection and placement. `modeld`
  receives qualified observations and resource envelopes only (Q-008). Nothing in this quest may
  add placement or eviction authority to the supervisor.
- **Stop the affected slice** (and preserve the negative evidence) if:
  - the mechanism's meaningfulness depends on SM100 kernels, FP8/FP4 KV formats, expert
    parallelism or the V4.1 model geometry;
  - the local engine already implements an equivalent contract and has a passing test for it (then
    the correct outcome is `NOT_PORTABLE` / `ALREADY_COVERED`, not a second implementation);
  - the property cannot be tested without torch, a GPU, or donor fixtures;
  - the port would be correct only by relaxing a semantic invariant;
  - a donor claim cannot be traced to source (mark `UNVERIFIED`, do not port it).
- No speed claim may originate from donor documentation, donor benchmarks, source reading, or the
  vendoring itself. Donor numbers were measured on 4–64 SM100 GPUs.

## Candidate mechanisms and their local gates

| # | Mechanism | Donor evidence | Lane | Local gate (falsifiable) | Negative result |
| --- | --- | --- | --- | --- | --- |
| 1 | Position-keyed sampling randomness (design) | `csrc/engine/prepare_sample.cpp:87,125,161,263`; `csrc/engine/sequence.hpp:96,108`; draw is a TileLang kernel: `tilelang_ops/cuda/random_sampling.py:234,249` | engine | Same tokens + same seed + same sampling params ⇒ identical token stream across two runs **and** across two batch compositions | Divergence at any position ⇒ the keying contract was not the mechanism; stop and record |
| 2 | **Determinism gates** (method transfer — the donor does **not** implement batch invariance; flags go to SM100 vendor kernels, `worker/runtime_config.py:56-67`) | `tests/models/test_batch_invariant.py`; `tests/models/test_sampler.py:44-100`; `environ.py:197` | engine | (a) single vs batched (1/2/4 neighbours) token-identical, greedy and `T=0.8`; (b) randomized PIT + KS against U(0,1) on sampler probabilities | A failing gate is a **finding**, not a stop; either way gnostral must not then describe itself as batch-invariant |
| 3 | Confidence/acceptance-scheduled variable-length MTP verification | `csrc/engine/mtp_varlen.hpp:20,48,133-267`; cumulative rate `csrc/engine/scheduler.hpp:1243-1251`; fixture oracle `tilelang_ops/schedule_mtp_varlen.py:35-98` | engine | A pure-Rust `VerificationPolicy` that, given per-request accept rates and a decode-cost curve, produces per-request verification lengths maximising expected tokens/ms; fixture reproduced exactly, `best_bsz >= num_seqs` held | No confidence head ⇒ measured trailing rate only, stated as a simplification; no gain ⇒ `NOT_QUALIFIED` |
| 4 | ZOS-lite: one-step-deferred post-processing with pinned staging (**not** optimistic ZOS) | `deepinfer/worker/worker.py:183-187`; `deepinfer/engine/llm_engine.py:67-119`; `csrc/engine/prepare_sample.cpp:83-119` | engine | First measure the CPU idle gap (schedule / forward launch / sample launch / post-process) at C1/C4; only if material, defer post-process one step and show TPOT p50/p99 change with output ordering unchanged | gap < ~2 % of step time ⇒ negative result, stop before code; output-ordering/counter regression ⇒ revert |
| 5 | Structured output × speculative rollback | `structure_output/matcher.py:23,206-233`; `sampler.py:366-376`; `csrc/engine/sequence.hpp` `FINISHED_SO_FAILED` | engine | SO-constrained generation with drafts enabled accepts a prefix only if the string stays a prefix of the grammar language; rollback ≤ drafts + 2; one failing request cannot poison the batch | Gap may be `UNVERIFIED`: verify first; if llguidance has no equivalent contract, encode the invariant as a test, not a port |
| 6 | Residency invariants + `BlockLoadPlan` segments | `cache_engine.py:32,88-111,221-307`; `block_load_plan.cpp:23-59` | engine | Segments cover exactly `total_k`; destinations disjoint; ring-wrap ordering equals linear; `src_block_id = -1` sentinel; one upload as a segment list is byte-equal to per-slice copies | If the plan adds allocation without removing syncs, report the abstraction as not worth it |
| 7 | Scheduler-time FLOPs / MFU accounting | `flops_estimator.hpp:239-257` | evidence | Closed-form pair counts agree exactly with brute force across `(q,k,window)` including capped/chunked; MFU matches a hand calculation on one step; published as `DERIVED` only | Cannot agree within ~1 % on a real step ⇒ keep pair-count only, MFU unqualified |
| 8 | YoCo/SWA bounded replay | `environ.py:187-194`; `v4_modules/yoco.py:14-21` | engine | Requires YoCo consumer layers; not applicable to non-YOCO models | `NOT_PORTABLE` (model-gated) — no local test is possible |
| 9 | Full optimistic ZOS scheduling | `csrc/engine/zos.cpp`; `llm_engine.py:961-1000`; device-side `seqlens_q/k` correction | engine | Not expressible without torch streams + CUDA graphs + SM-gated correction kernels | `NOT_PORTABLE` — Q-020 authorizes only the deferred-post-process half |
| 10 | EPLB | `model_executor/eplb/policy.py:12-140`; `eplb/shuffle.py` | engine, multi-rank only | Would need ≥2 ranks and symmetric memory | `NOT_PORTABLE` on a single consumer GPU |

**Donor-claim discipline.** The README's "asynchronous scheduling … enabled by default" is false
(`DEEPINFER_ZERO_OVERHEAD_SCHEDULING` defaults to `False`, `environ.py:70-71,179`), and
"batch-invariant" / "deterministic sampling" are vendor-kernel flags, not DeepInfer capabilities
(`worker/runtime_config.py:56-67`). A donor claim may not be copied into a quest's required claims
without its source and its verification status.

Any mechanism reclassified after reading gets its classification recorded with the reason. A
negative classification is a first-class outcome of this quest, not a failure to finish it.

## Outcome vocabulary

`REFERENCE_PINNED` (now — digests recorded, nothing built); `DESIGN_ONLY` (contract extracted, no
port authorized or no test possible); `PORTED_TESTED` (Rust implementation with a passing
dependency-free test, no performance claim); `PORTED_MEASURED` (frozen A/B, ≥3 samples/lane,
retained evidence); `ALREADY_COVERED` (local engine already satisfies the contract and has a test);
`NOT_PORTABLE` (blocked by hardware, model, torch coupling or rank topology — reason recorded);
`NOT_QUALIFIED` (port attempted, semantic/numerical gate failed).

## Evidence list (this quest)

1. `evidence/runs/deepinfer-reference-20261023.json` — archive sha256, full-tree digest,
   336 retained file hashes, 6 exclusions, absent submodules, license bytes, build blockers.
2. `docs/DEEPINFER_DONOR_ANALYSIS.md` — component map, portability matrix, absorption plan,
   non-absorb list.
3. `manifests/upstreams.toml` — `name = "deepinfer"` `state = "REFERENCE_ONLY"`,
   `revision = "UNRESOLVED"`, digests, inventory path.
4. `THIRD_PARTY.md` — project row and the reference-only paragraph.
5. `vendor/deepinfer/README.md` — layout, exclusions, build isolation, verification commands.
6. Per-mechanism quests (when opened): Q-018 (seeding + gates), Q-019 (MTP length), Q-020 (ZOS-lite),
   Q-021 (SO × speculation), Q-022 (residency), Q-023 (FLOPs) — each with its own failing test first;
   Q-024 (provider v1) lives on the supervisor lane: its contract lands before Q-013-D, its first
   provider implementation only after Q-013-D authenticates lifecycle.
7. Cross-check note: `research/runtime/UPSTREAM-INTAKE-20261021.md` records DeepInfer under the
   donor-intake discipline (exact upstream identity, upstream fact vs local observation, explicit
   non-claims); this quest is the formal intake that note requires.

## Non-claims

- No DeepInfer code has been compiled, imported or executed here; no torch, CUDA 13, CANN or
  DeepSeek-V4.1-Flash artifact exists on this host.
- No vendored tree, no dependency, no reproducibility claim over the six absent submodules.
- No performance, quality, latency or throughput claim for any DeepInfer mechanism.
- No revision/commit claim. `UNRESOLVED` means unresolved.
- No change to EngineProvider v0 authority, to the Q-013D→Q-016 critical path, or to the
  Q012-pinned CUDA runtime.
