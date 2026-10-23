<!--
PROPOSED DROP-IN FILE
Target: docs/DEEPINFER_DONOR_ANALYSIS.md
Style reference: research/runtime/STRATA-ABSORPTION-PROPOSAL.md
Quest: quests/Q-017-DEEPINFER-DONOR-01.md
Inventory: evidence/runs/deepinfer-reference-20261023.json
-->

# DeepInfer donor analysis — what is portable into the native Rust engine, and what is not

**Date:** 2026-10-23 · **State:** REFERENCE_PINNED; no port, no build, no inference.
Donor: DeepSeek-AI/DeepInfer, Apache-2.0, `REFERENCE_ONLY`.

## Intent

The user wants gnostral.rs to stay **local-first, versatile and flexible**. DeepInfer is a
DeepSeek datacenter serving framework. The only defensible reason to read it is to extract
mechanisms that make the *local* engine faster or more predictable; the indefensible reason is to
treat it as a runtime or a dependency. This document records which is which, with the exact donor
locations, the local seam each mechanism would land on, and the negative finding where a mechanism
does not survive.

Success is **not** "DeepInfer features are present". Success is one measurable, semantically
neutral property gained on the RTX 3070 host, with a test that would have failed before.

## 1. What DeepInfer actually is (and why it is not a dependency)

| Gate | Fact | Evidence |
| --- | --- | --- |
| License | Apache-2.0, `LICENSE` verbatim, no `NOTICE` | `LICENSE` (11 357 B, sha256 `c71d239d…`) |
| Language/runtime | Python 3.12 + `torch>=2.7`; build is a **torch C++ extension** | `pyproject.toml`, `setup.py` |
| Toolchain | GCC ≥ 13, CUDA 13 (NVIDIA) or CANN 9.2 (Ascend) | `README.md` prerequisites |
| Hardware | NVIDIA **SM100** or Ascend **950** | `README.md`; `requirements-nvidia.txt` (`torch==2.13.0+cu132`) |
| Model | **DeepSeek-V4.1-Flash only** | `README.md`; `deepinfer/hugging_face/processor.py:24` raises unless `model_type=deepseek_v41` |
| Completeness | `vendor/` submodules empty; `setup.py` compiles only `csrc/`; runtime imports `deep_gemm`, `deep_ep`, `flash_mla`, `deep_select` | `.gitmodules`, `setup.py`, import sites |

Two consequences that decide the vendoring design:

1. **There is no independently compilable C++ subset.** 20 of 33 files under `csrc/` include
   `<torch/…>` and 23 use pybind11 — including the files that look algorithmic
   (`csrc/engine/scheduler.hpp:1-6`, `block_manager.hpp:3`, `sequence.hpp:3-9`). A "just vendor the
   C++ core" option buys inspection bytes and **no build**.
2. **The archive is not a complete dependency set.** Six git submodules are absent and a
   zip-without-`.git` carries no gitlink revisions, so their revisions are unrecoverable. A
   `PINNED_UPSTREAM` claim over DeepInfer would assert byte-identical reproducibility that cannot
   be demonstrated.

There are **three independent hard gates**, not one soft check:

1. **Init-time hardware gate.** `is_sm_major_10()` is only a classifier, but `get_bf16_tflops_limit()`
   **raises** `RuntimeError("unknown GPU type")` for any non-`sm_major==10` NVIDIA GPU
   (`deep_cluster/hardware/device.py:151-156`), and `LLMEngine.__init__` calls it while building its
   `StatsLogger` (`deepinfer/engine/llm_engine.py:304`). The engine therefore fails at construction,
   before any kernel is reached.
2. **Kernel gate.** The model path imports SM100-class vendor libraries (absent from the archive) and
   TileLang kernels using Hopper+/Blackwell-only intrinsics.
3. **Model gate.** V4.1-specific formats and geometry (e.g. paged FP8/FP4 KV,
   `tilelang_ops/cache_common.py:109-160`).

A reader who only sees `is_sm_major_10()` returning `False` without an assert may conclude the gate is
cosmetic. It is not: gate 1 fires at init.

## 2. Component map and portability

Portability is judged for **re-implementation in Rust inside `vendor/mistralrs-strata`**, not for
compiling the donor.

| DeepInfer component | What it actually does | Evidence | Local seam | Portability | Local engine already has |
| --- | --- | --- | --- | --- | --- |
| `csrc/engine/mtp_varlen.hpp` | Per-request verification length from predicted accept rates + measured forward/sampler cost; two-layer median runtime estimator | `mtp_varlen.hpp:20,48,133,139,150` | `speculative/policy.rs`, `speculative/driver.rs` | **High** — std-only C++, algorithmic | static `proposal_len` (`policy.rs:1-25`); accept stats collected but unused (`driver.rs:576-598`) |
| `csrc/engine/scheduler.hpp` | Continuous batching, preemption, chunked prefill, ZOS bookkeeping | `scheduler.hpp:1-120, 698-733, 917, 1151, 1214` | `paged_attention/scheduler.rs` | **Low** — torch/pybind types throughout | paged scheduler, chunked prefill, prefix cache |
| `csrc/engine/block_manager.hpp` | Free-list block allocator with refcounts; **no prefix cache** | `block_manager.hpp:24-242` | `paged_attention/block_pool.rs`, `kv_cache_manager.rs` | **None** — local is a superset | dynamic block pool, block hashing, KV manager |
| `csrc/engine/sequence.hpp` | Sequence/SequenceMetadata/status, per-request `seq_seed` | `sequence.hpp:36-108` | `sequence.rs` | Low (types) / **High** (the `seq_seed` idea) | `request.rs:420 seed`, `sequence.rs:888-919 sampling_rng` |
| `csrc/engine/zos.cpp` | "Zero-overhead" input preparation for async scheduling | `zos.cpp:1-307`; `llm_engine.py:826-1000` | `paged_attention/scheduler.rs`, CUDA graphs | **Low** — inseparable from torch streams + graph capture | CUDA graph runner, jitter-ordered batch |
| `csrc/engine/prepare_sample.cpp` | Packs sampler inputs incl. `seq_seeds` | `prepare_sample.cpp:87,125,161,263` | `sampler.rs:2127` | **Medium** (the seeding contract) | shared fallback RNG + per-seq seeded stream |
| `deepinfer/model_executor/layers/sampler.py` | Temperature/top-p/top-k, SO bitmask, MTP rejection | `sampler.py:138-201,301-360` | `sampler.rs` | **Medium** — semantics, not code | sampler + llguidance constraints |
| `deepinfer/structure_output/` | xgrammar wrapper, bitmask fill, SO state machine | `matcher.py:23,120-210`, `compiler.py` | `pipeline/llg.rs` | **Medium** — the SO×speculative invariants | llguidance parser factory |
| `deepinfer/model_executor/eplb/` | Redundant-expert replication + rank packing | `eplb/policy.py:9,78`; `eplb/shuffle.py` | `crates/gnostral-expert-residency` | **None locally** — needs EP ranks, symmetric memory | budget-ranked local residency (different problem) |
| `deepinfer/worker/cache_engine.py` | KV cache alloc/free, swap, V4.1 FP8/FP4 formats | `cache_engine.py` (978 lines) | `paged_attention/kv_cache_manager.rs` | **None** — model-format-specific | generic paged KV |
| `deepinfer/ve/` | Vision/multimodal preprocessing | `ve/multimodal_processor.py` | — | **None** — out of scope | — |
| `deep_cluster/hardware/` | Device discovery, NUMA, soft classification | `device.py:140-170` | — | None | — |

## 3. What the local engine already covers (positioning)

The Strata worktree is not a blank engine. Before any port, these exist and have their own tests:

- paged attention scheduler with chunked prefill and prefix cache — `paged_attention/scheduler.rs`
- block pool, block hashing, KV cache manager, windowed KV pool —
  `paged_attention/{block_pool.rs,block_hash.rs,kv_cache_manager.rs,windowed_pool.rs}`
- a `speculative/` module: DFlash drafter (`speculative/dflash.rs`, 4 715 lines), proposer,
  verifier, static `proposal_len` policy, acceptance statistics
- llguidance structured constraints — `pipeline/llg.rs`
- deterministic greedy parameters — `sampler.rs:110` (`SamplingParams::deterministic`)

Therefore the honest framing is: **DeepInfer contributes specifications and one or two algorithms,
not features.** A mechanism whose contract the local engine already satisfies is `ALREADY_COVERED`
and gets a test if it lacks one — not a second implementation.

## 4. Absorption plan

Seven slices, in dependency order. Each is its own quest, branch and gate; none may weaken model
semantics. The quest ids follow dependency order, not donor rank; §4.8 maps them to the parallel
donor analysis's ranking.

### Q-018 — position-keyed sampling randomness **and** the two determinism gates (do this first)

**Property to gain (a):** the *k*-th random draw for a request is a pure function of
`(request seed, absolute sample position, stream id)` and is therefore unaffected by batch
composition, by speculative decoding being on or off, and by how many draft positions were
verified.

**Donor:** `csrc/engine/prepare_sample.cpp:87,125,161,263` (per-request `seq_seed` packed with the
request), `csrc/engine/sequence.hpp:96,108` (`sampling.seq_seed`), and the position-keyed draw in
`deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py:234,249`
(`sample_offset(cfg, acc_frag, seq_seeds[row], seq, …)` — seed **and** position are inputs, not
stream state). **The draw itself is a TileLang CUDA kernel and is not absorbable**; what transfers is
the keying contract, which we re-implement in Rust.

**Local gap, precisely:** `vendor/mistralrs-strata` already gives each sequence an *optional*
seeded stream (`request.rs:420`, `sequence.rs:888-919`), but it is a **stateful** `Isaac64Rng`: the
verifier draws one uniform per draft position at verification time
(`speculative/verifier.rs:480-495`, `for _ in 0..candidate.drafts { rng.random::<f32>() }`). Draw
count is therefore data-dependent. With today's fixed `proposal_len` that is stable; **the moment
Q-019 makes verification length dynamic, the stream shifts and speculative on/off stops being
reproducible from the same seed.** This is why Q-018 precedes Q-019 and not the reverse.

**First slice (a):** add `seed: Option<u64>` at the request boundary and a pure
`fn draw(seed: u64, stream: StreamId, position: u64) -> f32` (SplitMix64/Philox-style) replacing
the stream advance for acceptance uniforms; keep the existing behavioural path behind the same
default so no output changes for unseeded requests.

**Property to gain (b) — the gates:** two falsifiable properties transcribed as tests, *not*
capabilities. DeepInfer does **not** implement batch invariance or deterministic sampling: it
forwards flags to vendor kernels (`worker/runtime_config.py:56-67`:
`deep_gemm.use_deterministic_algorithms(True)`, `tile_kernels.use_deterministic_algorithms(True)`;
`attention_backend.py:141`), and those kernels are SM100-gated and absent. The donor's contribution
is the *test method*:

- **batch-invariance gate** — same prompt, same seed; single vs batched (1/2/4 unrelated requests);
  greedy and `T=0.8`; token-identical or fail (donor: `tests/models/test_batch_invariant.py`);
- **sampler-distribution gate** — collect `sum_smaller_probs` and token log-probs over N
  trajectories, randomized PIT then KS against U(0,1), including the `ssp − p ≥ −0.01` sanity check
  (donor: `tests/models/test_sampler.py:44-100`).

**Gate:** (a) dependency-free Rust test — two runs, same seed, same params, different batch
neighbours and speculative on/off, produce identical token ids **and** identical logprobs for every
position; (b) both gates pass on the current engine (they become regression gates) or the smallest
failing case is recorded as a finding.

**Stop condition:** if identical token ids require changing the kernels' reduction order, this is a
*determinism* research slice (Q-018b), not a port; report `CORRECTNESS_ONLY`. If a gate fails, that
is a result, not a blocker — but gnostral must not then describe itself as batch-invariant.

### Q-019 — confidence/acceptance-scheduled verification length

**Property to gain:** per-request verification length chosen to maximise expected accepted
tokens/ms from *measured* per-position accept rates and *measured* forward/sampler cost, instead of
a constant `proposal_len`.

**Donor:** `csrc/engine/mtp_varlen.hpp:133-267` — `MTPVarlen::update_runtime` keeps a
two-level median estimator of forward cost per graph size and sampler cost per batch size
(`:139-147`), and `schedule` sorts predicted accept rates, walks the throughput curve, and assigns
`seqlens_q` per request (`:150-260`). Its two named invariants are worth carrying verbatim:
confidence at position *i+1…n* must not influence whether draft *i* is verified; and the
best-batch-size decision uses the *N−2* round accept rate while the per-request length uses *N−1*
(`mtp_varlen.hpp:163-207`). The rate is a **cumulative product** of per-token confidence
(`scheduler.hpp:1243-1251`), not a per-token value. A pure-torch reference of the on-device
allocation exists and is the natural fixture oracle: `tilelang_ops/schedule_mtp_varlen.py:35-98`.

**Local seam:** `speculative/policy.rs:1-40` (`SpeculativeBatchPlan`, `SpeculativeGraphPlan`),
`speculative/target.rs:185-190` (`speculative_graph_plans`), `speculative/driver.rs:576-598`
(already computes `accepted_per_pos` and emits `SpeculativeBatchObservation` — the observation
exists; the feedback loop does not).

**First slice:** a pure `VerificationPolicy` (ideally its own small crate): inputs
`(predicted_accept_rates, decode_cost_curve, sampler_cost, batch_size)`, output per-request
`verification_len`. Port the two estimators and the schedule walk; transcribe the fixture table by
hand (no torch import); preserve the tie-break (rate desc, then `seq_idx` asc) and the
`best_bsz >= num_seqs` invariant (`mtp_varlen.hpp:260`).

**Gate:** property tests on constructed rate/cost tables (monotone: higher predicted accept rate ⇒
length never decreases at equal cost; empty curve ⇒ all-ones fallback, matching
`mtp_varlen.hpp:153-158`); fixture reproduction; then A/B on fixed prompts at C1/C4 with acceptance
rate, tokens/step and the policy trace recorded.

**Stop condition:** if predicted accept rates cannot be obtained without a confidence head (the
donor has one in its MTP model; the local DFlash path may not), the slice degrades to "use the
trailing measured accept rate" — allowed, but must be stated as a simplification, and a
divergence from the donor's estimator recorded. If the policy cannot beat static `proposal_len` on
the frozen A/B, the outcome is `NOT_QUALIFIED` and the static policy stays.

### Q-020 — ZOS-lite: one-step-deferred post-processing with pinned staging

**Property to gain:** remove the CPU bubble between a forward launch and the next schedule by
deferring step N's post-processing into step N+1, keeping that step's metadata alive and releasing
pooled objects one step late — **without** optimistic scheduling.

**Donor:** the deferred branch at `worker.py:183-187`; the one-step `DeferredRelease` for pools and
pinned-clone unpins at `llm_engine.py:67-119`; pinned CPU staging built by
`prepare_sample.cpp:83-119`. This is the portable half of "zero-overhead scheduling".

**Local seam:** `engine/cuda_decode.rs:1-90` (a completion worker thread behind a oneshot future),
`pipeline/execution.rs:24-33` (`StepLookahead::OneToken`). The local engine already defers *decode
completion*; the open question is whether *post-processing* is still on the step's critical path.

**First slice — measurement before code:** instrument the step loop with CPU timers
(schedule / forward-launch / sample-launch / post-process) and measure the idle gap between forward
launch and the next schedule at C1 and C4. **If the gap is already < ~2 % of step time, stop: the
negative result is obtained for free.**

**Gate (if the gap is material):** CPU-gap series before/after; TPOT p50/p99 at C1/C4; per-request
output ordering unchanged.

**Stop condition:** output-ordering or counter regression ⇒ revert. The current borrow contracts are
"valid until the next schedule" (`scheduler.hpp:1128-1132`, `outputs.hpp:176-177`), and a deferred
post-process extends those lifetimes by one step. **Explicit non-goal:** optimistic scheduling
(scheduling N before N−1's tokens are known) requires device-side `seqlens_q/k` correction and is
`NOT_PORTABLE` for this engine; it must not be smuggled into this quest.

### Q-021 — structured output under speculative verification

**Property to gain:** with drafts accepted in a block, an SO-constrained request accepts a prefix
only if the resulting string is still a valid prefix of the grammar language; the rollback bound is
never exceeded; one failing request cannot poison the batch.

**Donor:** `deepinfer/structure_output/matcher.py:23` (`MAX_ROLLBACK_TOKENS =
DEEPINFER_USE_NEXTN_DECODE + 2`), `matcher.py:206-233` (per-request matcher, special-token
rejection), `compiler.py:60-70` (xgrammar error classification), `csrc/engine/sequence.hpp`
`FINISHED_SO_FAILED`, `deepinfer/model_executor/layers/sampler.py:366-376` (`so_status` state
machine, `-1` = kill this request only).

**Local seam:** `pipeline/llg.rs` (`TopLevelGrammar`, `ParserFactory`), `sampler.rs` constraint
handling. **The gap is UNVERIFIED**: llguidance may already provide the needed prefix semantics.

**First slice:** verify the gap, then encode the invariants as tests — (a) after a rejected draft,
the grammar parser state equals the state before the draft was proposed (no forward-commit);
(b) tokens rolled back ≤ `max_draft_len + 2`; (c) an unparseable SO request terminates only itself.

**Stop condition:** if llguidance's API cannot express "is this a valid prefix" without advancing
state, the outcome is `DESIGN_ONLY` plus a guard test, not a port.

### Q-022 — residency invariants and the `BlockLoadPlan` segment compiler (WAKEKV-facing)

**Property to gain:** four small checkable rules rather than an implementation — (i) persistence only
at a grain boundary (`KVSTORE_GRAIN = 128`, `cache_engine.py:32`) so a stored prefix is always
self-consistent; (ii) retain `window + max_backtracking` headroom (`cache_engine.py:88-111`) so a
persisted window can be *resumed*, not merely stored; (iii) partition per-sequence state into
stored/dropped regions so "persist" is a canonical byte-prefix truncation (`cache_engine.py:221-307`);
(iv) compile a page-table or ring range into a DMA segment list once on CPU (`block_load_plan.cpp:23-59`).

**Local seam:** `crates/gnostral-expert-residency` (or a sibling crate); the WAKEKV rail in
`research/runtime/UPSTREAM-INTAKE-20261021.md`.

**First slice:** a pure-Rust experiment taking a synthetic paged page-table + `block_size` and
producing the segment list; assert segments cover exactly `total_k`, destinations are disjoint, a
ring range wraps with the same destination ordering as the linear case, and zero-fill emits the
documented sentinel (`src_block_id = -1`). Then express one upload as a single segment list and
verify byte-equality against the current per-slice copies (parity, not performance).

**Stop condition:** if the existing copy path is already one `memcpy` per slice and the plan adds
allocation without removing syncs, report that the abstraction is not worth it.

### Q-023 — scheduler-time FLOPs/MFU accounting

**Property to gain:** an exact, model-derived predicted-work number per step, from which a
defensible queueing/admission model follows for Q-015 and Q-016.

**Donor:** `flops_estimator.hpp:239-257` — closed-form pair counts including the capped-prefix
identity `capped_prefix(n,r,cap) = floor_prefix(min(n, r·cap), r) + (n − min(n, r·cap))·cap`
(`:244-249`).

**First slice:** implement the pair counts for gnostral's current model in Rust; compare against a
brute-force O(q·k) count over a matrix of `(q, k, window)` shapes including capped and chunked
cases.

**Gate:** exact agreement on all shapes; a per-step "predicted FLOPs / achieved FLOPs" line in the
run record; MFU matching an independent hand calculation on one measured step.

**Hard rule:** never use the estimate as an oracle for correctness or for a benchmark number; it is
published as `DERIVED`, exactly as `contracts/ENGINE_PROVIDER_V0.md` requires ("Estimates and
physical observations are separate facts").

**Stop condition:** if the estimator cannot agree within ~1 % on a real step (MoE routing and
quantization make achieved work hard to observe), keep the pair-count part only and record the MFU
metric as not qualified.

### Mapping to the parallel donor analysis

| Donor ranking (retained with the lab intake record) | Rank | This plan's quest |
| --- | --- | --- |
| acceptance-scheduled variable-length MTP verification | #1 | **Q-019** |
| ZOS-lite one-step-deferred post-processing | #2 | **Q-020** |
| persistence/residency invariants + `BlockLoadPlan` | #3 | **Q-022** |
| scheduler-time `FlopsEstimator` accounting | #4 | **Q-023** |
| batch-invariance + sampler PIT/KS **gates** | #5 | **Q-018 (b)** |
| structured output × speculation (their rank 6) | #6 | **Q-021** |
| position-keyed randomness contract | — (additive) | **Q-018 (a)** |
| provider-boundary versatility | — (out of engine lane) | **Q-024** (supervisor) |

This document was reconciled on 2026-10-23 with the longer donor analysis retained in the lab's
private audit record: its #1–#5 are adopted; its correction that batch-invariance/determinism are
*gates, not donor capabilities* is incorporated verbatim above; its finding that the non-SM100
hardware gate fires at `LLMEngine.__init__` corrected §1 of this document. The audit record itself is
not republished here; the machine-readable intake is `evidence/runs/deepinfer-reference-20261023.json`.

## 5. What must **not** be absorbed

- **SM100/Ascend kernels and KV formats** (FlashMLA, DeepGEMM, DeepEP, TileLang ops, FP8/FP4 V4.1
  cache layouts) — hardware- and model-specific; absent from the archive anyway.
- **The DeepSeek-V4.1 model code** (`model_executor/models/v4_modules/`, `models/mtp/`,
  `hugging_face/`) — a different model family; nothing to reuse.
- **Full optimistic ZOS scheduling as code** (`zos.cpp`, `llm_engine.py` step loop; device-side
  `seqlens_q/k` correction) — torch-stream and CUDA-graph coupled, and the correction kernel is
  SM-gated. `NOT_PORTABLE`. **Exception:** the deferred-post-process half is authorized as **Q-020**
  (ZOS-lite) because it is a lifetime change inside the engine, not a scheduling-model transplant;
  Q-020 must measure the CPU gap before writing code and must not smuggle optimistic scheduling in.
- **EPLB** — needs expert-parallel ranks and symmetric memory. On one consumer GPU the analogous
  question is *residency*, already answered locally by `crates/gnostral-expert-residency` with a
  different, provable contract. Recorded as `NOT_PORTABLE`.
- **The torch/pybind/C++ layer as a shortcut.** "Use the donor's C++ via FFI" is the same mistake
  Q-009 rejected for Strata approach 2, with a worse ledger: the donor C++ cannot compile without
  torch.
- **Any donor benchmark number.** Prefill was measured on 4 SM100 GPUs and decode on 64; those
  figures say nothing about an 8 GiB RTX 3070 and must never appear as local evidence.

## 6. Why the project is `REFERENCE_ONLY` (and the fallback is an unmodified excerpt set)

`docs/VENDORING_POLICY.md` sets the default: *"Prefer REFERENCE_ONLY plus patches and manifests.
Vendor a full source tree only when byte-identical reproducibility materially benefits the
experiment."* Note the two claims are different: **the project** is `REFERENCE_ONLY` (no tree), while
the **fallback** for a specific port review is a `PINNED_UPSTREAM` *unmodified excerpt set* — pinned,
because nothing is changed; a subset, which is a scope fact rather than a modification. Strata earned `PINNED_UPSTREAM` because it is a buildable local-benchmark engine with
a commit id and no missing dependencies. DeepInfer fails all three tests:

1. no commit id (archive without `.git`) — `revision = UNRESOLVED`;
2. no complete dependency set (six absent submodules with unrecoverable revisions);
3. not buildable on the target, and its C++ is not portable C++.

What *is* needed is byte-stable donor identity for review of Q-018…Q-023, and that is
achieved by digest pinning: archive sha256 `a2f6f0a4…`, full-tree digest
`30b8f3d319c86e34d14f1979aa288a8534400f754139b85da467d6aa04ee7fc0`, 336 per-file hashes in
`evidence/runs/deepinfer-reference-20261023.json`. If a reviewer later needs the donor bytes
inside the repository, the fallback is a `PINNED_UPSTREAM` *unmodified excerpt set* under
`vendor/deepinfer-reference/` with a `NOT_BUILDABLE` README — never a full tree that implies a
dependency. The state is `PINNED_UPSTREAM`, not `PATCHED_UPSTREAM`: the policy defines the latter as
upstream material **plus explicitly identified local patches** (`docs/VENDORING_POLICY.md:9`), and an
unmodified subset has none. Being a subset is a scope fact (`snapshot_scope` + inventory), not a
modification.

## 7. Provenance discipline for ports

```rust
// Derived from deepinfer/csrc/engine/mtp_varlen.hpp (Apache-2.0, © DeepSeek-AI).
// Re-implemented in Rust; see docs/DEEPINFER_DONOR_ANALYSIS.md and Q-019.
// Distributed under Apache-2.0; not relicensed under this repository's MIT license.
```

Every ported file carries the donor `path`, the license, the quest that authorized it, and the
donor `file:line` for the algorithm it derives from. A port that copies an upstream third-party
notice (vLLM/EleutherAI/HuggingFace in `rotary_embedding.py`) carries that notice too.

## 8. Qualification gate and completion boundary

A slice is qualified only with: a test that fails before and passes after; no change to vocab,
tokenizer, sampling math, routing, expert count, quantization or activation transform; and, for any
performance claim, the frozen A/B protocol from `docs/BENCHMARK_PROTOCOL.md` (one warmup, ≥3
measured repetitions per lane, all samples retained, medians reported, negative results kept).

**Current completion boundary:** donor identity is pinned, the component map is read from source,
the portability classification is recorded, and the local seams are identified by `file:line`. No
port, no build, no import, no GPU run, no performance claim. Async scheduling and EPLB are recorded
as non-portable with reasons.

## 9. Non-claims

- Not run, not built, not imported. No torch, CUDA 13, CANN or V4.1 artifact exists here.
- No dependency, no second engine, no provider. DeepInfer never becomes an EngineProvider.
- No performance claim of any kind, for any mechanism, in either direction.
- No revision claim: `UNRESOLVED` is the recorded state, with the resolving method documented.
- No statement about gnostral.rs's authority boundary: it is unchanged. Engine owns
  materialization; the supervisor owns intent, lifecycle and resource envelopes.
