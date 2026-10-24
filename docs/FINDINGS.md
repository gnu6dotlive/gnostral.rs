# Findings journal — engine program

What we learned, in the order we learned it, including what we got wrong. This page exists so
that a later session (human or not) inherits the reasoning, not only the code: a conversation is
summarised and forgotten, a commit message is terse, this page is not.

Rules:

- **Append-only.** An entry is never rewritten to look better. A correction is a new entry that
  names the one it corrects; the corrected entry gets a one-line `Superseded by` pointer and
  nothing else.
- **Claim class** from [EVIDENCE_MODEL.md](EVIDENCE_MODEL.md) on every entry: `PROVEN`,
  `OBSERVED`, `SUPPORTED`, `INFERRED`, `UNKNOWN`, `REJECTED`. Process findings (how we worked, not
  what the engine does) are marked `PROCESS`.
- **Evidence** is a path in this repository. Evidence that exists only on the lab disk is written
  `lab-only:` and listed under [Evidence not yet in the repository](#evidence-not-yet-in-the-repository);
  it is not program evidence until committed.
- Numbers come from receipts. A number quoted without a receipt is a defect in this page.

## Entries

### F-001 · 2026-10-23 · Native `Q3_K` numerics change late greedy tokens on the synthetic fixture

- **Class:** OBSERVED (cause: SUPPORTED by F-002).
- **What:** on the synthetic Qwen3-MoE GGUF (CPU, 32 frozen prompts x 32 tokens), `grouped` was
  token-identical to `per-route` on 32/32 prompts; the `gnostral` CPU kernel with
  `Numerics::Native` on 30/32 (`q12-de-translate` at token 18, `q19-summary-tech` at token 15).
- **Evidence:** `evidence/runs/synthetic-moe-e2e-20261023.json`, block `gnostral_before_candle_compat`.
- **Consequence:** the native kernel is not a drop-in for candle on the parity gate.

### F-002 · 2026-10-23 · `Numerics::CandleCompat` is bit-equal to the vendored candle kernels

- **Class:** PROVEN for the public candle `35d7ae7` reconstruction; UNKNOWN for the lab's
  candle-strata until a lab parity receipt exists.
- **What:** integer lanes in candle's AVX2 order, per-lane `fma`, `hsum_float_8`, or
  `vec_dot_unopt` without AVX2+FMA. Unit tests transcribe candle's intrinsics; the engine test is
  bit-equal; the synthetic end to end went from 30/32 to 32/32.
- **Evidence:** `crates/gnostral-cpu-kernels/tests/kquant.rs`, engine patch `0004`,
  `evidence/runs/synthetic-moe-e2e-20261023.json`.

### F-003 · 2026-10-23 · The upstream CPU MoE path is a different computation

- **Class:** SUPPORTED (source inspection, not a numerical test).
- **What:** without residency, `qtensor_indexed_moe_forward` dequantises whole expert stacks to F32
  and keeps F32 activations; the residency path quantises activations to `Q8_K`. 12/32 prompts
  token-identical on the fixture.
- **Consequence:** never compare a residency lane to the upstream CPU path under an exact-parity gate.

### F-004 · 2026-10-24 · The first GPU-sharing admission bound was wrong

- **Class:** REJECTED (the bound), PROVEN (that it was wrong: a counterexample).
- **What:** `U + B / P_min <= 1` ignored long audio jobs blocking short ones; the randomized
  simulation produced three missed deadlines on an admitted set. Replaced by the processor-demand
  test for non-preemptive EDF with blocking.
- **Evidence:** branch `feat/assistant-engine`, `crates/gnostral-assistant/tests/assistant.rs`,
  [Q-045](../quests/Q-045-GPU-SHARE-01.md).

### F-005 · 2026-10-24 · A test edit silently failed to apply

- **Class:** PROCESS.
- **What:** a scripted string replacement adding the assertion "more than 500 admitted sets were
  exercised" did not match after `rustfmt` had reformatted the line, so the assertion was absent
  while the quest text already stated the figure. Found and fixed before the commit; the published
  figure is checked by the test.
- **Rule adopted:** every scripted edit asserts that its target exists; no figure is written into a
  document before the test that checks it has run.

### F-006 · 2026-10-23 · First CUDA synthetic run: comparison accepted, numbers meaningless

- **Class:** OBSERVED.
- **What:** 2 layers, 16 experts, `--device cuda`: driver `NO_XID`, three lanes pinned, parity PASS
  on 12/12 runs, comparison `COMPARED`. Decode ranged 235 to 8072 tok/s within one lane.
- **Why it measured little:** `GNOSTRAL_EXPERT_CACHE_BYTES=1` with an empty profile keeps every
  expert on the CPU even with `--device cuda`; about 0.3 s of decode per capture is dominated by
  clocks, CUDA initialisation and the desktop.
- **Evidence:** lab-only: `evidence/runs/synthetic-moe-gpu-20261023/`.

### F-007 · 2026-10-23 · Larger CUDA synthetic run: range-based ordering on a noise model

- **Class:** OBSERVED on a random model; INFERRED that every layer was on the card (the placement
  was not recorded).
- **What:** 8 layers, 64 experts, 128 tokens, 7 repetitions, parity PASS on every run. Experts on
  the CPU: `gnostral` 227-273 tok/s against `per-route` 112-127 (ranges disjoint); `grouped`
  97-131 (overlaps). Mixed budget: `grouped` 237-252 against `per-route` 220-234 (disjoint);
  `gnostral` 221-322 (overlaps).
- **Evidence:** `evidence/runs/synthetic-moe-e2e-gpu-20261023.json`.
- **Limit:** a random 8-layer model; says nothing about Qwen3-30B-A3B.

### F-008 · 2026-10-24 · The 34.54 tok/s baseline is llama.cpp with every expert on the CPU

- **Class:** SUPPORTED (Q-009 as retained in [Q-025](../quests/Q-025-ENGINE-DIRECTION.md)).
- **Consequence:** the comparable gnostral configuration is "experts on the CPU, every other layer
  on the card". A mixed CPU/GPU expert budget measures residency gain, a different question.

### F-009 · 2026-10-24 · The first real-model M1 run stopped on its own reference

- **Class:** OBSERVED.
- **What:** Qwen3-30B-A3B Q2_K, CUDA, experts on the CPU. The `per-route` warmup diverged from the
  `per-route` reference capture on 9/32 prompts (token indices 8 to 29); the parity stop ended the
  run before any timing was recorded. CUDA was proven (pid listed by `nvidia-smi`), about 7.8 tok/s.
- **Evidence:** lab-only: `evidence/runs/m1-qwen3-30b-cpu-experts-20261024/` (`PARITY_FAIL`).
- **Process note:** an earlier attempt of the same run failed (lanes were not given `--model`) and
  its output directory was removed by the relaunch command. A controlled failure is evidence
  ([Q-025](../quests/Q-025-ENGINE-DIRECTION.md)); relaunches now use a new directory.

### F-010 · 2026-10-24 · Divergence follows the number of layers placed on CUDA

- **Class:** SUPPORTED (strong correlation over five configurations, two captures each; not yet
  provoked on purpose).
- **What:** runs with the same placement are bit-identical in tokens and logprobs (A: 16/16 layers
  on CUDA; B: all on CPU; C2: 15/15). Runs whose placement changed between captures (C1: 16 then
  15; D: 15 then 16) diverge on 9/9 prompts, logprobs from token index 0 on 8 of them. The
  cuBLAS workspace setting and `RAYON_NUM_THREADS=1` are not causal.
- **Evidence:** `evidence/runs/m1-repro-diag2-20261024/repro.md` and `repro.json`.

### F-011 · 2026-10-24 · Why the placement moved, and what it did to the speed figure

- **Class:** SUPPORTED (source inspection).
- **What:** under expert residency the expert stacks load host-side (`set_device(Device::Cpu)` in
  `MoEExperts::new_resident`), but the automatic device map sized each layer with all 128 expert
  stacks (`Qwen3MoELoader::layer_sizes_in_bytes`). The mapper therefore believed only 15 or 16 of 48
  layers fit an 8 GiB card, and the threshold sat close to the free VRAM left by a desktop using
  about 2.7 GiB, so it flipped between loads.
- **Consequence:** the 7.7-7.8 tok/s real-model figures ran the attention of 32-33 layers on the
  CPU. They are not "experts on the CPU" and are not comparable to llama.cpp `--cpu-moe` (F-008).

### F-012 · 2026-10-24 · Patch 0006 fixed a path the loader does not take

- **Class:** REJECTED as the fix; PROCESS.
- **What:** `0006` changed `GgufDeviceMapLoaderInner`, which only the legacy adapter path uses.
  `GgufModelBuilder` loads through `load_native_normal` -> normal pipeline -> `Qwen3MoELoader`. The
  lab still saw `Layers 0-15: cuda[0]`. The `0006` unit test called the function directly, so it
  proved arithmetic, not the load path. Found by the lab worker, not by us.
- **Rule adopted:** a test that bypasses the real path is labelled as such, and a placement fix is
  claimed only from the engine's own placement report.

### F-013 · 2026-10-24 · Patch 0007 targets the native path

- **Class:** INFERRED until the lab reports `Layers 0-47: cuda[0]`.
- **What:** `Qwen3MoELoader` sizing drops expert stacks under residency and reserves the cache
  budget once. Unit test with the Qwen3-30B-A3B geometry.
- **Evidence:** `patches/engine-program/0007-*.patch`.

### F-014 · 2026-10-24 · The prefix cache changes the computation, deterministically

- **Class:** OBSERVED.
- **What:** with the same placement, captures with and without the prefix cache differ, while each
  setting reproduces itself.
- **Decision:** benchmark captures run with the prefix cache off, so each prompt is independent of
  the ones before it; the default-on behaviour is measured separately later.

### F-015 · 2026-10-24 · Exact top-1/top-2 ties are common on this model

- **Class:** OBSERVED (the values); UNKNOWN (why they fall on multiples of 0.25).
- **What:** at divergence points the top-1 minus top-2 logprob gap is 0, 0.25, 0.5, 0.75 or 1.0,
  several exactly 0.
- **Consequence:** any change of summation order can flip a greedy token. Token parity between
  lanes requires identical placement and numerics, not "close" ones.

## Evidence not yet in the repository

These directories exist on the lab disk only. Until they are committed they are not program
evidence, and a disk failure would erase them.

| Directory | Entry |
| --- | --- |
| `evidence/runs/synthetic-moe-gpu-20261023/` | F-006 |
| `evidence/runs/m1-qwen3-30b-cpu-experts-20261024/` | F-009 |
| `evidence/runs/m1-repro-diag3-20261024/` (0006 probe log) | F-012 |
