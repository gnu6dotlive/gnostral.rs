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

### F-016 · 2026-10-24 · The driver probe reported `NO_XID` without reading anything

- **Class:** PROVEN (reproduced on a host without an NVIDIA driver and with an empty journal).
- **What:** the plan's probe (`journalctl -k -b | grep -q 'NVRM: Xid'`, else `NO_XID`) printed
  `NO_XID` when the kernel log was empty or unreadable, and the bench then accepted a comparison on a
  machine with no GPU at all. Absence of evidence was reported as a clean driver.
- **Fix:** `harness/bench/driver_probe.py` prints `NO_XID` only after reading this boot's kernel log
  (journal, else `dmesg`) with an NVIDIA driver loaded, `UNKNOWN` otherwise, plus a detail line saying
  what the verdict rests on. Six tests, including the empty-journal case.
- **Consequence for earlier evidence:** the `NO_XID` of F-006 and F-007 came from the old probe. The
  operator also checked the kernel log by hand and reported no Xid, so those runs are not known to be
  dirty, but their driver verdict is SUPPORTED by that manual check, not by the probe.

### F-017 · 2026-10-24 · The gnostral kernel lane is slower than candle on a tiny CPU-only model

- **Class:** OBSERVED (two runs); cause UNKNOWN.
- **What:** synthetic 2-layer fixture, every layer on the CPU, 4-core Xeon with AVX-512 in a cloud
  container: `gnostral` 451-477 tok/s against `per-route` 820-903 and `grouped` 865-922 (ranges
  disjoint, separation `BELOW`). On the lab's 8-layer CUDA fixture the same lane sat above `per-route`
  (F-007).
- **Hypothesis to test, not a conclusion:** fixed per-dispatch cost. The pool wakes its threads with a
  condition variable on every projection (two dispatches per MoE layer per token), and on this host
  those threads share four cores with candle's own rayon pool. With matrices this small the wake-up
  and contention cost could exceed the compute. This is the second Phase 2 hypothesis; it gets an
  instrumented measurement before any change.

### F-018 · 2026-10-24 · The synthetic fixture mistyped its byte tokens

- **Class:** PROVEN (llama.cpp refused the old file and loads the new one).
- **What:** `make_qwen3moe_gguf.py` typed the 256 GPT-2 byte-level tokens as `BYTE` (6), which GGUF
  reserves for SentencePiece `<0xXX>` tokens. mistral.rs ignored it; llama.cpp tried to parse the
  token text as `<0xXX>` and failed to load the vocabulary. They are now `NORMAL` (1).
- **Consequence:** the generator's output changed: seed 20261023 now gives
  `d1dc2d72...9abce50` instead of `581b7325...69b5c6a`. Earlier receipts stay valid for the file they
  name; reproducing them needs the generator at their commit.

### F-019 · 2026-10-24 · llama-server omits probabilities for tokens inside an incomplete UTF-8 character

- **Class:** SUPPORTED (source: `tools/server/server-context.cpp` at b11857 calls `add_token`, which
  records probabilities, only once the generated text is valid UTF-8) and OBSERVED (32 tokens, 27
  probability entries on one fixture prompt).
- **Consequence:** per-token logprobs from `/completion` are not one per token. The llama.cpp lane
  records the server's entries as reported, tagged by token id, checks they are an ordered
  subsequence of the full token list, and its self-reproducibility compares both bit for bit. It does
  not pretend to have a logprob for every token. Real text with accents or non-Latin scripts will hit
  this too.

### F-020 · 2026-10-24 · On the tiny CPU fixture llama.cpp decodes about 3x faster than our per-route lane

- **Class:** OBSERVED (one four-lane run, three measured repetitions each); cause UNKNOWN.
- **What:** same random 2-layer model, everything on a 4-core CPU: llama.cpp 2488-3134 tok/s,
  `per-route` 820-915, `grouped` 771-870, `gnostral` 438-459. Every lane reproduced itself and kept
  its pinned placement; the comparison itself was refused (driver `UNKNOWN`, F-016).
- **Why it matters:** with matrices this small, per-token fixed costs dominate. It is a hint, to be
  measured with the Q-042 timers, that the gap to llama.cpp on the real model may include fixed
  overhead per layer and per token, not only kernel throughput. It is not a statement about
  Qwen3-30B-A3B.
- **Evidence:** `evidence/runs/synthetic-moe-four-lane-cpu-20261024/summary.json`.

### F-021 · 2026-10-24 · Our expert kernel is about 3x slower than ggml's at decode shapes

- **Class:** OBSERVED on a cloud VM (4-core Xeon with AVX-512, 260 MiB L3), not on the lab host.
- **What:** Qwen3-30B-A3B expert geometry, one token, 8 experts, 2 threads on both sides, the same 8
  experts every iteration on both sides (`test-backend-ops` uses ids 0..7): our `grouped_expert_ffn`
  3.23 ms median per layer; ggml's three `MUL_MAT_ID` (gate, up, down) about 1.01-1.03 ms. Per block,
  single thread: our `Q2_K` dot 19.4 ns, `Q3_K` 30.9 ns; ggml at 2 threads works out to roughly 10 and
  19 ns per block per thread.
- **Limits:** ggml's figure excludes SwiGLU and the activation re-quantisation between up and down; this
  host's L3 holds all 128 experts, so neither side is memory-bound here as it would be on a desktop CPU.
- **Evidence:** `evidence/runs/kernel-microbench-cloud-20261024/summary.json` (raw samples, ggml output,
  the two-line `test-backend-ops` diff that adds the `Q2_K`/`Q3_K` cases).

### F-022 · 2026-10-24 · Waking the thread pool costs tens of microseconds per dispatch

- **Class:** OBSERVED (same VM).
- **What:** `Pool::run` over trivial tasks: 27 us at 2 threads, 68 us at 4. The grouped kernel dispatches
  twice per MoE layer per token, so 48 layers at 4 threads would spend about 6.5 ms per token waking
  threads, against a 28.6 ms budget per token at the M1 target of 35 tok/s.
- **Next:** the second Phase 2 hypothesis now has a number on this host; it needs the lab's timeline
  (patch `0009`) before any change, and any change keeps the bit-equality tests.

### F-023 · 2026-10-24 · The engine's bit-exact numerics path is 3x slower than the native one for Q2_K

- **Class:** OBSERVED (same VM).
- **What:** per `Q2_K` block, `Numerics::CandleCompat` 58 ns against `Native` 19 ns (AVX-512 integer part
  in both); `Q3_K` 34 vs 31 ns. The candle-compatible combination runs its eight lanes and the `mins` x
  `bsums` products in scalar code per block. The engine's `gnostral` lane uses `CandleCompat` (F-002).
- **Next:** candle itself does that combination with `_mm256_fmadd_ps` on the same eight lanes, so a
  vectorised version can stay bit-identical; it is a Phase 3 candidate, to be done with the existing
  intrinsics-transcription tests as the gate.

## Evidence not yet in the repository

These directories exist on the lab disk only. Until they are committed they are not program
evidence, and a disk failure would erase them.

| Directory | Entry |
| --- | --- |
| `evidence/runs/synthetic-moe-gpu-20261023/` | F-006 |
| `evidence/runs/m1-qwen3-30b-cpu-experts-20261024/` | F-009 |
| `evidence/runs/m1-repro-diag3-20261024/` (0006 probe log) | F-012 |
