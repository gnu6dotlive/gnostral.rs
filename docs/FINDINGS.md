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
- **Evidence:** `evidence/runs/synthetic-moe-gpu-20261023/`, committed in `9495db5`.

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
- **Evidence:** `evidence/runs/m1-qwen3-30b-cpu-experts-20261024/` (`PARITY_FAIL`), committed in `9495db5`.
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
- Superseded by F-031.

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

### F-024 · 2026-10-24 · The bit-exact path is now faster than the native one, and still bit-exact

- **Class:** OBSERVED (same VM, same session, before and after binaries interleaved); bit equality PROVEN
  by tests against candle itself and end to end.
- **What:** Phase 3a moved the `CandleCompat` combination (F-023) into one `target_feature` function per
  row, with candle's FMA order and `hsum_float_8`. Per block, AVX-512: `Q2_K` 56.9 -> 9.9 ns, `Q3_K` 33.6
  -> 19.3 ns; AVX2: 59.7 -> 10.7 and 37.8 -> 21.7 ns. Expert layer of Qwen3-30B-A3B geometry (8 experts
  rotating from 128, one token): 2 threads 5.53 -> 2.24 ms, 4 threads 3.15 -> 1.50 ms, ranges disjoint.
  Against ggml's 1.01-1.03 ms at 2 threads (F-021), the engine's lane goes from about 5.4x to 2.2x behind.
- **Bit equality:** the crate's transcription tests pass, and both deliberate mutations (FMAs swapped,
  sequential horizontal sum) make them fail; the engine test `gnostral_cpu_kernel_is_bit_equal_to_candle`
  passes; on the synthetic fixture every lane's 1024 tokens and 2048 logprob entries are bit-identical to
  the committed four-lane evidence (only the binary hash differs).
- **Not shown:** an end-to-end speed change. The earlier fixture run is from another session and the
  fixture is too small for the kernel to dominate (F-017). The lab's M1 run is where it can show.
- **Evidence:** `evidence/runs/kernel-candlecompat-vectorised-20261024/`.

### F-025 · 2026-10-24 · The native path pays per-block overhead the fast path does not

- **Class:** OBSERVED (same VM) for the numbers; INFERRED for the cause, from the code.
- **What:** after F-024 the `Native` numerics (18.4 ns per `Q2_K` block) is about 2x slower than
  `CandleCompat` (9.9 ns) although it does no more arithmetic. Native calls a lane function per block
  through a `target_feature` boundary that cannot inline, stores the eight lanes to memory, sums them in
  scalar code and computes `summs` over 16 scalar products. A whole-row native loop would keep integer
  sums exact, so it can be bit-identical to today's native output. Native is not on the engine's path
  (the engine uses `CandleCompat`), so this is recorded, not scheduled.
- **Process (PROCESS):** the same unchanged native AVX2 dot measured 37.2 ns in one session (F-021
  evidence) and 20.6 ns in this one, on VMs reporting the same CPU model. Microbenchmark numbers from
  different sessions are not comparable; only interleaved before/after in one session is evidence of a
  change.

### F-026 · 2026-10-24 · The owner's new target model may not load in this engine tree

- **Class:** OBSERVED from source for the engine; UNKNOWN for the model until its file is inventoried.
- **What:** the owner's survey of 2026-10-24 (8 GB VRAM, 32 GB RAM; a document outside this repository,
  its figures not measured here) names Qwen3.6-35B-A3B in Unsloth `UD-IQ4_XS` (16.5 GiB by its count) as
  the LLM to use, replacing Qwen3-30B-A3B. In this tree:
  - `mistralrs-quant/src/gguf/archive.rs` `candle_dtype` maps only F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
    Q8_0, Q8_1 and Q2_K to Q8_K; every other type, `IQ4_XS` (23) included, fails the load;
  - expert residency accepts Q2_K to Q6_K, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0; the Q-029 kernels (and the
    gain in F-024) cover Q2_K and Q3_K only, other types run candle's kernels;
  - patch `0007`'s device-map sizing covers the `qwen3moe` loader; the engine has separate `qwen3next`
    and `qwen3_5_moe` (hybrid linear and full attention) paths that the program has not touched.
- **Not known:** Qwen3.6's GGUF architecture string and which types an `UD-IQ4_XS` file actually holds
  per tensor. `harness/model-inventory/gguf_inventory.py` answers both from the header in seconds;
  it has to run on the lab's copy of the file before any engineering is chosen for this model.

### F-027 · 2026-10-24 · The program's engine work exists for one architecture: `qwen3moe`

- **Class:** OBSERVED from source; the porting estimate is INFERRED.
- **What:** expert residency, and with it the Q-029 kernels and patch `0007`'s sizing, runs only for
  `Qwen3MoeForCausalLM`: `mistralrs-core/src/pipeline/normal.rs` refuses a residency request for any other
  architecture ("native expert residency pilot only supports unadapted single-file Qwen3 MoE GGUF without
  ISQ"), and `models/qwen3_moe.rs` is the only model that builds `MoEExperts::new_resident`. The engine's
  GGUF registry (`gguf/normal_registry.rs`) does know `qwen35moe`, `qwen3next`, `lfm2moe` and `gpt-oss`; a
  `qwen35moe` file loads through the Qwen3Next adapter. So a Qwen3.5/3.6 MoE in a loadable quantisation
  would run without any of the program's work, as a plain mistral.rs model. The message says
  "single-file", but no shard check was found in that condition: UNKNOWN whether split files are refused.
- **Porting estimate (INFERRED):** `models/qwen3_next.rs` uses the same `MoEExperts` block plus a dense
  shared expert, so residency would need the `new_resident` switch, the architecture gate, and a sizing fix
  like `0007`. Verifying it needs a synthetic `qwen35moe` fixture with well-formed linear-attention tensors.
- **Process (PROCESS):** the first `gguf_inventory.py` (`39334c9`) ignored the architecture, and would have
  reported gpt-oss's MXFP4 experts as refused (they load through a dedicated binding,
  `gguf/normal_bindings.rs`) and a `qwen35moe` file with Q4_K experts as running on residency. Both are
  fixed, each with a test; the architecture gate's test was checked to fail when `qwen35moe` is wrongly
  allowed. Found while checking the owner's updated report against the source, before anyone used the
  tool's verdict.

### F-028 · 2026-10-24 · On a qwen35moe fixture the engine matches the model definition; llama.cpp does not on k-quant experts

- **Class:** OBSERVED against a float64 reference; two independent implementations agree with it (the
  engine through candle, and gguf-py's dequantisers inside the reference).
- **Setup:** `make_qwen35moe_gguf.py` fixture (3 Gated DeltaNet layers, 1 gated full attention layer,
  routed plus shared experts), loaded by the engine's plain qwen35moe path and by llama.cpp b11857.
  `qwen35moe_reference.py` is written from transformers 5.20.2 `modeling_qwen3_5_moe.py` and the
  transformations of llama.cpp's own converter; each engine is judged by teacher forcing on its own
  tokens (reference argmax at every step, gap to its top-2 logprobs).
- **What:** with F32 experts both engines match the reference (argmax differs 0/1024 for each; gap at
  most 0.001 nats for the engine, 0.006 for llama.cpp) and produce identical tokens, 1024/1024. With
  Q2_K/Q3_K experts the engine still matches (0/1024, max gap 0.003), llama.cpp does not: the
  reference argmax differs at 82/1024 steps, median gap 0.11 nats, maximum 23. Disabling llama.cpp's
  weight repacking changes nothing (82/1024); Q2_K alone (gate/up) and Q3_K alone (down) each show it
  (median gap 0.043 and 0.010 nats against the engine's 5e-5 and 7e-5). Both engines are consistent
  with themselves: decode-path tokens equal one-prefill tokens at 192/192 steps each.
- **Not known:** the cause inside llama.cpp. The expert blocks are random, possibly unlike real
  weights, and the host has AVX-512 and AMX; nothing here says llama.cpp is wrong on real models.
- **Consequence:** a second engine is a witness, not a referee. Agreement with llama.cpp is not evidence
  of correctness on k-quant fixtures (the llama.cpp lane is already parity class `self`). The reference
  becomes the qwen35moe oracle for the residency port: residency output, plain output and reference.
- **Evidence:** `evidence/runs/qwen35moe-fixture-oracle-20261024/`.

### F-029 · 2026-10-24 · I first read the cross-engine divergence as a possible engine fault

- **Class:** PROCESS.
- **What:** the engine and llama.cpp agreed on every first token and diverged at step 2 on 15 of 32
  prompts, always the same two tokens swapped. I treated llama.cpp as the reference and looked for a
  fault in the engine's recurrent-state handoff. Each engine's self-consistency check passed, ablations
  on random weights were too noisy to decide, and only the float64 reference showed the deviation was
  on llama.cpp's side, in its k-quant experts, not in the linear attention.
- **Rule taken:** build the referee before attributing a difference between two implementations.

### F-030 · 2026-10-24 · The engine's plain qwen35moe CPU path is far slower than llama.cpp on these fixtures

- **Class:** OBSERVED (single runs per fixture); cause UNKNOWN.
- **What:** decode 20 tok/s with Q2_K/Q3_K experts and 92 tok/s with F32 experts, against llama.cpp's
  1275 and 762 tok/s on the same files and host. The engine's runs show system time close to user time
  (57 s of system time in a 126 s run).
- **Hypothesis to test, not a conclusion:** per-step allocation or dequantisation in the qwen35moe expert
  or Gated DeltaNet path. It matters for the port: residency on top of this path should wait for its
  profile.
- **Cause located in F-036:** the x86 MoE fallback dequantises every expert at every step.

### F-031 · 2026-10-24 · On the lab, 0007 places all 48 layers on the card, and the placement holds under pressure

- **Class:** OBSERVED on the lab host (RTX 3070 8 GB); supersedes F-013's INFERRED.
- **What:** with `0001` to `0010` applied, the native GGUF path logs `Layers 0-47: cuda[0]` and reports
  `0-47:cuda:0;nm=cuda:0`; before `0007` the same probe logged `Layers 0-15: cuda[0]`, `Layers 16-47: cpu`.
  (The lab also reported the process VRAM peak rising from 782 to 1102 MiB; no receipt for it is in
  the repository, so it is not evidence here.) Reproducibility matrix A, two captures of 9
  prompts x 32 tokens: tokens differ 0/9, logprobs differ 0/9; with 1.5 GB of VRAM held by another
  CUDA process the placement stays 48/48 and the outputs are bit-identical to the run without it.
- **Consequence:** F-009 to F-011's placement nondeterminism is gone for this configuration; M1 can be
  measured.
- **Evidence:** `evidence/runs/m1-repro-diag3-20261024/` (`probe.stdout.txt` before, `probe-0007.*` after,
  `A-plain/`, `A-ballast/`).

### F-032 · 2026-10-24 · First accepted M1 comparison: our best lane at 58% of llama.cpp

- **Class:** OBSERVED on the lab host, comparison accepted (`COMPARED`, driver `NO_XID`).
- **What:** Qwen3-30B-A3B Q2_K, every expert on the CPU, 48 layers on the card, 32 prompts x 32 tokens,
  1 warmup and 5 measured runs per lane, prefix cache off, placement pinned, each lane reproducing its
  own warmup. Decode tok/s, median (min-max): `per-route` 10.2 (9.8-10.5), `grouped` 10.7 (10.2-10.9,
  overlaps per-route), `gnostral` 23.6 (20.4-24.3, above), llama.cpp `--cpu-moe` 40.7 (40.0-45.1).
  Whole-card VRAM medians 3744, 3729, 3670 and 3945 MiB. This is the first real-model measurement
  with the Phase 3a kernels (F-024).
- **Limits:** n = 5; the host was not isolated (a llama-server container had been running for a day and
  the desktop uses the card); a refused first run (v2) read 8.7 and 46.2 tok/s for per-route and
  llama.cpp, so runs differ by 10-20% on this host; ambient VRAM is `UNKNOWN` in the table. Q-025's
  34.54 tok/s for llama.cpp came from another protocol and is not comparable with 40.7.
- **Process:** the v2 run was refused because llama.cpp only logs `offloaded A/B layers to GPU` at
  `-lv 4` in that build; the lane read `offloaded=none` against a pinned `49/49`. The fix was the log
  level, added to the lane's arguments (part of its identity), not a relaxed check.
- **Evidence:** `evidence/runs/m1-qwen3-30b-cpu-experts-v3-20261024/bench-cpu-experts/` (`bundle.json`,
  `pareto.md`, receipts).

### F-033 · 2026-10-24 · Where a gnostral decode step goes: CPU experts, then host-side work the GPU waits for

- **Class:** OBSERVED (one traced request of 32 tokens, lab host); the bandwidth reading is INFERRED
  until measured.
- **What:** medians over the 31 decode steps of `trace-gnostral.json`: step 42.3 ms (23.7 tok/s, in
  line with the bench's 23.6) = forward 40.1 + sampling 2.1 + gaps 0.03. Inside the forward: CPU
  experts 21.3, host transfers 2.9, MoE bookkeeping 0.25, GPU wait 0.18, everything else 15.9. (The
  lab's first summary used means, 44.9 ms per step; same picture.)
- **Reading:** GPU wait near zero means the card finishes its queued work before the host reaches each
  sync: the 15.9 ms is host-side work (op dispatch, launches, synchronisation) the GPU waits for.
  The lab measures it with nsys next.
- **Expert phase:** each token reads 8 experts x 48 layers x 1.71 MB = 655 MB of weights; in 21.3 ms that
  is 30.8 GB/s. If the lab's RAM ceiling is near that, the phase is bound by memory and a faster kernel
  barely helps; reading fewer bytes (experts cached on the card) does. `kquant_bench --mode
  bandwidth` measures the ceiling; on the cloud VM it reads 34 GB/s at 4 threads while the grouped FFN
  streams its weights at 12.5 GB/s, so there the kernel, not memory, is the limit.
- **Protocol note:** sampling with logprobs (the captures need them) costs 2.1 ms per step.
- **Evidence:** `evidence/runs/m1-profile-20261024/`.

### F-034 · 2026-10-24 · A 2 GiB expert cache lifts gnostral to 28.9 tok/s, at 1.8 GB more VRAM than llama.cpp

- **Class:** OBSERVED on the lab host, comparison accepted (`COMPARED`).
- **What:** same protocol as F-032, mixed configuration: 2 GiB of expert cache (about 20% of the
  experts) filled from a profile recorded on 6 other prompts at 64 tokens (out of sample: no prompt in
  common with the 32 measured). Decode tok/s, median (min-max): `per-route` 16.3 (13.4-16.7),
  `grouped` 18.7 (17.6-18.9, above), `gnostral` 28.9 (28.6-30.3, above), llama.cpp `--cpu-moe` 44.7
  (42.7-47.0, every expert on the CPU). Whole-card VRAM medians: our lanes 5756-5943 MiB, llama.cpp
  3943 MiB.
- **Reading:** against the all-CPU run of F-032, gnostral goes from 23.6 to 28.9 (+22%) and its ranges
  do not overlap, but the two runs were taken at different times on a host that varies 10-20% between
  runs; llama.cpp read 40.7 there and 44.7 here with the same arguments. On this protocol our best lane
  is at 65% of llama.cpp while using 46% more card memory.
- **Evidence:** `evidence/runs/m1-qwen3-30b-mixed-20261024/bench-mixed-experts/pareto.md` and receipts,
  `profile-recorded.json`.

### F-035 · 2026-10-24 · The decode is launch-bound on the host; 18 of every 42 launches per layer are KV copies

- **Class:** the A/B is OBSERVED (bit identity verified from the committed captures); the kernel counts
  come from an nsys report kept on the lab disk, so they are UNKNOWN as evidence until its `nsys stats`
  export is committed; the attribution of the copies is OBSERVED in the source.
- **What (lab report):** per decode step, 2040 kernel launches (42 per layer), 1225 asynchronous
  allocations, 145 device-to-host copies, 867 `copy2d_bf16` (18 per layer); the card busy about 23% of
  the time under nsys; attention runs the naive path.
- **A/B:** forcing off `maybe_synchronize` (a full device sync before each naive attention when less
  than 4 GB is free, which is always the case on the 3070) changed nothing measurable (lab: 23.9-24.9
  against 24.8-25.9 tok/s, overlapping; those numbers have no committed receipt) and the 6 captures
  have byte-identical logprob files. The per-layer device-to-host copies already synchronise.
- **Attribution (source):** the lab build has no `flash-attn` feature, so CUDA attention takes
  `run_attention_noflash`, which calls `repeat_kv(k, 8)` and `repeat_kv(v, 8)` (Qwen3-30B-A3B: 32 query
  heads, 4 KV heads). `repeat_kv` is `Tensor::cat` of 8 copies: 16 `copy2d` per layer, plus 2 for the KV
  append, 18, the measured count. The copies also grow with the context: at 4096 tokens they would move
  about 64 MB per layer per token. With `flash-attn`, groups up to 8 are handled natively
  (`FLASH_ATTN_NATIVE_MAX_GQA_GROUP = 8`), so no copy.
- **Next:** a build with `flash-attn` is the cheapest test of the attention share of the 15.9 ms
  host-side time (F-033); a code change that computes grouped attention without materialising the
  repeat would serve GPUs flash-attn does not support.
- **Evidence:** `evidence/runs/m1-sdpa-sync-ab-20261024/`, `evidence/runs/m1-nsys-20261024/` (capture
  only), `vendor/mistralrs-strata/mistralrs-core/src/attention/mod.rs`.
- **Receipts committed by the lab (`9495db5`), class now OBSERVED:** `nsys stats` CSVs for the one-prompt,
  32-token capture, 33 forward passes (warmup, prefill, 31 decode steps; the output-head kernel runs 33
  times). Per pass: 2,055 kernels, 867 `copy2d_bf16` (18.06 per layer), 1,273 `cuMemAllocAsync`, 152
  device-to-host copies. The lab's 2040, 1225 and 145 were counted another way; the shape of the
  finding does not change. The sync A/B's `GNOSTRAL_METRICS` lines are in `metrics.txt`.

### F-036 · 2026-10-24 · On x86 the plain MoE path dequantises every expert, every step

- **Class:** the cause is SUPPORTED (source, sampled stacks, scaling with total experts, an allocator
  A/B); the numbers are OBSERVED on the cloud VM with the qwen35moe fixtures, 2 to 4 interleaved runs
  each, captures byte-identical across runs; what it means for real models is INFERRED.
- **Source:** `qtensor_indexed_moe_forward` (`mistralrs-quant/src/gguf/cpu.rs`) first asks candle's
  `QTensor::indexed_gemv`. In candle `35d7ae7` that function is implemented for `aarch64` only and
  returns `None` elsewhere; our build shim does not touch it. The fallback dequantises the whole
  `[n_experts, n, k]` tensor to F32, then `gather_forward` copies one expert matrix per (token, top-k)
  pair with `index_select` before the matmul. Every GGUF MoE model that reaches the shared experts layer
  on an x86 CPU without residency takes this path, not only qwen35moe.
- **Stacks:** 45 gdb samples during the 16-expert k-quant run: of 43 running stacks, 15 are `memcpy`
  under `index_select` and 13 are `to_float` under `dequantize`, both inside `cpu_indexed_moe_forward`.
- **Scaling (same top-4, same 4 prompts x 32 tokens):**

  | Experts, type | Decode tok/s | TTFT ms | System s | Minor faults |
  | --- | --- | --- | --- | --- |
  | 16, Q2_K/Q3_K | 25.6-26.2 | 1778-1846 | 5.0-5.1 | 2.57 M |
  | 64, Q2_K/Q3_K | 3.9-4.2 | 2050-2353 | 20.9-22.8 | 15.25 M |
  | 16, F32 | 108-116 | 1866-1990 | 5.6-5.9 | 2.59 M |

  Four times the experts with the same four active cost 6.4 times the decode time. F32 experts, whose
  dequantisation is a copy, decode four times faster than k-quant ones.
- **Allocator A/B (diagnostic):** with glibc told never to use `mmap` and never to trim, system time
  falls from 21 s to about 1 s, faults from 15.25 M to under 0.15 M, the 64-expert decode goes from
  4.1 to 7.8 tok/s and TTFT halves at both sizes (1.8 to 0.9 s at 16 experts). The 16-expert decode
  does not move (25.7 to 24.9 tok/s). Captures are byte-identical with and without. A 64-expert
  dequantised tensor is exactly 32 MiB, glibc's ceiling for serving a request from the heap, so each
  step maps and unmaps it again: one prompt of 8 tokens makes 224 `mmap` and 152 `munmap` calls at 64
  experts against 106 and 45 at 16. 128 forward passes x 12 tensors x 8192 pages is 12.6 M faults,
  the measured difference.
- **Decomposition:** the two independent estimates of the dequantisation cost agree at about 1.17 ns
  per weight on this VM (k-quant against F32 at 16 experts; 16 against 64 experts with the allocator
  tuned). At 16 experts decode is mostly dequantisation; prefill is mostly page faults from the
  per-token copies.
- **What it means (INFERRED):** at the Qwen3-30B-A3B geometry (128 experts, hidden 2048, expert width
  768, 48 layers) this path would dequantise 29 G weights per decoded token, tens of seconds per token
  at this VM's rate, before any fault cost. The plain engine path is not a usable CPU-experts baseline
  for any real MoE on x86. The M1 lanes are not affected: residency routes qwen3moe experts through
  its own executors.
- **UNKNOWN:** whether the lab's private candle-strata (`f0f334aa`) implements `indexed_gemv` on x86.
  The lab can read it in one command.
- **Consequence:** tuning the allocator is not the fix. Either the fallback stops dequantising experts
  no token selected (dequantise the selected ones only and group tokens per expert, which can stay
  bit-identical to today's output), or qwen35moe experts go through residency, which bypasses the
  fallback. F-030's comparison against llama.cpp measured this fallback, not the model.
- **Evidence:** `evidence/runs/qwen35moe-cpu-moe-fallback-20261024/summary.json` and receipts.

### F-037 · 2026-10-25 · Two copies of one binary were "ranked" with two runs each

- **Class:** PROCESS; the probability is PROVEN (counting), the anecdote OBSERVED without a kept receipt.
- **What:** smoke-testing `harness/diag/binary_ab.py` on the cloud VM with the same capture binary under
  two labels, two measured runs each, the range rule answered "b slower" (594-643 against 669-720
  tok/s). The directory was overwritten by the next smoke run, so that one has no receipt. With five
  runs each the same test answered "inconclusive" (receipt below).
- **Why it matters:** if both sides were identical, n runs each fall in disjoint ranges with probability
  2 / C(2n, n): 33% at n = 2, 10% at n = 3, 0.8% at n = 5. The range rule protects against overlapping
  ranges, not against small n. The four-lane bench already measures 5 runs per lane.
- **Consequence:** `binary_ab.py` defaults to 5 rounds, refuses fewer than 3, and prints that probability
  next to every result; the lab contract states the same rule.
- **Evidence:** `evidence/runs/binary-ab-identical-builds-cpu-20261025/ab-summary.json`.

## Evidence not yet in the repository

None as of `9495db5`. The three directories listed here until then (`synthetic-moe-gpu-20261023/` for
F-006, `m1-qwen3-30b-cpu-experts-20261024/` for F-009, `m1-qwen3-30b-cpu-experts-v2-20261024/`, the
refused run of F-032) are committed, scrubbed of home paths. A run that stays on the lab disk is listed
here until it is committed.
