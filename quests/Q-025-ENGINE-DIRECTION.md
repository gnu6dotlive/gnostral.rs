# Q-025 — ENGINE-DIRECTION-01

Review date: 2026-10-23
State: **PROPOSED** — protocol frozen before any run; no measurement exists yet.
Authorization: one bounded local benchmark session under `gnu6-lab-run --exclusive`.
This quest does not authorize a new engine dependency, a new runtime, a new model
download, or any change to the Q-007/Q-008 authority boundary.

## Question

Which local execution path deserves the next engine investment, measured on the lab's
own reference workload: **llama.cpp CPU-MoE**, **mistralrs-strata (current candidate)**,
or **mistralrs-strata + native expert residency**?

The project's retained evidence already makes this a decision, not a curiosity:

- Q-009, matched measured VRAM, same GGUF: llama.cpp CPU-MoE **34.54 tok/s** vs Strata **9.42 tok/s**.
- Q-010, paired candidate A/B: Strata baseline **9.51 tok/s** vs borrowed-CPU-expert candidate
  **20.18 tok/s** (2.12×), still below the llama.cpp figure on the recorded profile.
- The Q-010 source is one `compute_host` function plus a Candle CPU expert matmul, not the
  residency cache; the residency `forward()` remains a serial per-route prototype.

Continuing to absorb Strata without confronting that comparison is the largest strategic
risk in the repository. This quest freezes the comparison before more engine work.

## Frozen protocol

Freeze these values before the first run and keep every sample.

| Field | Frozen value |
| --- | --- |
| Model | Qwen3-30B-A3B Q2_K GGUF, local read-only path; record full-file SHA-256 |
| Host | RTX 3070 8 GiB, 32 GiB host RAM, graphical session protected, `gnu6-lab-run --exclusive` |
| Context | 1024 tokens |
| Prompt | the retained six-check prompt set, exact bytes hashed |
| Sampling | temperature 0, fixed seed, no penalty, `n_predict` fixed (record the value) |
| CPU threads | 8 (record actual) |
| Warmup | 1 discarded run per lane |
| Repetitions | 3 measured per lane, interleaved lane order recorded in advance |
| VRAM | whole-card used MiB sampled during the run (not per-process estimates) |
| Host memory | process-tree RSS and PSS, peak |
| Reclaim | after stop: whole-card ambient within ±128 MiB of pre-run ambient; driver health `NO_XID` |
| Semantic parity | exact expected response (or a hash-bound expected token stream) checked before timing is counted |

Lanes (exact binaries hashed; no rebuild between lanes):

1. `llama.cpp` CPU-MoE (`--cpu-moe` / documented equivalent) on the frozen GGUF.
2. `mistralrs-strata` lab revision, CPU/GPU expert split as in Q-009/Q-010.
3. `mistralrs-strata` + `gnostral-expert-residency` candidate (the Q-010 fast path).

If any lane's exact binary or model identity cannot be pinned, that lane is `BLOCKED`, not
approximated.

## Measurements

For every sample, retain: prefill tok/s, client TTFT, decode tok/s, TPOT, peak whole-card VRAM,
process-tree RSS/PSS peaks, expert-cache hits/misses or `UNKNOWN`, bytes copied, reclaim delta,
driver health, exit status, and the exact command line.

Report medians and all individual samples. Do not compare a warm file cache against a cold one
without saying so; record file-cache state and engine warmness.

## Decision rules (pre-declared)

1. If the three lanes do not all produce semantically valid output on the frozen probe, the run is
   `NOT_QUALIFIED`; no lane is chosen.
2. Choose the lane with the highest median decode tok/s **only if** its prefill/TTFT is not worse by
   more than 10 % and its peak VRAM stays under the same envelope.
3. If llama.cpp wins, the engine lane's next investment is to study its CPU-MoE batching/repack
   design, not to continue Strata absorption by default.
4. If the Rust candidate wins within the envelope, continue the residency path and batch `forward()`
   before any performance claim.
5. If the lanes are within noise (<2 %), record `NO_DECISION` and do not re-open the question
   without a new mechanism.

## Required claims and stop conditions

- Claims are valid only for this host, model, quantisation, binary hashes and protocol.
- Do not claim portability, AMD/Intel support, a production server, or general MoE superiority.
- Preserve every failed run and every lost sample; a controlled failure is evidence.
- Stop the session on an unexpected Xid, driver reset, uncommanded process, or desktop pressure.
- Do not rebuild lanes between repetitions; if a binary must change, the protocol restarts.
- No GitHub Actions run may be used to learn whether this works. The session is local and bounded.

## Deliverables

- `evidence/runs/engine-direction-<date>.json` — machine-readable receipts, all samples.
- `research/runtime/ENGINE-DIRECTION-<date>.md` — report with the decision and non-claims.
- Only after that: update `docs/ROADMAP.md`, `docs/ENGINE_LEDGER.md` and a quest row in
  `QUESTLOG.md`; no roadmap change before the receipt exists.

## Authority boundary

The benchmark compares engines; it does not merge them. llama.cpp and Strata remain separate
engines behind the supervisor. Expert/KV placement, batching and kernels stay engine-owned
(`contracts/ENGINE_PROVIDER_V0.md`). A winner does not inherit authority over another engine's
materialization.
