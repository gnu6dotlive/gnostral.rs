# Q-031 — ENGINE-PERF-01

Review date: 2026-10-23
State: **PROPOSED** — no orchestrator exists, no run has been made.
Authorization: a new harness under `harness/bench/`, schemas under `schemas/`, synthetic receipts for
testing, and one real multi-lane run on the lab host. This quest does not authorize a benchmark
protocol change, a lane shortcut, or a result published without its samples.
Workstream: W5 · Milestone: M0 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can **one command** run the frozen Q-025 protocol across N lanes, normalize every receipt, render the
Pareto table and keep all samples — while refusing to compare identities that are not pinned?

The failure mode this prevents is a hand-assembled comparison table whose provenance cannot be
reconstructed by a second person.

## Deliverable and file seams

- `harness/bench/engine_perf.py` (or equivalent) — lane driver, 1 warmup + ≥ 3 measured
  repetitions, interleaved order, per-lane identity pins.
- `schemas/` — receipt schema covering every field Q-025 requires (prefill, TTFT, decode, TPOT, VRAM,
  RSS/PSS, reclaim, driver health, exit status, command line).
- Pareto renderer: quality, decode, prefill, VRAM, long context, concurrency, determinism.
- `evidence/runs/engine-perf-<date>.json` — the run bundle, raw logs under `evidence/logs/`.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Pinned identity | every lane carries binary hash, model hash, command line and protocol version | any lane missing an identity → the lane is `BLOCKED` |
| Protocol fidelity | 1 warmup, ≥ 3 measured repetitions, interleaved as frozen | fewer repetitions or reordered lanes after seeing results |
| Refusal | an un-pinned or mismatched lane aborts the comparison, not the run | "close enough" identity accepted |
| Pareto output | table generated from the receipts, not typed by hand | any number in the table absent from a receipt |
| Sample retention | every sample and every failed run kept | only medians retained |
| Recomputability | a second person can rerun the command and reproduce the table from the receipts | provenance unclear |
| Testability | orchestrator tested with synthetic receipts offline | untested driver used for a real decision |

The benchmark's numeric verdict belongs to the milestone that owns the threshold (M1/M2/M3); this
quest produces the receipt, it does not award a result.

## Non-claims

- The orchestrator improves no engine and proves no speedup.
- Running lanes does not make their results comparable if identities or protocols differ.
- No claim that the Pareto table settles the substrate question; D1 is decided on the Q-025 receipt.
- Synthetic receipts test plumbing only and are never evidence.

## Definition of done

1. One command runs the frozen protocol across the Q-025 lanes with pinned identities.
2. A real run produces a bundle with all samples, raw logs and a rendered Pareto table.
3. The refusal path is demonstrated with a deliberately un-pinned lane.
4. A second person replays the command and obtains the same table from the same receipts.
