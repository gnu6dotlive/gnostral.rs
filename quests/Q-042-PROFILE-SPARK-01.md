# Q-042 — PROFILE-SPARK-01

Review date: 2026-10-23
State: **PROPOSED** — timer crate landed and wired into the engine (`patches/engine-program/0002`, feature `gnostral-trace`, `GNOSTRAL_TRACE=1`); token-neutral on the synthetic CPU fixture; no profile of a real model exists (planned after the placement fix, see [FINDINGS](../docs/FINDINGS.md)).
Authorization: instrumentation behind an environment flag in the engine and harness, plus a
single-request trace export. This quest does not authorize always-on tracing, a profiler dependency
that changes the runtime, or a performance conclusion drawn without its own receipt.
Workstream: W5 · Milestone: M1 support · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Where does a step actually spend its time — schedule, forward, CPU expert, upload, sample,
post-process — and can that be captured as a timeline for **one request** without changing what the
engine computes?

Two later quests depend on this: Q-037 must not start before upload stalls are shown to matter, and
Q-020 (ZOS-lite) must measure the CPU gap before writing code. Neither can proceed on intuition.

## Deliverable and file seams

- Per-phase timers: schedule, forward, CPU expert, upload, sample, post-process; monotonic clock;
  `UNKNOWN` where a phase cannot be attributed.
- Environment flag, off by default; no tracing in release paths unless requested.
- Export: one request's timeline as a machine-readable receipt under `evidence/runs/`, plus a small
  rendered summary.
- The timer read path must be allocation-free in the hot loop where the engine requires it.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Coverage | every named phase reports a duration or an explicit `UNKNOWN` | a phase silently missing |
| Reconciliation | phase sum reconciled against measured wall time, any gap reported | unexplained gap presented as zero |
| Semantic neutrality | identical tokens with timers on and off | any token difference |
| Default off | flag off ⇒ no measurable overhead in the step loop | tracing always on |
| Overhead honesty | instrumentation cost measured and reported, never assumed zero | "negligible" asserted |
| Export | one request's timeline is a replayable receipt with binary and model identity | screenshot-only evidence |
| Clock | monotonic source recorded; no wall-clock arithmetic | clock source unstated |

## Non-claims

- No profile exists. This quest does not say where time goes.
- A single request's timeline is not a workload average and is not a benchmark.
- The timer cannot attribute time spent inside a closed kernel or driver.
- No claim that any phase is "the bottleneck" before its receipt exists.

## Definition of done

1. Timers compile in the engine and the crate's CPU tests cover the neutral path.
2. One real single-request export with identity pins, reconciliation and measured overhead.
3. A retained receipt that Q-037 and Q-020 can cite for their precondition decisions.
