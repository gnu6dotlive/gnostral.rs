# Q-039 — SLO-SCHED-01

Review date: 2026-10-23
State: **PROPOSED** — admission/fairness policy and deterministic simulation landed offline (`gnostral-scheduler`); not wired; no measurement exists.
Authorization: scheduler policy work in `mistralrs-core/src/paged_attention/scheduler.rs` and the
admission path, with client-side tail measurement on the lab host. This quest does not authorize
granting the supervisor scheduling authority over engine internals, a default-on adaptive policy, or
a concurrency claim before the receipt.
Workstream: W3 · Milestone: M3 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can deadline/priority-aware admission and scheduling produce a **fair** C1/C4 tail with a receipt per
request — aggregate C4 throughput at the program's concurrency target, TPOT(C4) inside the stated
bound, and no request starved — without changing tokens?

Q-016 already owns client SLO qualification for the supervisor. This quest supplies engine-side
scheduling facts to it; it does not replace it.

## Deliverable and file seams

- Admission policy: deadline and priority classes, explicit rejection when a deadline cannot be met.
- Scheduler hooks in `paged_attention/scheduler.rs`; per-request queue wait, TPOT and finish cause.
- Fairness receipt: every admitted request's waits and progress, not an aggregate alone.
- Measurement: C1 and C4 lanes on the frozen workload, client-observed TTFT/TPOT/tails.
- Feeds the Q-016 evidence set; does not write Q-016's verdict.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Policy determinism | same request set and deadlines ⇒ same admission decisions | wall-clock-dependent admission |
| No starvation | every admitted request progresses and finishes or is explicitly rejected | a request waiting behind others indefinitely |
| Fairness receipt | per-request wait/TPOT recorded | aggregate-only reporting |
| Token parity | same tokens as the C1 baseline for the same request | any quality change |
| Tail honesty | P50/P95/P99 reported with the sample count and all samples | percentile without samples |
| Concurrency target | C4 aggregate ≥ 1.5× C1 and TPOT(C4) ≤ 1.5× TPOT(C1) — the M3 gate — measured, not assumed | stating either as a result before the run |
| Default posture | adaptive policy off by default until paired-run evidence exists | enabled by default on a hunch |

## Non-claims

- No tail, throughput or fairness number exists.
- A scheduling policy that looks fair in a simulator has not been shown fair with a real engine.
- No claim of deadline guarantees under desktop pressure; that needs Q-041.
- No authority transfer: the engine schedules; the supervisor observes (Q-008).

## Definition of done

1. Policy unit tests green offline, including the rejection path.
2. A receipt with C1 and C4 samples, per-request fairness and tails, all samples retained.
3. Either the M3 target is met with a receipt, or a recorded negative with the policy off by default.
