# Q-027 — RESIDENCY-DYNAMIC-01

Review date: 2026-10-23
State: **PROPOSED** — dynamic admission/eviction policy and budget arithmetic landed offline (`gnostral-expert-residency::dynamic`, `gnostral-memory-plan`); engine patch `0003` records observed expert demand as a loadable profile (synthetic fixture only); engine patches `0006`/`0007` make the automatic device map stop counting host-loaded expert stacks ([F-011 to F-013](../docs/FINDINGS.md), lab verification pending); dynamic admission not wired into the engine; no measurement exists.
Authorization: policy work in `crates/gnostral-expert-residency/`, a future
`crates/gnostral-memory-plan/`, and residency integration in
`mistralrs-core/src/moe/experts/residency.rs`. This quest does not authorize KV or expert placement
authority for the supervisor, a new engine dependency, or a claim about a static profile that was
never compared.
Workstream: W2 · Milestone: M2 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can expert residency be driven by an **online** access histogram with leases, pin counts and a
segmented allocator, so that a hot set stays resident, misses are admitted explicitly, memory can be
returned, and the expert cache and KV cache share one honest budget — without making decode latency
worse than the static profile it replaces?

The current residency crate carries the static plan, lease and route bookkeeping from the
Q-009/Q-010 path. It has no histogram, no admission decision, no eviction policy and no budget
arbitration.

## Deliverable and file seams

- `crates/gnostral-expert-residency/` — histogram, admission/eviction policy, leases, pin counts,
  hit/miss accounting, `UNKNOWN` propagation.
- `crates/gnostral-memory-plan/` — segmented budget arithmetic shared by experts, KV and workspace.
- Integration: `mistralrs-core/src/moe/experts/residency.rs`; the cache must never evict a slot that
  is leased for an in-flight route.
- Telemetry: hits, misses, evictions, fragmentation and `UNKNOWN` are reported as observed facts, not
  inferred estimates.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Determinism | same route history + same admissions ⇒ same resident set | policy depends on wall-clock timing or hash iteration |
| Lease safety | no eviction while a slot is leased or pinned; upload ticket and lease cannot be confused | any slot released while referenced |
| Reference steady state | the frozen reference workload's working set is not evicted once warm | eviction of the reference set under steady state |
| Admission honesty | a miss is admitted explicitly or refused explicitly; never treated as a hit | silent fallback to a stale slot |
| Budget arbitration | experts + KV + workspace + desktop reserve stay inside one envelope; the split is recorded per step | any component exceeding its share without an explicit decision |
| Elastic return | a released segment is returned and the fragmentation delta is reported | "returned" claimed without a reclaim observation |
| Negative-result gate | if dynamic residency is slower than the static profile beyond the envelope, keep static and publish it | enabling a slower policy by default |

Reclaim within ±128 MiB and 32K context are **M2** gates measured after Q-028/Q-038; this quest does
not claim them.

## Non-claims

- No hit-rate, latency, VRAM or throughput number exists. No cache improves anything until measured.
- A correct lease table proves bookkeeping, not that the policy helps.
- No claim that eviction hysteresis is sufficient under desktop pressure; that needs Q-041.
- The supervisor gains no eviction or placement authority from this quest (Q-008 stands).

## Definition of done

1. Policy unit tests green offline: determinism, lease safety, refusal paths, budget arithmetic.
2. A receipt with hit/miss/eviction counts, fragmentation and a static-vs-dynamic paired comparison
   on the lab host, all samples retained.
3. Either a measured improvement inside the envelope, or a recorded negative with static kept as the
   default.
