# Q-040 — MULTI-ENGINE-01

Review date: 2026-10-23
State: **PROPOSED** — no measurement exists; no supervisor implementation exists.
Authorization: supervisor work in `crates/gnostral-supervisor/`, the provider v1 contract in
`harness/engine-provider-contract/src/v1.rs`, and per-engine qualification profiles. This quest does
not authorize merging engines, granting one engine another's materialization authority, or claiming
engine portability from a typed trait.
Workstream: W4 · Milestone: M4 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can **mistralrs-strata in-process**, **llama.cpp CLI/server** and the **C++ Strata server** serve
through one supervisor with a per-engine qualification profile — selection receipt, streaming and
cancel — with **no capability inheritance** between engines?

The v1 contract is a draft; the provider implementation (Q-024b) sits on the existing
`Q-013-D → Q-016` lane. This quest consumes that lane, it does not bypass it.

## Deliverable and file seams

- `crates/gnostral-supervisor/` — registry, admission, lifecycle, SLO facts, selection receipt.
- `harness/engine-provider-contract/src/v1.rs` — conformance group for start/poll streaming, cancel,
  error taxonomy.
- Per-engine profile: exact binary, model, formats, measured envelopes; facts are observed, never
  inherited from another engine's profile.
- Contracts under `contracts/`; conformance tests against stubs, then against real engines.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Conformance | each engine passes the shared conformance group independently | one engine's pass reused for another |
| No capability inheritance | an engine not qualified for a format/capability is refused | capability borrowed from a sibling profile |
| Selection receipt | every start records why that engine was chosen, with its facts | silent default engine |
| Streaming and cancel | partial output streams and cancellation terminates the request | buffered-only or uncancellable path claimed as streaming |
| Crash isolation | a killed engine leaves no orphan and the supervisor reports it | supervisor hangs or leaks the process tree |
| A→B→A | load A, unload, load B, unload, reload A with reclaim receipts | cumulative drift unmeasured |
| Admission | `UNKNOWN` facts are denied, not treated as healthy | plausible defaults admitted |
| Authority | no placement or eviction command crosses the provider boundary (v1 authority test) | any authority leak |

## Non-claims

- No multi-engine performance, fairness or availability number exists.
- Three engines behind one API is not one merged engine, not a cluster, and not hot-swap.
- No claim of concurrent model residency; A→B→A is sequential unless co-residency is separately
  measured.
- No authority over experts, KV blocks, layers or kernels for the supervisor (Q-008 stands).

## Definition of done

1. Shared conformance green for each engine separately, with its own profile and identity.
2. One A→B→A receipt with reclaim and no orphan, all samples retained.
3. A selection receipt that shows the engine chosen and the facts used.
4. A negative test proving a borrowed capability is refused.
