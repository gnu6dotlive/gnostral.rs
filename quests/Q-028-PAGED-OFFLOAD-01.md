# Q-028 — PAGED-OFFLOAD-01

Review date: 2026-10-23
State: **PROPOSED** — no measurement exists; no code exists.
Authorization: one GPU session on the lab host per iteration, patch series in
`mistralrs-core/src/paged_attention/` and the device-map logic in `pipeline/normal.rs`. This quest
does not authorize a CPU PagedAttention implementation, a scheduler rewrite, or enabling batching by
default before its receipt exists.
Workstream: W2 · Milestone: M2 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

PagedAttention is disabled whenever the device map contains a CPU layer
(`pipeline/normal.rs:979-983`), which is why the MoE offload path runs batch size 1. Can paged KV +
continuous batching be enabled for the configuration that actually ships — **all GPU layers, only
experts host-resident** — with identical tokens, and what do C1/C2/C4 cost?

## Deliverable and file seams

- `mistralrs-core/src/paged_attention/scheduler.rs`, block pool and KV cache manager.
- Device-map / pipeline gating: enable paged attention only when the condition is true, and refuse
  loudly when it is not (no silent degradation).
- If the target configuration still needs CPU dense layers: implement the **hybrid KV policy** (KV
  stays on GPU) rather than full CPU PagedAttention. Full CPU paged KV is out of scope unless its own
  evidence justifies it.
- Receipts: C1, C2, C4 with per-request TTFT/TPOT and fairness, plus whole-card VRAM.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Configuration truth | paged attention runs only for the all-GPU-layer + host-expert map; every other map refuses explicitly | enabled for a map it does not support |
| Token parity | batched output identical to the C1 baseline on the frozen prompt set | any divergence |
| Concurrency exercised | ≥ 2 concurrent requests complete; each request's tokens are individually valid | batching that silently serialises or drops a request |
| No starvation | every admitted request makes progress; per-request wait is recorded | one request starved behind another |
| Memory | whole-card peak inside the M1 envelope; no OOM | OOM or desktop pressure |
| Fallback | if CPU dense layers are required, hybrid KV (GPU KV) is measured instead and the result is labelled as the fallback | claiming CPU paged KV without evidence |
| M2 numbers | C4 aggregate ≥ 1.5× C1 is the [M2](../docs/ENGINE_PROGRAM.md#milestone-ladder) gate, measured through Q-031 | this quest restating it as a result |

## Non-claims

- No concurrency speedup, TTFT or VRAM number exists yet.
- Enabling a scheduler feature is not evidence that batching helps this offload configuration.
- No claim about full CPU PagedAttention, multi-model batching, or prefix-cache hit rates.
- No claim that the current device-map gate is wrong; it may be correct for other configurations.

## Definition of done

1. The gating condition is explicit and tested, including refusal paths.
2. A receipt with C1/C2/C4 samples, per-request fairness and peak VRAM, all samples retained.
3. Either a measured gain under the M2 envelope, or the hybrid KV fallback measured, or a recorded
   negative with batching left off by default.
