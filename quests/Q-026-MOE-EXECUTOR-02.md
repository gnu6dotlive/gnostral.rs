# Q-026 — MOE-EXECUTOR-02

Review date: 2026-10-23
State: **PROPOSED** — policy half landed (incl. ordered scatter, bit-equal to the per-route loop); engine integration not landed; no performance measurement exists.
Authorization: CPU-only work in `crates/gnostral-moe-exec/`, and only after its gates pass, one
sequenced patch series inside the pinned `mistralrs-strata` fork. This quest does not authorize a
new engine dependency, a new CUDA kernel family, a change to the Q-007/Q-008 authority boundary, or
a release tag.
Workstream: W1 · Milestone: M1 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can the per-route expert loop be replaced by **one deterministic dispatch plan per layer** — routes
partitioned by placement (GPU-resident / host / uploading), each partition executed as one grouped
call, with a single weighted scatter-add — such that the grouped path is numerically identical to
the naive per-route reference, and the executor is ready for the M1 measurement?

The retained evidence says this is where the gap lives: the Q-010 2.12× came from removing one copy
per route, and the residency `forward()` is still a serial loop with one synchronization per forward.

## Deliverable and file seams

- `crates/gnostral-moe-exec/` — plan builder, group splitting, conservation proof, reference
  executor, `Upload` refusal. This is the policy half and stays testable without CUDA.
- Grouped row layout `[routes_per_group, hidden]` per expert, so a kernel sees one batch, not one row.
- Integration seams: `mistralrs-core/src/moe/experts/{residency.rs,backends.rs,forward.rs,mod.rs}`,
  `mistralrs-core/src/models/qwen3_moe.rs`.
- Reuse the existing grouped-MoE dispatcher in `backends.rs` for the GPU path; do not write a new
  kernel family.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Plan determinism | same routes + placements ⇒ same group order and positions, under any input permutation | order depends on arrival or hash-map iteration |
| Conservation | every route index appears exactly once, `validate_conservation` green | any lost or duplicated route |
| Placement isolation | GPU / host / upload never share a group; `Upload` groups refuse to execute | silent coercion to CPU or GPU |
| CPU numerical parity | grouped output **bit-equal** to the naive per-route oracle on randomized small experts | any drift; no tolerance is allowed here |
| Upload gating | `UploadGroupNotReady` until the ticket completes | execution on stale weights |
| Engine integration | tokens identical to the baseline on the frozen prompt set; quality oracle (Q-030) green | unexplained token divergence |
| M1 numbers | owned by [M1](../docs/ENGINE_PROGRAM.md#milestone-ladder) and measured through Q-031 only | this quest claiming any tok/s |

## Current state of this quest

- **Landed:** policy half in `crates/gnostral-moe-exec/` — deterministic grouping, conservation,
  placement isolation, reference executor, plus `gather_group` / `scatter_ordered` (weighted scatter in
  `(token, ordinal)` order, bit-equal to the per-route loop for any permutation and group cap); 13
  tests pass offline. The CPU group kernel it is meant to call is `grouped_expert_ffn` in
  `crates/gnostral-cpu-kernels/` (Q-029).
- **Not landed:** the engine patch series. No engine code has been compiled or run with it.
- No measurement of any kind exists.

## Non-claims

- The landed crate proves bookkeeping and CPU reference semantics. It proves nothing about GPU
  dispatch, engine throughput or the M1 gate.
- No speedup, latency or VRAM claim. A plan that is correct may still be slower.
- No claim that one dispatch per layer is sufficient for the 1.7× gap; that is what M1 measures.
- Nothing here changes routing, top-k selection, expert count, quantization or activation semantics.

## Definition of done

1. Crate tests green offline, including bit-equal parity against the naive oracle.
2. One sequenced patch series integrated in the fork behind a Cargo feature, not an ad-hoc edit.
3. A retained receipt showing identical tokens to the baseline on the frozen prompt set.
4. The engine-level number is produced by Q-031 under M1, and is not claimed by this quest.
