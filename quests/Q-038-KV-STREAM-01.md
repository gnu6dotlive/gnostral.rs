# Q-038 — KV-STREAM-01

Review date: 2026-10-23
State: **PROPOSED** — tiering, streaming-plan arithmetic and ticketed block moves landed offline (`gnostral-memory-plan::stream`); no streaming path in the engine; no measurement exists.
Authorization: one GPU session per iteration after Q-028 lands; KV layout and streaming work in the
paged-attention path and the pinned-host staging area. This quest does not authorize a KV format
change that alters tokens, a compression scheme chosen for scoring, or a context-tier claim before
its receipt.
Workstream: W2 · Milestone: M2 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can long-context KV live in pinned host memory with prefetch, so that **32K tokens run on an 8 GiB
card** at the documented decode floor, with tokens exactly equal to the short-context baseline?

Strata's own KV_PREFETCH ABBA showed ~0 gain in its first test. Prefetch is therefore a hypothesis
inside this quest, not a promised mechanism: the streaming and the parity gate may succeed while
prefetch is recorded as neutral.

## Deliverable and file seams

- KV streaming path: eviction of cold KV blocks to pinned host memory, explicit ticket/lease,
  prefetch on the copy stream, stall accounting.
- Integration with the segmented budget from `crates/gnostral-memory-plan/` and Q-027.
- Long-context tier harness: 1K, 8K, 32K on the same prompts, exact token comparison.
- Receipts under `evidence/runs/`, raw logs retained.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Exact tokens | 32K output token-identical to the short-context baseline | any divergence |
| Context reach | 32K tokens complete on 8 GiB without OOM | OOM, or a silent truncation |
| Decode floor | 32K decode ≥ 15 tok/s (the program's long-context target) | below the floor |
| Stall budget | time lost to KV stalls measured and reported per step | stalls unmeasured or inferred |
| Memory | whole-card peak inside the envelope; host working set ≤ 24 GiB | exceeding either |
| Prefetch honesty | prefetch gain reported even if it is zero or negative | prefetch assumed beneficial |
| Tier truth | if 32K is unreachable, publish the honest tier instead of claiming it (D3) | a claim above the measured tier |

## Non-claims

- No 32K, decode, stall or bandwidth number exists.
- Exact-token parity on the frozen prompts is not a quality guarantee at every context length.
- Streaming is not compression; it moves bytes, it does not make them free.
- D3 may legitimately end with smaller context tiers and no 32K claim.

## Definition of done

1. A receipt with 1K/8K/32K samples, exact-token comparison, stall breakdown and peak memory, all
   samples retained.
2. Either 32K under the gate, or a published context tier with the negative recorded.
3. Prefetch's contribution stated as measured, including zero.
