# gnostral.rs — upstream systems intake

**Date:** 2026-10-21  
**Status:** research index; no qualification implied

This note records upstream developments that can materially change an existing gnostral.rs research rail. It is intentionally separate from the product roadmap and does not authorize runtime adoption.

## Current intake

| Rail | Upstream development | Local question | Next bounded action |
|---|---|---|---|
| WakeKV / KV residency | Reactive/reversible KV residency plus active work in competing KV offload/eviction systems | Can a cooling attention head be demoted to recoverable CPU memory and promoted again on renewed demand without corruption or unacceptable PCIe stalls? | `WAKEKV-01A`: matched-VRAM baseline vs reversible CPU reservoir; measure head drift, memory saved, refill/stall, TTFT, TPOT, throughput and quality |
| KV representation | Quantized KV work and native quantized attention paths | Is residency policy independent from the representation used in the cold tier? | Only after 01A: compare FP16 vs Q8 (and later Q4 if supported) host reservoir |
| Bonsai | Upstream Hadamard / folded-weight support continues to expand | Are our pinned Bonsai artifacts semantically and operationally compatible with current upstream representation metadata? | `BONSAI-UPSTREAM-01`: metadata audit, semantic parity, then performance comparison |
| Burn Remote v5 | Quantized tensor transport and explicit protocol incompatibility | Can representation/transport be separated cleanly from materialization and residency? | Byte-exact two-process loopback for supported Q8/Q4/FP8/FP4 schemes; v5 acceptance/v4 refusal |
| Strata challengers | GPU-resident MoE LRU and related expert-cache proposals | Which ownership, ordering, lease and reuse invariants survive when expert residency becomes dynamic? | Challenger note + invariant comparison; no change to pinned runtime |
| Ferrum / SLO | TTFT/TPOT/ITL-aware optional scheduling | Can an outer latency policy inform demand without owning materialization? | Q-016 trace oracle first; scheduling experiments only after baseline |
| CubeCL | Explicit stream/communicator ordering for concurrent CUDA collectives | Which transfer-ordering invariants should Strata retain? | Source-level donor note; multi-GPU execution remains blocked by hardware |

## Authority boundary

EngineProvider v0 remains intentionally narrow:

> The supervisor controls intent, lifecycle and resource envelopes. The engine controls materialization.

`modeld` may observe/report residency and resource envelopes. It must not silently become the owner of KV block eviction, compression, promotion, expert placement or page mapping.

## Evidence rule

For every intake item, preserve:

1. exact upstream project/revision;
2. local artifact/engine identity;
3. hardware and bounded profile;
4. upstream fact versus local observation;
5. failed runs and negative controls;
6. explicit non-claims;
7. the smallest falsifiable next experiment.

A donor is not a dependency merely because it is interesting. A local benchmark is not generalized beyond its qualified profile.

## Current gnostral.rs rails

- Q-003: elastic-memory/residency evidence.
- Q-007/Q-008: engine-provider and materialization authority boundary.
- Q-009: Strata native expert residency / grouped prefill / transfer overlap research.
- Q-010: paired borrowed CPU-expert performance evidence.
- Q-012: three-family sequential functional qualification.
- Q-013D→Q-016: authenticated lifecycle, admission and client-observed SLO work.

## Non-goals

This file does not claim multi-model co-residency, arbitrary model placement, generic hot swapping, production KV paging, multi-GPU qualification or cross-vendor support.
