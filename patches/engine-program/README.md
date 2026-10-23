# Engine-program patches (mistralrs-strata)

Research quests: Q-026 (grouped MoE executor), Q-029 (CPU expert kernel). Local research patches;
not submitted upstream, not upstream-accepted.

## Base

`0001` applies on top of the engine tree recorded in
`evidence/runs/mistralrs-strata-vendor-20261023.json`: public mistral.rs
`24dbf5c256f232176ee5949485ba264049407fbe` plus
`patches/strata-residency/20261023-mistralrs-strata-integration.patch` (1,889 files, byte-identical
to the inventory). `git apply --check` against that tree passes.

| Patch | SHA-256 | Contents |
| --- | --- | --- |
| `0001-mistralrs-strata-grouped-executor-q026-q029.patch` | `3b950fb6...d88642` | Grouped executor in `residency.rs`, `forward_with` from `backends.rs`, optional deps on `gnostral-moe-exec` and `gnostral-cpu-kernels` under the existing `gnostral-expert-residency` feature, parity tests |
| `0002-mistralrs-strata-trace-cpu-optin-capture-q042-q030-q031.patch` | `988cc52c...7220d4` | On top of `0001`: Q-042 timers (`gnostral-trace` feature, `GNOSTRAL_TRACE=1`, reset after the warmup dummy run), the all-CPU residency opt-in `GNOSTRAL_EXPERT_RESIDENCY_CPU=1`, and the `gnostral_capture` example (Q-030 captures + `GNOSTRAL_METRICS` for Q-031) |
| `candle-strata-public-reconstruction.BUILD-ONLY.patch` | `f9549036...98e7b` | **Not for the lab.** Lets the public candle `35d7ae7` stand in for the unpublished lab candle-strata (`f0f334aa`): reconstructed `expert_slice_to`, the Q-010 `cpu_expert_matmul_k`, `cuda_allocation_bytes`, and load-refusing Bonsai PQ2_0/PTQ1_0 stubs |

The lab tree already carries the real candle-strata; apply `0001` then `0002` there, not the BUILD-ONLY patch.

## Runtime selection (same binary, A/B by environment)

| Variable | Values | Default |
| --- | --- | --- |
| `GNOSTRAL_MOE_EXECUTOR` | `grouped`, `per-route` | `grouped` |
| `GNOSTRAL_CPU_EXPERT_KERNEL` | `candle`, `gnostral` | `candle` |
| `GNOSTRAL_MOE_GPU_ROWS` | `per-row`, `batched` | `per-row` |
| `GNOSTRAL_EXPERT_RESIDENCY_CPU` | `1` | unset (pilot stays CUDA-only) |
| `GNOSTRAL_TRACE` (needs feature `gnostral-trace`) | `1` | unset; `GNOSTRAL_TRACE_OUT` names the receipt |

- `grouped` builds one plan per layer: GPU groups are enqueued first, host groups run while the device
  works, one synchronisation collects device results, and the scatter-add follows the per-route
  `(token, ordinal)` order. Each expert is sliced once per group instead of once per route.
- `candle` keeps the vendored k_quants kernel, so the CPU result is **bit-equal** to `per-route`.
- `gnostral` runs `grouped_expert_ffn` (Q2_K/Q3_K x Q8_K, persistent pool) directly on the borrowed
  GGUF bytes; it requires a SiLU expert activation and agrees with candle within tolerance, not bit
  for bit.
- `per-row` keeps m=1 CUDA kernels (same numerics as `per-route`, no per-route sync); `batched`
  switches to the m>1 kernel family and must pass Q-030 before use.

## Verified here (CPU only, no CUDA build)

```sh
cargo test -p mistralrs-core --features gnostral-expert-residency,gnostral-trace --lib moe   # 52 passed, 1 ignored
```

- `grouped_cpu_executor_is_bit_equal_to_per_route`: 9 tokens x top-4 over 6 Q2_K/Q3_K experts,
  two route orders, every output bit equal.
- `gnostral_cpu_kernel_tracks_candle_within_tolerance`: within `1e-3` of the row scale; refuses a
  non-SiLU activation.
- `executor_configuration_is_explicit`: invalid or contradictory settings are errors.

End to end on a synthetic Qwen3-MoE GGUF (`harness/synthetic-moe/`, receipt
`evidence/runs/synthetic-moe-e2e-20261023.json`), 32 frozen prompts x 32 tokens through the real
GGUF load, residency and sampling path:

- `grouped` vs `per-route`: 32/32 prompts token-identical; `gnostral`: 30/32 (two late divergences).
- Trace on vs off: 32/32 token-identical (Q-042 semantic neutrality).
- `gnostral-bench run`: three lanes pinned, warmup + 3 interleaved rounds, every run kept; the
  comparison is refused because driver health is `UNKNOWN` on a CPU host and the `gnostral` lane
  fails strict parity, exactly as the rules require.
- The upstream CPU MoE path (no residency) dequantises the expert stacks to F32 per call and keeps
  F32 activations, so it is not token-identical to the Q8_K residency path (12/32 on this fixture).

## Not verified

- No CUDA compile and no GPU run: the `GpuResident` path, `GpuRows::Batched` and the CUDA-gated
  residency tests have not been executed.
- No token-parity receipt on a real model (Q-030) and no speed measurement (Q-031, M1).
- No claim that `grouped` or `gnostral` is faster.
