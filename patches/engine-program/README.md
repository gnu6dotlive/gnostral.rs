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
| `0003-mistralrs-strata-expert-demand-profile-q027.patch` | `43db8ef6...7a0532` | On top of `0002`: `GNOSTRAL_EXPERT_PROFILE_OUT=path` records routed demand per expert and writes a profile in the exact schema `GNOSTRAL_EXPERT_PROFILE` loads (ranked by observed routes) plus `path.demand.json` with the counts; written atomically every 256 layer forwards and on drain |
| `0004-mistralrs-strata-gnostral-kernel-candle-compat-q029.patch` | `c9d96463...dc773f` | On top of `0003`: the `gnostral` lane calls `grouped_expert_ffn` with `Numerics::CandleCompat`, so it is bit-equal to the `candle` lane; the residency test asserts bit equality |
| `candle-strata-public-reconstruction.BUILD-ONLY.patch` | `f9549036...98e7b` | **Not for the lab.** Lets the public candle `35d7ae7` stand in for the unpublished lab candle-strata (`f0f334aa`): reconstructed `expert_slice_to`, the Q-010 `cpu_expert_matmul_k`, `cuda_allocation_bytes`, and load-refusing Bonsai PQ2_0/PTQ1_0 stubs |

The lab tree already carries the real candle-strata; apply `0001` to `0004` in order there, not the BUILD-ONLY patch.

## Runtime selection (same binary, A/B by environment)

| Variable | Values | Default |
| --- | --- | --- |
| `GNOSTRAL_MOE_EXECUTOR` | `grouped`, `per-route` | `grouped` |
| `GNOSTRAL_CPU_EXPERT_KERNEL` | `candle`, `gnostral` | `candle` |
| `GNOSTRAL_MOE_GPU_ROWS` | `per-row`, `batched` | `per-row` |
| `GNOSTRAL_EXPERT_RESIDENCY_CPU` | `1` | unset (pilot stays CUDA-only) |
| `GNOSTRAL_TRACE` (needs feature `gnostral-trace`) | `1` | unset; `GNOSTRAL_TRACE_OUT` names the receipt |
| `GNOSTRAL_EXPERT_PROFILE_OUT` | path | unset (no demand recording) |

- `grouped` builds one plan per layer: GPU groups are enqueued first, host groups run while the device
  works, one synchronisation collects device results, and the scatter-add follows the per-route
  `(token, ordinal)` order. Each expert is sliced once per group instead of once per route.
- `candle` keeps the vendored k_quants kernel, so the CPU result is **bit-equal** to `per-route`.
- `gnostral` runs `grouped_expert_ffn` (Q2_K/Q3_K x Q8_K, persistent pool) directly on the borrowed
  GGUF bytes; it requires a SiLU expert activation. With `Numerics::CandleCompat` the integer part
  uses the widest ISA (AVX-512BW, AVX2, scalar) and the float part replays the Candle kernel this
  host dispatches to (AVX2+FMA lanes, else `vec_dot_unopt`), so it is **bit-equal** to `candle`.
  This holds against the vendored public candle `35d7ae7`; if the lab candle-strata changed its
  `Q2_K`/`Q3_K` `vec_dot`, the Q-030 gate catches it.
- `per-row` keeps m=1 CUDA kernels (same numerics as `per-route`, no per-route sync); `batched`
  switches to the m>1 kernel family and must pass Q-030 before use.

## Verified here (CPU only, no CUDA build)

```sh
cargo test -p mistralrs-core --features gnostral-expert-residency,gnostral-trace --lib moe   # 53 passed, 1 ignored
```

- `grouped_cpu_executor_is_bit_equal_to_per_route`: 9 tokens x top-4 over 6 Q2_K/Q3_K experts,
  two route orders, every output bit equal.
- `gnostral_cpu_kernel_is_bit_equal_to_candle`: every output bit equal to the `candle` per-route
  lane; refuses a non-SiLU activation.
- `executor_configuration_is_explicit`: invalid or contradictory settings are errors.

End to end on a synthetic Qwen3-MoE GGUF (`harness/synthetic-moe/`, receipt
`evidence/runs/synthetic-moe-e2e-20261023.json`), 32 frozen prompts x 32 tokens through the real
GGUF load, residency and sampling path:

- `grouped` vs `per-route`: 32/32 prompts token-identical; `gnostral`: 32/32 (30/32 before
  `0004`, two late divergences from the float combination order).
- Trace on vs off: 32/32 token-identical (Q-042 semantic neutrality).
- `gnostral-bench run`: three lanes pinned, warmup + 3 interleaved rounds, every run kept, every
  lane passes strict greedy parity; the comparison is still refused because driver health is
  `UNKNOWN` on a CPU host, exactly as the rules require.
- The upstream CPU MoE path (no residency) dequantises the expert stacks to F32 per call and keeps
  F32 activations, so it is not token-identical to the Q8_K residency path (12/32 on this fixture).

Recording a profile for a real model: run the frozen workload with residency enabled and an
empty-ranking input profile (budget `1`, so every expert runs on the host), with
`GNOSTRAL_EXPERT_PROFILE_OUT` set; the written file is directly loadable as
`GNOSTRAL_EXPERT_PROFILE` with the real budget. On the synthetic fixture: 2,048 layer forwards, 32
experts ranked, the recorded profile reloaded by the engine. If the process exits abruptly the
profile reflects demand up to the last multiple of 256 forwards.

## Not verified

- No CUDA compile and no GPU run: the `GpuResident` path, `GpuRows::Batched` and the CUDA-gated
  residency tests have not been executed.
- No token-parity receipt on a real model (Q-030) and no speed measurement (Q-031, M1).
- No claim that `grouped` or `gnostral` is faster.
