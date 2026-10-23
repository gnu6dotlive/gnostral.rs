# Q-029 — CPU-KERNEL-Q8-01

Review date: 2026-10-23
State: **PROPOSED** — no measurement exists; no kernel exists.
Authorization: CPU-only kernel work in the vendored Candle quant path and `mistralrs-quant/`-adjacent
code, with microbenchmarks on this machine. This quest does not authorize a new CUDA kernel family, a
quantization change for benchmarking, or a change to block formats on disk.
Workstream: W1 · Milestone: M1 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

llama.cpp's core advantage in the Q-009/Q-010 comparison is a quantized CPU expert path: activations
quantized once to Q8_0, then multiplied against Q2_K/Q3_K blocks in a vectorized kernel with a thread
pool, instead of dequantizing to F32 per expert. Can the same contract be implemented here with
**bit-reproducible parity** to the ggml reference on the same blocks?

## Deliverable and file seams

- Q8_0 activation quantization once per layer, dequantized only for accumulation.
- Fused gate/up matmul where the model wiring allows it.
- Batched `[tokens_per_expert, hidden] × [hidden, inter]` quantized GEMM over grouped rows from Q-026.
- Runtime dispatch: AVX2 and AVX-512 paths plus a scalar fallback that must agree with them.
- Bounded thread pool; reduction order fixed per kernel so thread count does not change results.
- Vendored Candle: `candle-core/src/quantized/{k_quants.rs,mod.rs}` (patch series, not ad-hoc edits).

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Dequant parity | dequantized Q2_K/Q3_K blocks **byte-equal** to the vendored reference on a fixed block corpus | any mismatch; block-format drift |
| Quantize round-trip | Q8_0 quantize → dequantize matches the reference within a tolerance **frozen in the protocol before the first run** | tolerance chosen after seeing results |
| GEMM parity | per-element error against the ggml reference within the same frozen tolerance | exceeding it, or comparing against a different block layout |
| Fused gate/up | fused result equals unfused gate and up results within the same tolerance | fusion that changes routing or gating |
| Thread invariance | identical output for 1, 4 and 8 threads on the fixed corpus | thread-count-dependent output without a declaration |
| ISA coverage | scalar, AVX2 and AVX-512 paths agree; a missing ISA falls back explicitly | silent fallback that changes results |
| Microbench | median of ≥ 3 measured repetitions per lane, all samples retained | a single sample reported as a result |
| M1 numbers | decode ≥ 35 tok/s is [M1](../docs/ENGINE_PROGRAM.md#milestone-ladder)'s gate, not this quest's | quoting a synthetic-block figure as engine throughput |

## Non-claims

- A synthetic-block microbenchmark is not engine throughput and is never reported as tok/s for the
  model. The only engine numbers come from Q-031 under M1.
- No claim of llama.cpp parity or superiority; the gate is parity with the reference implementation,
  not a speed comparison.
- No claim that Q8_0 activations preserve quality in this engine until Q-030 runs on real weights.
- No new kernel family beyond the quantized CPU path already implied by the vendored kernels.

## Definition of done

1. Kernel tests green offline with the frozen tolerance recorded in the receipt.
2. Microbench with warmup and ≥ 3 repetitions, all samples kept, ISA and thread count recorded.
3. The kernel is callable from the Q-026 grouped executor, with parity tests over grouped rows.
4. A documented negative if the kernel cannot reach the reference's per-block cost; the decision is
   then D2's, not a larger claim.
