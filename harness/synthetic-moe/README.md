# Synthetic Qwen3-MoE smoke (CPU)

A functional end-to-end check of the patched engine without downloading any model.

| File | Purpose |
| --- | --- |
| `make_qwen3moe_gguf.py` | Writes a random, well-formed Qwen3-MoE GGUF (F32 attention/norms/router, Q2_K gate/up and Q3_K down expert stacks, byte-level GPT-2 tokenizer, chat template). Fixed seed, ~12 MB. |
| `make_qwen35moe_gguf.py` | The same for `qwen35moe`, the architecture of the owner's target family (F-026, F-027): Gated DeltaNet linear attention on three layers of four, gated full attention on the fourth, routed plus shared experts. Its docstring holds the tensor and metadata contract read from both the engine's and llama.cpp b11857's sources; both load the file. ~27 MB. |
| `qwen35moe_reference.py` | Float64 forward pass of a `qwen35moe` GGUF written from transformers' `modeling_qwen3_5_moe.py` and llama.cpp's converter transformations; judges an engine capture by teacher forcing (reference argmax per step, logprob gaps). Needs numpy and llama.cpp's `gguf-py`. `--expert-type f32` on the generator removes quantisation noise; `--zero` and the other expert types are diagnostics (F-028). `--expert-activations q8_k` rounds routed-expert inputs as Q8_K engines do (llama.cpp, candle, the residency path; F-039). |
| `run_e2e.py` | Generates the model, captures the `per-route` reference with `gnostral_capture`, then runs `gnostral-bench run` over `per-route`, `grouped` and `gnostral`; each lane checks greedy parity with the Q-030 oracle before printing its metrics. |

```sh
# engine side: apply patches/engine-program/0001 and 0002, then
cargo build --release --example gnostral_capture -p mistralrs --features gnostral-expert-residency,gnostral-trace
cargo build --release --manifest-path harness/bench/Cargo.toml
python3 harness/synthetic-moe/run_e2e.py \
  --capture-bin <engine>/target/release/examples/gnostral_capture \
  --bench-bin harness/bench/target/release/gnostral-bench --out /tmp/synthetic-e2e
```

The CPU residency path is enabled with `GNOSTRAL_EXPERT_RESIDENCY_CPU=1`,
`GNOSTRAL_EXPERT_CACHE_BYTES=1` and a profile with an empty ranking (the script writes it).
`--device cuda` drops `--cpu` from the capture tool (default `cpu`). The plan also carries a `driver_probe`
that scans the kernel log for `NVRM: Xid` and prints `GNOSTRAL_DRIVER NO_XID|XID`; the bench refuses a
GPU comparison without `NO_XID`.

What this proves and what it does not: see
[`evidence/runs/synthetic-moe-e2e-20261023.json`](../../evidence/runs/synthetic-moe-e2e-20261023.json).
The model is noise; no timing it produces is engine performance, and no quality claim follows.

## Device run (`--device cuda`)

Two configurations are run and kept apart in `e2e-receipt.json`: `cpu-experts` (budget of one byte, empty
profile) and `mixed-experts` (a recording pass with `GNOSTRAL_EXPERT_PROFILE_OUT`, then
`--mixed-fraction` of the expert bytes as `GNOSTRAL_EXPERT_CACHE_BYTES`). Size the model with
`--hidden/--layers/--experts/--moe-inter`. Each lane samples `nvidia-smi` every 200 ms, reports
`vram_used_mib` (whole-card peak) and a `GNOSTRAL_DEVICE` line, and **fails** if the capture pid never
shows up as a compute app. Without `nvidia-smi` the values stay `UNKNOWN`. Lane verdicts compare full
ranges: overlapping min/max is "inconclusive", never a ranking.

## Identity checks (bench plan v2)

- Every capture runs with the prefix cache off (`--no-prefix-cache`, FINDINGS F-014) and records the
  top-2 logprobs.
- On CUDA the capture gets an explicit `--gpu-layers` (default: all layers, override with
  `--gpu-layers N`); the automatic device map is never used for a benchmark (F-011). The engine's own
  placement report (patch `0008`) is forwarded as `GNOSTRAL_PLACEMENT` and must equal the plan's
  `expected_placement` on every run.
- A lane's first capture in a configuration (the warmup) becomes its reference; every measured capture
  must match it in tokens and logprob bits (`quality_oracle.py self-repro`).
- The driver probe is `harness/bench/driver_probe.py`: `NO_XID` only after reading the boot's kernel
  log with an NVIDIA driver loaded, `UNKNOWN` otherwise (F-016). On a CPU-only host the comparison is
  therefore refused, by design.
- `--llama-server` adds a llama.cpp lane (parity class `self`), see [`harness/lanes/`](../lanes/README.md).
