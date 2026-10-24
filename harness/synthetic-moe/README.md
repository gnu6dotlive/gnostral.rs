# Synthetic Qwen3-MoE smoke (CPU)

A functional end-to-end check of the patched engine without downloading any model.

| File | Purpose |
| --- | --- |
| `make_qwen3moe_gguf.py` | Writes a random, well-formed Qwen3-MoE GGUF (F32 attention/norms/router, Q2_K gate/up and Q3_K down expert stacks, byte-level GPT-2 tokenizer, chat template). Fixed seed, ~12 MB. |
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
