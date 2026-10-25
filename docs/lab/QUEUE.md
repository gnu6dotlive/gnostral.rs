# Lab queue

Maintained by the cloud session; read [`CONTRACT.md`](CONTRACT.md) first. Take the items in order. An
item is done when its report in [`reports/`](reports/README.md) says `DONE` or `REFUSED-BY-GATE`; the
cloud then journals it and updates this file. `<date>` is `yyyymmdd`, the day you run it.

Last updated: 2026-10-25, engine series `0001` to `0011`. Apply `0011` during L-001 if the tree stops at
`0010`, and rebuild (contract section 7) before L-004.

## L-001 · Session baseline and the candle-strata question

**Why:** a fresh worker must prove what the machine holds before measuring anything. F-036 left one
fact `UNKNOWN`: whether the lab's candle-strata implements `QTensor::indexed_gemv` on x86. If it does
not, the plain engine path dequantises every expert at every step on this machine too.

1. Run the start-of-session block of the contract (section 6) and put every output in the report.
2. In the candle-strata tree: `grep -n "fn indexed_gemv" candle-core/src/quantized/mod.rs`, then read
   that function. Report only: its line number; whether the body is compiled on x86_64 or only under
   `#[cfg(target_arch = "aarch64")]`; whether it returns `Ok(None)` on x86. **No code in the report.**
3. Inventory what is already built, without rebuilding:
   ```sh
   sha256sum .ci-cache/mistralrs-target/release/examples/gnostral_capture \
             .ci-cache/mistralrs-target/release/gnostral-bench
   ```
   and the llama.cpp `llama-server` used by the M1 lane (its path is in
   `evidence/runs/m1-qwen3-30b-cpu-experts-v3-20261024/plan-cpu-experts.json`): `--version` output and
   SHA-256.
4. Fill the table of contract section 5 with what you found; mark each line `same` or `differs`.

**Commit:** the report only. **Done when:** every line of section 5 is confirmed or corrected, and the
`indexed_gemv` question has a yes/no answer.

## L-002 · Header inventory of the local models

**Why:** F-026 asks whether the owner's target model loads in this engine at all, and through which
expert path. The inventory reads GGUF headers only (seconds, no tensor data).

```sh
R=evidence/runs/model-inventory-lab-<date>; mkdir -p $R
ls -la /mnt/models/active/gnostral/ > $R/listing.txt
# for each model (all shards of one model on one command line):
python3 harness/model-inventory/gguf_inventory.py --json <file.gguf> [<shard2> ...] > $R/<model-name>.json
python3 harness/model-inventory/gguf_inventory.py        <file.gguf> [<shard2> ...] > $R/<model-name>.txt
```

If the owner's Qwen3.5/3.6 target is not under `/mnt/models/active/gnostral/`, ask the owner where it
is; do not download anything. **Commit:** `$R` (scrubbed) and the report, which lists each model's
verdict line. **Done when:** every local model has an inventory, the target included.

## L-003 · Memory bandwidth and expert-kernel microbenchmarks on the lab host

**Why:** every kernel number so far comes from the cloud VM (F-021 to F-025). F-033 measured the CPU
experts at 21.3 ms per decode step and reading about 30.8 GB/s; whether that is the memory ceiling of
this machine is the open question. Stop the owner's container first.

```sh
export CARGO_TARGET_DIR=~/Projects/gnostral.rs/.ci-cache/kernels-target
cargo build --release --locked --manifest-path crates/gnostral-cpu-kernels/Cargo.toml --bin kquant_bench
B=$CARGO_TARGET_DIR/release/kquant_bench
R=evidence/runs/kernel-microbench-lab-<date>; mkdir -p $R
lscpu > $R/lscpu.txt; nproc > $R/nproc.txt; sha256sum $B > $R/binary.sha256
for T in 1 4 8 20 $(nproc); do $B --mode bandwidth --threads $T > $R/bandwidth-t$T.json; done
for T in 1 4 8 20 $(nproc); do $B --mode ffn --pool-experts 128 --numerics candle --threads $T > $R/ffn-pool128-t$T.json; done
$B --mode ffn --pool-experts 0 --numerics candle --threads 20 > $R/ffn-hot-t20.json
$B --mode dispatch --threads 20 > $R/dispatch-t20.json
$B --mode dot > $R/dot.json
```

Each file is one JSON object. **Commit:** `$R` and the report, with a table per thread count of
`median_gb_per_s` for `bandwidth` and `ffn-pool128`. **Done when:** all files exist; interpretation is
the cloud's job.

## L-004 · flash-attn build, interleaved A/B, kernel counts

**Why:** F-035: without the `flash-attn` feature, CUDA attention copies the KV heads 8 times per layer
(18 `copy2d` launches of 42 per layer). With it, groups up to 8 run natively. This measures what that
is worth on M1, gates first.

1. Build into a separate directory (long: flash kernels; add `-j 4` if memory runs short):
   ```sh
   cd vendor/mistralrs-strata
   CARGO_TARGET_DIR=~/Projects/gnostral.rs/.ci-cache/flash-target cargo build --release -p mistralrs \
     --example gnostral_capture --features cuda,flash-attn,gnostral-expert-residency,gnostral-trace
   cd ~/Projects/gnostral.rs
   ```
   If the build fails, the report carries the last 60 lines of the error; stop there.
2. A/B, container stopped (about 12 captures of 32 prompts):
   ```sh
   R=evidence/runs/m1-flash-attn-ab-<date>
   python3 harness/diag/binary_ab.py \
     --bin noflash=.ci-cache/mistralrs-target/release/examples/gnostral_capture \
     --bin flash=.ci-cache/flash-target/release/examples/gnostral_capture \
     --model /mnt/models/active/gnostral/Qwen3-30B-A3B-Q2_K.gguf --layers 48 --experts 128 \
     --device cuda --out $R
   ```
   Both builds must reproduce their own warmups; they are not expected to agree with each other.
3. Kernel counts for both builds, same session, one prompt (the `.nsys-rep` files stay out of the
   repository):
   ```sh
   export GNOSTRAL_EXPERT_RESIDENCY_CPU=1 GNOSTRAL_EXPERT_CACHE_BYTES=1 \
          GNOSTRAL_EXPERT_PROFILE=$PWD/$R/profile-empty.json \
          GNOSTRAL_MOE_EXECUTOR=grouped GNOSTRAL_CPU_EXPERT_KERNEL=gnostral
   for b in noflash flash; do
     case $b in noflash) BIN=.ci-cache/mistralrs-target;; flash) BIN=.ci-cache/flash-target;; esac
     nsys profile -t cuda,osrt -o ~/nsys-$b-<date> $BIN/release/examples/gnostral_capture \
       --model-dir /mnt/models/active/gnostral --file Qwen3-30B-A3B-Q2_K.gguf \
       --manifest harness/quality/prompts/quality-prompts-32-20261023.json --promptset-sha256 x \
       --engine gnostral --max-tokens 32 --out $R/nsys-$b-capture.json --no-prefix-cache \
       --only q19-summary-tech --gpu-layers 48 > $R/nsys-$b.stdout 2>&1
     nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
       --format csv --output $R/nsys-$b ~/nsys-$b-<date>.nsys-rep
   done
   ```
4. Start the container again.

**Commit:** `$R` (scrubbed: the stdout files hold home paths) and the report with the A/B verdict, its
`chance_of_separated_ranges_if_equal`, VRAM per build and the `copy2d` instance counts. **Done when:**
`ab-summary.json` says `COMPARED` or `REFUSED` with reasons, and both nsys summaries exist.

## L-005 · The owner's target with expert residency (M2), gates first

**Why:** engine patch `0011` gives `qwen35moe` the residency path; on the cloud fixture it removed the
dequantise-every-expert cost and its three lanes are bit-identical (F-038). Whether that holds on the
real model, on CUDA, with the real candle-strata, is this item. **Needs:** L-001 done with the tree at
`0011` and rebuilt; L-002 found the target and its verdict is not `REFUSED_AT_LOAD`. Residency needs a
**single-file** GGUF: if the target is split in shards, stop and report.

Take from the target's L-002 inventory JSON: `L` = `qwen35moe.block_count`, `E` = `qwen35moe.expert_count`,
`H` = `qwen35moe.embedding_length`, `I` = `qwen35moe.expert_feed_forward_length`.

1. Smoke, one prompt, all layers on the card, every expert on the CPU (container stopped):
   ```sh
   R=evidence/runs/m2-<model-name>-smoke-<date>; mkdir -p $R
   M=<target.gguf>
   python3 -c "import hashlib,json,sys; h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(json.dumps({'schema':1,'model_sha256':h,'layers':int(sys.argv[2]),'experts_per_layer':int(sys.argv[3]),'ranking':[]}))" $M $L $E > $R/profile-empty.json
   GNOSTRAL_EXPERT_CACHE_BYTES=1 GNOSTRAL_EXPERT_PROFILE=$R/profile-empty.json GNOSTRAL_MOE_EXECUTOR=grouped \
     .ci-cache/mistralrs-target/release/examples/gnostral_capture --model-dir $(dirname $M) --file $(basename $M) \
     --manifest harness/quality/prompts/quality-prompts-32-20261023.json --promptset-sha256 x --engine grouped \
     --max-tokens 32 --out $R/smoke.json --no-prefix-cache --only q01-capital --gpu-layers $L > $R/smoke.stdout 2>&1
   ```
   The report carries: exit code, the `GNOSTRAL_PLACEMENT` and `GNOSTRAL_METRICS` lines, the residency
   lines of the log (`grep residency $R/smoke.stdout`), `nvidia-smi` memory during the run, and the decoded
   text of the 32 tokens if it is readable. A load error ends the item: report its last 40 lines.
2. Only if the smoke ran: the four-lane comparison, the M1 protocol on the target:
   ```sh
   python3 harness/synthetic-moe/run_e2e.py \
     --capture-bin .ci-cache/mistralrs-target/release/examples/gnostral_capture \
     --bench-bin .ci-cache/mistralrs-target/release/gnostral-bench \
     --out evidence/runs/m2-<model-name>-cpu-experts-<date> --model $M \
     --layers $L --experts $E --hidden $H --moe-inter $I \
     --device cuda --cpu-experts-only --repetitions 5 \
     --llama-server /mnt/sandbox/sessions/gnostral-q009-llamacpp-cuda-20261010/build/llama-cuda/bin/llama-server \
     --llama-args="-ngl 99 --cpu-moe -t 8 -c 4096 --fit off -lv 4" \
     --llama-placement "llama.cpp offloaded=<L+1>/<L+1> cpu_moe=on"
   ```
   (M1 used these llama.cpp arguments; if that `llama-server` build cannot load `qwen35moe`, the report
   says so and the comparison runs without the llama.cpp lane: drop the three `--llama-*` flags.)
   The `gnostral` lane runs only Q2_K/Q3_K experts: if L-002's expert path for the target is
   `residency_candle_kernel`, expect that lane to fail and the comparison to be refused; report it and
   stop, the cloud will add lane selection to `run_e2e.py`.

**Commit:** both run directories (scrubbed) and the report: per lane, parity, self-reproduction,
placement, decode tok/s ranges, VRAM; the `pareto.md` verdict as written. **Done when:** the comparison
says `COMPARED` or is refused with its reasons.
