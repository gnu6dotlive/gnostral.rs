# Benchmark lanes

Scripts that run one engine once and print the lines `gnostral-bench run` reads (plan v2):

| Line | Meaning |
| --- | --- |
| `GNOSTRAL_PLACEMENT <canonical>` | the placement the engine reports; must equal the plan's `expected_placement` on every run |
| `GNOSTRAL_SELF_REPRO REFERENCE\|PASS\|FAIL ...` | the warmup is the lane's reference; measured runs must match it in tokens and logprob bits |
| `GNOSTRAL_PARITY PASS\|FAIL` | exact-class lanes only: token identity with the reference lane |
| `GNOSTRAL_METRICS {json}` | decode/prefill tok/s, TTFT, whole-card VRAM when sampled |
| `GNOSTRAL_DEVICE ...`, `GNOSTRAL_AGREEMENT {json}`, `GNOSTRAL_LANE_DETAIL ...` | informational, kept in the bundle |

| File | Purpose |
| --- | --- |
| `lanelib.py` | shared plumbing: identity hashes, the GPU sampler (`nvidia-smi` every 200 ms, never a guessed value), the self-reproducibility rule, Q-030 token agreement |
| `llama_server_lane.py` | the llama.cpp lane (Q-025 three-way): `llama-server`, strictly greedy, prompt cache off, parity class `self` |
| `fake_llama_server.py`, `test_llama_server_lane.py` | a stand-in server with llama.cpp's response shapes for CI; not an engine |

## llama.cpp lane

```sh
python3 harness/synthetic-moe/run_e2e.py ... \
  --llama-server /path/to/llama-server \
  --llama-args="-ngl 99 --cpu-moe -t 8 -c 4096 --fit off" \
  --llama-placement "llama.cpp offloaded=49/49 cpu_moe=on"
```

- Use the `=` form for `--llama-args`: a value such as `--cpu-moe` would otherwise be read as an option.
- `-ngl` is required. llama.cpp can fit unset parameters to free memory; pass `--fit off` where the build
  supports it. The placement is read from the server's own `offloaded A/B layers to GPU` log line and
  pinned; any drift fails the lane, exactly as for our engine.
- Build `llama-server` with `-DBUILD_SHARED_LIBS=OFF`: the plan pins the executable, and a shared build can
  change underneath it. The lane prints `build_tree_sha256` over the executable and its `libllama`/`libggml`
  libraries either way.
- The server records no probabilities for tokens that end inside an incomplete UTF-8 character
  (FINDINGS F-019). The lane keeps the full token list and the server's entries as reported; it does not
  invent one logprob per token.
- The lane's token agreement with the reference is reported, not gated: another engine computes in a
  different order and is held to reproducing itself (Q-025 semantic validity), with divergences explained
  in the Q-030 ledger.
