# Model inventory

`gguf_inventory.py` reads a GGUF header (no tensor data, so seconds for a 20 GB file) and says what the
model asks of this engine tree: architecture, hyperparameters, bytes per tensor class and quantisation
type, and for each layer's routed experts which engine path can run them.

```sh
python3 harness/model-inventory/gguf_inventory.py /models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
python3 harness/model-inventory/gguf_inventory.py --json model-00001-of-00002.gguf model-00002-of-00002.gguf
```

Expert paths, from fastest to refused:

| Path | Meaning |
| --- | --- |
| `gnostral_kernel` | `qwen3moe`, gate/up and down Q2_K or Q3_K: the Q-029 kernels (`GNOSTRAL_CPU_KERNEL=gnostral`) |
| `residency_candle_kernel` | `qwen3moe`, residency accepts the types; host experts run candle's quantised matmul |
| `loads_residency_refuses_types` | `qwen3moe`, the engine loads the types but residency refuses these stacks |
| `loads_no_residency_for_architecture` | the engine loads the model, but residency is a `qwen3moe`-only pilot (F-027) |
| `refused_at_load` | at least one type has no mapping (MXFP4 maps only for gpt-oss experts); the load fails |
| `fused_gate_up_unchecked` | a fused `ffn_gate_up_exps` tensor; whether residency could use it is not checked |

The verdict also says `UNKNOWN_ARCHITECTURE` when `general.architecture` is not in the engine's native GGUF
registry, and `LOADS_WITHOUT_PROGRAM` when every expert layer loads but none gets the program's engine work.

The support sets are copied from the engine sources the JSON lists under `support_sources`. They
describe this repository's engine tree. A lab build with other types must update them first, or the
verdict is about the wrong engine.

The header is checked, not trusted: each tensor's size is computed from shape and type, and a tensor
that overlaps the next one or runs past the end of its file is an error (exit 2).

Why this exists: `docs/FINDINGS.md` F-026 and F-027.
