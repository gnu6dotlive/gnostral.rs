"""Write a tiny, fully synthetic Qwen3.5-MoE (`qwen35moe`) GGUF for engine smoke and parity tests.

Same purpose and tokenizer as `make_qwen3moe_gguf.py`, for the architecture of the owner's target model
family (docs/FINDINGS.md F-026, F-027): hybrid layers, Gated DeltaNet linear attention on three layers out
of four and gated full attention on the fourth, routed experts plus a dense shared expert. Weights come
from a fixed seed; the text the model produces is noise.

The tensor and metadata contract below was read from both consumers' sources on 2026-10-24, engine =
vendor/mistralrs-strata, llama.cpp = b11857 (701fd25). A file that only one of them accepts is not a
fixture, so both are required to load it.

- Metadata. Engine (`gguf/normal_registry.rs`, `gguf/normal_config.rs::build_qwen3_next`): expert_count,
  expert_used_count, expert_feed_forward_length, expert_shared_feed_forward_length,
  full_attention_interval, rope.dimension_count, ssm.conv_kernel, ssm.inner_size, ssm.state_size,
  ssm.group_count, ssm.time_step_rank. llama.cpp (`src/models/qwen35moe.cpp`) also requires
  rope.dimension_sections (4 values).
- Layer kinds: layer i is full attention when (i + 1) % full_attention_interval == 0 (both sources).
- Linear layer: attn_qkv [conv_dim, hidden], attn_gate [value_dim, hidden], ssm_beta and ssm_alpha
  [v_heads, hidden], ssm_conv1d [conv_dim, conv_kernel], ssm_dt.bias [v_heads], ssm_a [v_heads],
  ssm_norm [head_dim_v], ssm_out [hidden, value_dim]. llama.cpp takes head_k_dim = head_v_dim =
  ssm.state_size, k_heads = ssm.group_count, v_heads = ssm.time_step_rank; the engine takes head_v_dim =
  ssm.inner_size / v_heads. So ssm.inner_size = state_size * v_heads here, or the two disagree. The
  engine's `tiled` value-head layout for qwen35moe needs v_heads to be a multiple of k_heads.
- ssm_a holds -exp(A_log): the engine binds A_log = log(-ssm_a), so every value is negative.
- Full layer: attn_q [2 * heads * head_dim, hidden] (query and output gate), attn_k and attn_v
  [kv_heads * head_dim, hidden], attn_output [hidden, heads * head_dim], attn_q_norm and attn_k_norm
  [head_dim].
- Every layer: attn_norm and post_attention_norm [hidden], router ffn_gate_inp [experts, hidden], routed
  experts as in the qwen3moe fixture (Q2_K gate/up, Q3_K down), shared expert ffn_gate_shexp and
  ffn_up_shexp [shexp, hidden], ffn_down_shexp [hidden, shexp], ffn_gate_inp_shexp [hidden].
- Norms except ssm_norm are stored as 1 + w, the converter's Qwen3-Next convention: the engine applies
  affine(1, -1) then a (1 + w) RMSNorm, llama.cpp multiplies by the stored value. A stored 1.0 is a unit
  scale in both. ssm_norm is the raw weight in both.

Shapes above are row-major (PyTorch order); GGUF stores them fastest-first.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import random
import sys
from pathlib import Path

import make_qwen3moe_gguf as base

ARCH = "qwen35moe"
ROPE_SECTIONS = [11, 11, 10, 0]
# Diagnostic ablations: the named output projection is written as zeros, so that component adds
# nothing to the residual stream. Used to locate a difference between two engines, never for timing.
ZERO_CHOICES = ("gdn_out", "attn_out", "shexp_down")
# Which expert projections are k-quantised (Q2_K gate/up, Q3_K down); the others are F32.
EXPERT_TYPES = {"kquant": {"gate", "up", "down"}, "f32": set(), "q2k_gate_up": {"gate", "up"}, "q3k_down": {"down"}}


def f32_values(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def check(args: argparse.Namespace) -> None:
    if args.hidden % base.QK_K or args.moe_inter % base.QK_K:
        raise SystemExit("hidden and moe_inter must be multiples of 256 (Q2_K/Q3_K experts)")
    if args.layers < args.full_attention_interval:
        raise SystemExit("layers must be at least full_attention_interval, or no full attention layer exists")
    if args.v_heads % args.k_heads:
        raise SystemExit("v_heads must be a multiple of k_heads (engine tiled value-head layout)")
    if sum(ROPE_SECTIONS) * 2 != args.rope_dim or args.rope_dim > args.head_dim:
        raise SystemExit("rope sections must cover rope_dim / 2, and rope_dim must fit head_dim")


def build(args: argparse.Namespace) -> dict:
    check(args)
    rng = random.Random(args.seed)
    tokens = base.byte_tokens() + ["<|im_start|>", "<|im_end|>"]
    vocab = len(tokens)
    h, heads, kv, hd = args.hidden, args.heads, args.kv_heads, args.head_dim
    e, inter, shexp, layers = args.experts, args.moe_inter, args.shexp_inter, args.layers
    state, k_heads, v_heads = args.state_size, args.k_heads, args.v_heads
    key_dim, value_dim = state * k_heads, state * v_heads
    conv_dim = 2 * key_dim + value_dim

    w = base.Writer()
    w.string("general.architecture", ARCH)
    w.string("general.name", "gnostral-synthetic-qwen35moe")
    for key, value in [
        ("context_length", 4096),
        ("embedding_length", h),
        ("block_count", layers),
        ("attention.head_count", heads),
        ("attention.head_count_kv", kv),
        ("attention.key_length", hd),
        ("attention.value_length", hd),
        ("rope.dimension_count", args.rope_dim),
        ("expert_count", e),
        ("expert_used_count", args.top_k),
        ("expert_feed_forward_length", inter),
        ("expert_shared_feed_forward_length", shexp),
        ("full_attention_interval", args.full_attention_interval),
        ("ssm.conv_kernel", args.conv_kernel),
        ("ssm.inner_size", value_dim),
        ("ssm.state_size", state),
        ("ssm.group_count", k_heads),
        ("ssm.time_step_rank", v_heads),
        ("vocab_size", vocab),
    ]:
        w.u32(f"{ARCH}.{key}", value)
    w.i32s(f"{ARCH}.rope.dimension_sections", ROPE_SECTIONS)
    w.f32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-6)
    w.f32(f"{ARCH}.rope.freq_base", 10_000_000.0)
    w.boolean(f"{ARCH}.expert_weights_norm", True)
    base.write_tokenizer(w, tokens)

    unit = lambda n: base.f32_tensor(rng, n, 0, 1.0)  # noqa: E731
    zero = set(args.zero)

    def maybe_zero(component: str, data: bytes) -> bytes:
        return bytes(len(data)) if component in zero else data
    w.tensor("token_embd.weight", [vocab, h], base.GGML_F32, base.f32_tensor(rng, vocab * h, 0.5))
    for i in range(layers):
        p = f"blk.{i}."
        w.tensor(p + "attn_norm.weight", [h], base.GGML_F32, unit(h))
        w.tensor(p + "post_attention_norm.weight", [h], base.GGML_F32, unit(h))
        if (i + 1) % args.full_attention_interval == 0:
            w.tensor(p + "attn_q.weight", [2 * heads * hd, h], base.GGML_F32,
                     base.f32_tensor(rng, 2 * heads * hd * h, 0.02))
            w.tensor(p + "attn_k.weight", [kv * hd, h], base.GGML_F32, base.f32_tensor(rng, kv * hd * h, 0.02))
            w.tensor(p + "attn_v.weight", [kv * hd, h], base.GGML_F32, base.f32_tensor(rng, kv * hd * h, 0.02))
            w.tensor(p + "attn_output.weight", [h, heads * hd], base.GGML_F32,
                     maybe_zero("attn_out", base.f32_tensor(rng, h * heads * hd, 0.02)))
            w.tensor(p + "attn_q_norm.weight", [hd], base.GGML_F32, unit(hd))
            w.tensor(p + "attn_k_norm.weight", [hd], base.GGML_F32, unit(hd))
        else:
            w.tensor(p + "attn_qkv.weight", [conv_dim, h], base.GGML_F32, base.f32_tensor(rng, conv_dim * h, 0.02))
            w.tensor(p + "attn_gate.weight", [value_dim, h], base.GGML_F32,
                     base.f32_tensor(rng, value_dim * h, 0.02))
            w.tensor(p + "ssm_beta.weight", [v_heads, h], base.GGML_F32, base.f32_tensor(rng, v_heads * h, 0.02))
            w.tensor(p + "ssm_alpha.weight", [v_heads, h], base.GGML_F32, base.f32_tensor(rng, v_heads * h, 0.02))
            w.tensor(p + "ssm_conv1d.weight", [conv_dim, args.conv_kernel], base.GGML_F32,
                     base.f32_tensor(rng, conv_dim * args.conv_kernel, 0.2))
            w.tensor(p + "ssm_dt.bias", [v_heads], base.GGML_F32, base.f32_tensor(rng, v_heads, 0.1))
            w.tensor(p + "ssm_a", [v_heads], base.GGML_F32,
                     f32_values([-rng.uniform(0.5, 1.5) for _ in range(v_heads)]))
            w.tensor(p + "ssm_norm.weight", [state], base.GGML_F32, unit(state))
            w.tensor(p + "ssm_out.weight", [h, value_dim], base.GGML_F32,
                     maybe_zero("gdn_out", base.f32_tensor(rng, h * value_dim, 0.02)))
        w.tensor(p + "ffn_gate_inp.weight", [e, h], base.GGML_F32, base.f32_tensor(rng, e * h, 0.5))
        quantised = EXPERT_TYPES[args.expert_type]
        for name, shape in [("gate", [e, inter, h]), ("up", [e, inter, h]), ("down", [e, h, inter])]:
            n = shape[0] * shape[1] * shape[2]
            if name not in quantised:
                w.tensor(p + f"ffn_{name}_exps.weight", shape, base.GGML_F32, base.f32_tensor(rng, n, 0.02))
            elif name == "down":
                w.tensor(p + "ffn_down_exps.weight", shape, base.GGML_Q3_K, base.q3k_blocks(rng, n // base.QK_K))
            else:
                w.tensor(p + f"ffn_{name}_exps.weight", shape, base.GGML_Q2_K, base.q2k_blocks(rng, n // base.QK_K))
        w.tensor(p + "ffn_gate_inp_shexp.weight", [h], base.GGML_F32, base.f32_tensor(rng, h, 0.02))
        w.tensor(p + "ffn_gate_shexp.weight", [shexp, h], base.GGML_F32, base.f32_tensor(rng, shexp * h, 0.02))
        w.tensor(p + "ffn_up_shexp.weight", [shexp, h], base.GGML_F32, base.f32_tensor(rng, shexp * h, 0.02))
        w.tensor(p + "ffn_down_shexp.weight", [h, shexp], base.GGML_F32,
                 maybe_zero("shexp_down", base.f32_tensor(rng, h * shexp, 0.02)))
    w.tensor("output_norm.weight", [h], base.GGML_F32, unit(h))
    w.tensor("output.weight", [vocab, h], base.GGML_F32, base.f32_tensor(rng, vocab * h, 0.5))

    out = Path(args.out)
    w.write(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"path": str(out), "sha256": digest, "architecture": ARCH, "layers": layers,
            "full_attention_layers": [i for i in range(layers) if (i + 1) % args.full_attention_interval == 0],
            "experts": e, "top_k": args.top_k, "hidden": h, "moe_inter": inter, "shexp_inter": shexp,
            "gdn": {"k_heads": k_heads, "v_heads": v_heads, "state_size": state, "conv_kernel": args.conv_kernel},
            "vocab": vocab, "seed": args.seed, "zeroed": sorted(zero), "expert_type": args.expert_type}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--rope-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--full-attention-interval", type=int, default=4)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--moe-inter", type=int, default=256)
    p.add_argument("--shexp-inter", type=int, default=256)
    p.add_argument("--k-heads", type=int, default=2)
    p.add_argument("--v-heads", type=int, default=4)
    p.add_argument("--state-size", type=int, default=64)
    p.add_argument("--conv-kernel", type=int, default=4)
    p.add_argument("--expert-type", choices=tuple(EXPERT_TYPES), default="kquant",
                   help="f32 removes quantisation noise when comparing engines with a float reference")
    p.add_argument("--zero", action="append", default=[], choices=ZERO_CHOICES,
                   help="diagnostic: write this output projection as zeros (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(build(parser().parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
