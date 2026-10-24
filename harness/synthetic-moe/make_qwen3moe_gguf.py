"""Write a tiny, fully synthetic Qwen3-MoE GGUF for engine smoke and parity tests.

No weights are downloaded and no real model is reproduced: every tensor comes from a
fixed seed. Attention, norms, router and embeddings are F32; expert stacks are random
but well-formed Q2_K (gate/up) and Q3_K (down) blocks, the quantisation of the
program's reference model. The tokenizer is a GPT-2 byte-level BPE with 256 byte
tokens and two specials, so every byte string tokenises losslessly.

The output is a functional fixture: its text is noise, its value is that the engine
loads it through the real GGUF path and runs the real MoE path end to end.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

ALIGN = 32
QK_K = 256
GGML_F32, GGML_Q2_K, GGML_Q3_K = 0, 10, 11
T_U32, T_F32, T_BOOL, T_STRING, T_ARRAY = 4, 6, 7, 8, 9
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def byte_tokens() -> list[str]:
    """GPT-2 bytes_to_unicode table, in byte order."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    mapping = {}
    extra = 0
    for b in range(256):
        if b in printable:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + extra)
            extra += 1
    return [mapping[b] for b in range(256)]


def f16_bits(value: float) -> bytes:
    return struct.pack("<e", value)


class Writer:
    def __init__(self) -> None:
        self.kv: list[bytes] = []
        self.infos: list[tuple[str, list[int], int, bytes]] = []

    @staticmethod
    def _str(s: str) -> bytes:
        raw = s.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def u32(self, key: str, v: int) -> None:
        self.kv.append(self._str(key) + struct.pack("<II", T_U32, v))

    def f32(self, key: str, v: float) -> None:
        self.kv.append(self._str(key) + struct.pack("<If", T_F32, v))

    def boolean(self, key: str, v: bool) -> None:
        self.kv.append(self._str(key) + struct.pack("<I?", T_BOOL, v))

    def string(self, key: str, v: str) -> None:
        self.kv.append(self._str(key) + struct.pack("<I", T_STRING) + self._str(v))

    def strings(self, key: str, values: list[str]) -> None:
        body = b"".join(self._str(v) for v in values)
        self.kv.append(self._str(key) + struct.pack("<IIQ", T_ARRAY, T_STRING, len(values)) + body)

    def i32s(self, key: str, values: list[int]) -> None:
        body = struct.pack(f"<{len(values)}i", *values)
        self.kv.append(self._str(key) + struct.pack("<IIQ", T_ARRAY, 5, len(values)) + body)

    def tensor(self, name: str, shape_rows_major: list[int], ggml_type: int, data: bytes) -> None:
        # GGUF stores dimensions fastest-first.
        self.infos.append((name, list(reversed(shape_rows_major)), ggml_type, data))

    def write(self, path: Path) -> None:
        out = bytearray(b"GGUF")
        out += struct.pack("<IQQ", 3, len(self.infos), len(self.kv))
        for kv in self.kv:
            out += kv
        offset = 0
        offsets = []
        for name, dims, ggml_type, data in self.infos:
            offsets.append(offset)
            out += self._str(name) + struct.pack("<I", len(dims))
            out += struct.pack(f"<{len(dims)}Q", *dims)
            out += struct.pack("<IQ", ggml_type, offset)
            offset += len(data)
            offset += (-offset) % ALIGN
        out += b"\0" * ((-len(out)) % ALIGN)
        for (_, _, _, data), start in zip(self.infos, offsets):
            assert len(out) % ALIGN == 0
            out += data
            out += b"\0" * ((-len(data)) % ALIGN)
        path.write_bytes(bytes(out))


def f32_tensor(rng: random.Random, n: int, std: float, const: float | None = None) -> bytes:
    if const is not None:
        return array.array("f", [const] * n).tobytes()
    return array.array("f", (rng.gauss(0.0, std) for _ in range(n))).tobytes()


def q2k_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += rng.randbytes(16) + rng.randbytes(64) + f16_bits(0.004) + f16_bits(0.002)
    return bytes(out)


def q3k_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += rng.randbytes(32) + rng.randbytes(64) + rng.randbytes(12) + f16_bits(0.0008)
    return bytes(out)


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    tokens = byte_tokens() + ["<|im_start|>", "<|im_end|>"]
    vocab = len(tokens)
    h, heads, kv, hd = args.hidden, args.heads, args.kv_heads, args.head_dim
    e, inter, layers = args.experts, args.moe_inter, args.layers
    if h % QK_K or inter % QK_K:
        raise SystemExit("hidden and moe_inter must be multiples of 256")

    w = Writer()
    arch = "qwen3moe"
    w.string("general.architecture", arch)
    w.string("general.name", "gnostral-synthetic-qwen3moe")
    for key, value in [
        ("context_length", 4096),
        ("embedding_length", h),
        ("block_count", layers),
        ("feed_forward_length", inter),
        ("attention.head_count", heads),
        ("attention.head_count_kv", kv),
        ("attention.key_length", hd),
        ("attention.value_length", hd),
        ("expert_count", e),
        ("expert_used_count", args.top_k),
        ("expert_feed_forward_length", inter),
        ("vocab_size", vocab),
    ]:
        w.u32(f"{arch}.{key}", value)
    w.f32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.f32(f"{arch}.rope.freq_base", 1_000_000.0)
    w.boolean(f"{arch}.expert_weights_norm", True)
    w.string("tokenizer.ggml.model", "gpt2")
    w.string("tokenizer.ggml.pre", "qwen2")
    w.strings("tokenizer.ggml.tokens", tokens)
    # GPT-2 byte-level tokens are NORMAL (1); BYTE (6) means SentencePiece "<0xXX>" tokens, which
    # llama.cpp parses and rejects here. CONTROL (3) for the two chat specials.
    w.i32s("tokenizer.ggml.token_type", [1] * 256 + [3, 3])
    w.strings("tokenizer.ggml.merges", [])
    w.u32("tokenizer.ggml.eos_token_id", vocab - 1)
    w.u32("tokenizer.ggml.padding_token_id", vocab - 1)
    w.boolean("tokenizer.ggml.add_bos_token", False)
    w.string("tokenizer.chat_template", CHAT_TEMPLATE)

    w.tensor("token_embd.weight", [vocab, h], GGML_F32, f32_tensor(rng, vocab * h, 0.5))
    for i in range(layers):
        p = f"blk.{i}."
        w.tensor(p + "attn_norm.weight", [h], GGML_F32, f32_tensor(rng, h, 0, 1.0))
        w.tensor(p + "attn_q.weight", [heads * hd, h], GGML_F32, f32_tensor(rng, heads * hd * h, 0.02))
        w.tensor(p + "attn_k.weight", [kv * hd, h], GGML_F32, f32_tensor(rng, kv * hd * h, 0.02))
        w.tensor(p + "attn_v.weight", [kv * hd, h], GGML_F32, f32_tensor(rng, kv * hd * h, 0.02))
        w.tensor(p + "attn_output.weight", [h, heads * hd], GGML_F32, f32_tensor(rng, h * heads * hd, 0.02))
        w.tensor(p + "attn_q_norm.weight", [hd], GGML_F32, f32_tensor(rng, hd, 0, 1.0))
        w.tensor(p + "attn_k_norm.weight", [hd], GGML_F32, f32_tensor(rng, hd, 0, 1.0))
        w.tensor(p + "ffn_norm.weight", [h], GGML_F32, f32_tensor(rng, h, 0, 1.0))
        w.tensor(p + "ffn_gate_inp.weight", [e, h], GGML_F32, f32_tensor(rng, e * h, 0.5))
        w.tensor(p + "ffn_gate_exps.weight", [e, inter, h], GGML_Q2_K, q2k_blocks(rng, e * inter * h // QK_K))
        w.tensor(p + "ffn_up_exps.weight", [e, inter, h], GGML_Q2_K, q2k_blocks(rng, e * inter * h // QK_K))
        w.tensor(p + "ffn_down_exps.weight", [e, h, inter], GGML_Q3_K, q3k_blocks(rng, e * h * inter // QK_K))
    w.tensor("output_norm.weight", [h], GGML_F32, f32_tensor(rng, h, 0, 1.0))
    w.tensor("output.weight", [vocab, h], GGML_F32, f32_tensor(rng, vocab * h, 0.5))

    out = Path(args.out)
    w.write(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"path": str(out), "sha256": digest, "layers": layers, "experts": e, "top_k": args.top_k,
            "hidden": h, "moe_inter": inter, "vocab": vocab, "seed": args.seed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--moe-inter", type=int, default=256)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
