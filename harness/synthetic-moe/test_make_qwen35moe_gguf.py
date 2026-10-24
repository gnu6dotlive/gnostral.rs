"""The synthetic qwen35moe GGUF holds the tensor and metadata contract written in its generator."""

from __future__ import annotations

import array
import tempfile
import unittest
from pathlib import Path

import make_qwen35moe_gguf as maker
from test_make_qwen3moe_gguf import parse

ARCH = maker.ARCH


def build(tmp: str, **overrides) -> tuple[dict, dict, dict]:
    args = maker.parser().parse_args(["--out", str(Path(tmp) / "m.gguf")])
    for key, value in overrides.items():
        setattr(args, key, value)
    info = maker.build(args)
    kv, tensors = parse(Path(info["path"]))
    return info, kv, {t[0]: t for t in tensors}


def rows_major(entry) -> list[int]:
    return list(reversed(entry[1]))


class Qwen35MoeFixtureTests(unittest.TestCase):
    def test_metadata_both_consumers_require(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info, kv, _ = build(tmp)
        self.assertEqual(kv["general.architecture"], ARCH)
        for key in ["expert_count", "expert_used_count", "expert_feed_forward_length",
                    "expert_shared_feed_forward_length", "full_attention_interval", "rope.dimension_count",
                    "ssm.conv_kernel", "ssm.inner_size", "ssm.state_size", "ssm.group_count",
                    "ssm.time_step_rank", "attention.key_length"]:
            self.assertIn(f"{ARCH}.{key}", kv, key)
        self.assertEqual(len(kv[f"{ARCH}.rope.dimension_sections"]), 4)
        self.assertEqual(sum(kv[f"{ARCH}.rope.dimension_sections"]) * 2, kv[f"{ARCH}.rope.dimension_count"])
        # llama.cpp: head_v_dim = state_size; engine: head_v_dim = inner_size / v_heads
        self.assertEqual(kv[f"{ARCH}.ssm.inner_size"], kv[f"{ARCH}.ssm.state_size"] * kv[f"{ARCH}.ssm.time_step_rank"])
        self.assertEqual(info["full_attention_layers"], [3])

    def test_layer_kinds_and_shapes_follow_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info, kv, t = build(tmp)
        h, hd, heads, kv_heads = info["hidden"], 256, 2, 1
        g = info["gdn"]
        key_dim, value_dim = g["state_size"] * g["k_heads"], g["state_size"] * g["v_heads"]
        conv_dim = 2 * key_dim + value_dim
        for i in range(info["layers"]):
            p = f"blk.{i}."
            full = i in info["full_attention_layers"]
            self.assertEqual(p + "attn_q.weight" in t, full, i)
            self.assertEqual(p + "attn_qkv.weight" in t, not full, i)
            if full:
                self.assertEqual(rows_major(t[p + "attn_q.weight"]), [2 * heads * hd, h])
                self.assertEqual(rows_major(t[p + "attn_k.weight"]), [kv_heads * hd, h])
                self.assertEqual(rows_major(t[p + "attn_output.weight"]), [h, heads * hd])
                self.assertEqual(rows_major(t[p + "attn_q_norm.weight"]), [hd])
            else:
                self.assertEqual(rows_major(t[p + "attn_qkv.weight"]), [conv_dim, h])
                self.assertEqual(rows_major(t[p + "attn_gate.weight"]), [value_dim, h])
                self.assertEqual(rows_major(t[p + "ssm_beta.weight"]), [g["v_heads"], h])
                self.assertEqual(rows_major(t[p + "ssm_alpha.weight"]), [g["v_heads"], h])
                self.assertEqual(rows_major(t[p + "ssm_conv1d.weight"]), [conv_dim, g["conv_kernel"]])
                self.assertEqual(rows_major(t[p + "ssm_a"]), [g["v_heads"]])
                self.assertEqual(rows_major(t[p + "ssm_norm.weight"]), [g["state_size"]])
                self.assertEqual(rows_major(t[p + "ssm_out.weight"]), [h, value_dim])
            self.assertEqual(rows_major(t[p + "ffn_gate_inp_shexp.weight"]), [h])
            self.assertEqual(rows_major(t[p + "ffn_down_shexp.weight"]), [h, info["shexp_inter"]])
            self.assertEqual(t[p + "ffn_gate_exps.weight"][2], maker.base.GGML_Q2_K)
            self.assertEqual(t[p + "ffn_down_exps.weight"][2], maker.base.GGML_Q3_K)

    def test_ssm_a_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info, _, t = build(tmp)
            data = Path(info["path"]).read_bytes()
            for i in range(info["layers"]):
                if i in info["full_attention_layers"]:
                    continue
                name, dims, _, offset, start = t[f"blk.{i}.ssm_a"]
                values = array.array("f", data[start + offset:start + offset + 4 * dims[0]])
                self.assertTrue(all(v < 0 for v in values), (i, list(values)))

    def test_same_seed_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(build(a)[0]["sha256"], build(b)[0]["sha256"])

    def test_inconsistent_shapes_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for overrides in [{"v_heads": 3}, {"layers": 3}, {"rope_dim": 48}, {"hidden": 300}]:
                with self.assertRaises(SystemExit, msg=str(overrides)):
                    build(tmp, **overrides)


if __name__ == "__main__":
    unittest.main()
