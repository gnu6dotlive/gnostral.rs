"""The inventory reads real GGUF headers and refuses inconsistent ones."""

from __future__ import annotations

import argparse
import io
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic-moe"))
import gguf_inventory as inv  # noqa: E402
import make_qwen3moe_gguf as maker  # noqa: E402

Q4_K, IQ4_XS, MXFP4 = 12, 23, 39


def synthetic(path: Path, layers: int = 2) -> dict:
    args = argparse.Namespace(out=str(path), seed=3, hidden=256, heads=2, kv_heads=1, head_dim=128,
                              layers=layers, experts=4, top_k=2, moe_inter=256)
    return maker.build(args)


def expert_model(path: Path, arch: str, gate_type: int, down_type: int, hidden: int = 256, inter: int = 256,
                 experts: int = 2) -> None:
    w = maker.Writer()
    w.string("general.architecture", arch)
    w.u32(f"{arch}.expert_count", experts)
    w.strings("tokenizer.ggml.tokens", [f"t{i}" for i in range(100)])
    w.tensor("token_embd.weight", [8, hidden], maker.GGML_F32, bytes(8 * hidden * 4))
    for name, ggml_type, shape in [("gate", gate_type, [experts, inter, hidden]),
                                   ("up", gate_type, [experts, inter, hidden]),
                                   ("down", down_type, [experts, hidden, inter])]:
        _, block, block_bytes = inv.GGML_TYPES[ggml_type]
        n = shape[0] * shape[1] * shape[2] // block * block_bytes
        w.tensor(f"blk.0.ffn_{name}_exps.weight", shape, ggml_type, bytes(n))
    w.write(path)


def raw_gguf(path: Path, tensors: list[tuple[str, list[int], int, int]], data_bytes: int) -> None:
    """Hand-written header so offsets can be made inconsistent on purpose."""
    def s(x: str) -> bytes:
        return struct.pack("<Q", len(x)) + x.encode()
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    out += s("general.architecture") + struct.pack("<I", 8) + s("qwen3moe")
    for name, dims, ggml_type, offset in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", ggml_type, offset)
    out += b"\0" * ((-len(out)) % 32) + bytes(data_bytes)
    path.write_bytes(bytes(out))


class InventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_synthetic_fixture_runs_on_the_gnostral_kernel(self) -> None:
        info = synthetic(self.tmp / "m.gguf")
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertEqual(doc["architecture"], "qwen3moe")
        self.assertTrue(doc["verdict"].startswith("GNOSTRAL_KERNEL"), doc["verdict"])
        self.assertTrue(doc["architecture_known_to_engine"])
        self.assertTrue(doc["expert_residency_architecture"])
        self.assertEqual(doc["routed_expert_paths"], {"gnostral_kernel": [0, 1]})
        self.assertEqual(doc["hyperparameters"]["qwen3moe.expert_count"], 4)
        e, h, i = info["experts"], info["hidden"], info["moe_inter"]
        experts = doc["bytes_by_class"]["routed_experts"]
        self.assertEqual(experts["Q2_K"], {"tensors": 4, "bytes": 4 * e * i * h // 256 * 84})
        self.assertEqual(experts["Q3_K"], {"tensors": 2, "bytes": 2 * e * h * i // 256 * 110})
        self.assertEqual(doc["types_refused_at_load"], [])
        # per layer: q, k, v, output; attn, q, k and ffn norms plus output_norm; one router per layer
        self.assertEqual(doc["bytes_by_class"]["attention"]["F32"]["tensors"], 4 * 2)
        self.assertEqual(doc["bytes_by_class"]["norms"]["F32"]["tensors"], 4 * 2 + 1)
        self.assertEqual(doc["bytes_by_class"]["router"]["F32"]["tensors"], 2)
        self.assertNotIn("other", doc["bytes_by_class"])

    def test_tensor_classes_follow_llama_cpp_names(self) -> None:
        for name, cls in [("blk.3.ssm_norm.weight", "linear_attention_or_ssm"),
                          ("blk.3.attn_norm.weight", "norms"), ("blk.3.attn_qkv.weight", "attention"),
                          ("blk.3.ffn_up_shexp.weight", "shared_experts"), ("output.weight", "embeddings_and_output")]:
            self.assertEqual(inv.tensor_class(name), cls, name)
        self.assertEqual(inv.tensor_class("blk.0.attn_qkv.weight", "qwen35moe"), "linear_attention_or_ssm")
        self.assertEqual(inv.tensor_class("blk.0.attn_gate.weight", "qwen3next"), "linear_attention_or_ssm")
        self.assertEqual(inv.tensor_class("blk.0.attn_qkv.weight", "phi3"), "attention")
        self.assertEqual(inv.tensor_class("blk.3.attn_q.weight", "qwen35moe"), "attention")

    def test_iq4_xs_experts_are_refused_at_load(self) -> None:
        expert_model(self.tmp / "m.gguf", "qwen35moe", IQ4_XS, IQ4_XS)
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertTrue(doc["verdict"].startswith("REFUSED_AT_LOAD"), doc["verdict"])
        self.assertEqual(doc["types_refused_at_load"], ["IQ4_XS"])
        self.assertEqual(doc["routed_expert_paths"], {"refused_at_load": [0]})
        self.assertTrue(doc["architecture_known_to_engine"])
        self.assertTrue(doc["expert_residency_architecture"])

    def test_q4_k_experts_take_the_candle_kernel(self) -> None:
        expert_model(self.tmp / "m.gguf", "qwen3moe", Q4_K, Q4_K)
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertEqual(doc["routed_expert_paths"], {"residency_candle_kernel": [0]})
        self.assertTrue(doc["verdict"].startswith("PARTIAL"), doc["verdict"])

    def test_qwen35moe_takes_residency_since_0011(self) -> None:
        expert_model(self.tmp / "m.gguf", "qwen35moe", Q4_K, Q4_K)
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertEqual(doc["routed_expert_paths"], {"residency_candle_kernel": [0]})
        self.assertTrue(doc["verdict"].startswith("PARTIAL"), doc["verdict"])

    def test_lfm2moe_loads_without_the_program(self) -> None:
        expert_model(self.tmp / "m.gguf", "lfm2moe", Q4_K, Q4_K)
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertEqual(doc["routed_expert_paths"], {"loads_no_residency_for_architecture": [0]})
        self.assertTrue(doc["verdict"].startswith("LOADS_WITHOUT_PROGRAM"), doc["verdict"])

    def test_mxfp4_experts_load_only_for_gpt_oss(self) -> None:
        expert_model(self.tmp / "g.gguf", "gpt-oss", MXFP4, MXFP4)
        doc = inv.inventory([self.tmp / "g.gguf"])
        self.assertEqual(doc["types_refused_at_load"], [])
        self.assertTrue(doc["verdict"].startswith("LOADS_WITHOUT_PROGRAM"), doc["verdict"])
        expert_model(self.tmp / "q.gguf", "qwen3moe", MXFP4, MXFP4)
        doc = inv.inventory([self.tmp / "q.gguf"])
        self.assertEqual(doc["types_refused_at_load"], ["MXFP4"])
        self.assertEqual(doc["routed_expert_paths"], {"refused_at_load": [0]})

    def test_unknown_architecture_is_named(self) -> None:
        expert_model(self.tmp / "m.gguf", "mystery", 10, 11)
        doc = inv.inventory([self.tmp / "m.gguf"])
        self.assertFalse(doc["architecture_known_to_engine"])
        self.assertEqual(doc["verdict"], "UNKNOWN_ARCHITECTURE: the engine's native GGUF registry has no `mystery`")

    def test_large_arrays_are_counted_not_kept(self) -> None:
        expert_model(self.tmp / "m.gguf", "qwen3moe", 10, 11)
        kv, tensors = inv.read_shard(self.tmp / "m.gguf")
        self.assertEqual(kv["tokenizer.ggml.tokens"]["array_len"], 100)
        self.assertEqual(len(kv["tokenizer.ggml.tokens"]["head"]), inv.ARRAY_KEEP)
        self.assertEqual(len(tensors), 4)

    def test_overlapping_tensors_are_an_error(self) -> None:
        q2k_row = 84  # one Q2_K block: 256 elements
        raw_gguf(self.tmp / "m.gguf", [("a.weight", [256, 2], 10, 0), ("b.weight", [256], 10, q2k_row)],
                 data_bytes=4 * q2k_row)
        with self.assertRaisesRegex(inv.InventoryError, "a.weight overlaps b.weight"):
            inv.inventory([self.tmp / "m.gguf"])

    def test_tensor_past_the_end_is_an_error(self) -> None:
        raw_gguf(self.tmp / "m.gguf", [("a.weight", [256, 4], 10, 0)], data_bytes=84)
        with self.assertRaisesRegex(inv.InventoryError, "past the end"):
            inv.inventory([self.tmp / "m.gguf"])

    def test_truncated_header_exits_2(self) -> None:
        synthetic(self.tmp / "m.gguf")
        (self.tmp / "cut.gguf").write_bytes((self.tmp / "m.gguf").read_bytes()[:200])
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(inv.main([str(self.tmp / "cut.gguf")]), 2)
        self.assertIn("truncated", err.getvalue())

    def test_shards_are_inventoried_together(self) -> None:
        expert_model(self.tmp / "a.gguf", "qwen3moe", 10, 11)
        expert_model(self.tmp / "b.gguf", "qwen3moe", 10, 11)
        doc = inv.inventory([self.tmp / "a.gguf", self.tmp / "b.gguf"])
        self.assertEqual(doc["tensors"], 8)
        self.assertEqual(doc["bytes_by_class"]["routed_experts"]["Q2_K"]["tensors"], 4)


if __name__ == "__main__":
    unittest.main()
