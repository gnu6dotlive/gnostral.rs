"""The synthetic GGUF parses back with an independent minimal reader."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import make_qwen3moe_gguf as maker

TYPE_SIZES = {maker.GGML_F32: (1, 4), maker.GGML_Q2_K: (256, 84), maker.GGML_Q3_K: (256, 110)}
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, data: bytes) -> None:
        self.d, self.o = data, 0

    def take(self, fmt: str):
        values = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += struct.calcsize("<" + fmt)
        return values if len(values) > 1 else values[0]

    def string(self) -> str:
        n = self.take("Q")
        s = self.d[self.o:self.o + n].decode("utf-8")
        self.o += n
        return s

    def value(self, t: int):
        if t == maker.T_STRING:
            return self.string()
        if t == maker.T_ARRAY:
            et, n = self.take("IQ")
            return [self.value(et) for _ in range(n)]
        fmt = {4: "I", 5: "i", 6: "f", 7: "?"}[t]
        return self.take(fmt)


def parse(path: Path) -> tuple[dict, list]:
    r = Reader(path.read_bytes())
    assert r.take("4s") == b"GGUF"
    version, n_tensors, n_kv = r.take("IQQ")
    assert version == 3
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.take("I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("Q") for _ in range(r.take("I"))]
        ggml_type, offset = r.take("IQ")
        infos.append((name, dims, ggml_type, offset))
    data_start = r.o + (-r.o) % maker.ALIGN
    return kv, [(n, d, t, o, data_start) for n, d, t, o in infos]


class SyntheticGgufTests(unittest.TestCase):
    def test_small_model_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.gguf"
            args = maker.argparse.Namespace(out=str(out), seed=1, hidden=256, heads=2, kv_heads=1,
                                            head_dim=128, layers=1, experts=2, top_k=1, moe_inter=256)
            info = maker.build(args)
            kv, tensors = parse(out)
            self.assertEqual(kv["general.architecture"], "qwen3moe")
            self.assertEqual(kv["qwen3moe.expert_count"], 2)
            self.assertEqual(len(kv["tokenizer.ggml.tokens"]), info["vocab"])
            self.assertEqual(len(set(kv["tokenizer.ggml.tokens"])), info["vocab"])
            names = [t[0] for t in tensors]
            self.assertIn("blk.0.ffn_gate_exps.weight", names)
            size = out.stat().st_size
            for name, dims, ggml_type, offset, start in tensors:
                self.assertEqual(offset % maker.ALIGN, 0, name)
                block, block_bytes = TYPE_SIZES[ggml_type]
                elements = 1
                for d in dims:
                    elements *= d
                self.assertEqual(elements % block, 0, name)
                self.assertLessEqual(start + offset + elements // block * block_bytes, size, name)
            exps = dict((t[0], t) for t in tensors)["blk.0.ffn_down_exps.weight"]
            self.assertEqual(exps[1], [256, 256, 2])  # fastest-first: inter, hidden, experts
            self.assertEqual(exps[2], maker.GGML_Q3_K)

    def test_same_seed_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for i in range(2):
                args = maker.argparse.Namespace(out=str(Path(tmp) / f"{i}.gguf"), seed=7, hidden=256, heads=2,
                                                kv_heads=1, head_dim=128, layers=1, experts=2, top_k=1, moe_inter=256)
                hashes.append(maker.build(args)["sha256"])
            self.assertEqual(hashes[0], hashes[1])

    def test_unaligned_shapes_are_refused(self) -> None:
        args = maker.argparse.Namespace(out="/dev/null", seed=1, hidden=300, heads=2, kv_heads=1,
                                        head_dim=128, layers=1, experts=2, top_k=1, moe_inter=256)
        with self.assertRaises(SystemExit):
            maker.build(args)


if __name__ == "__main__":
    unittest.main()
