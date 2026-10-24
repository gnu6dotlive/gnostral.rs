"""What a GGUF file asks of the engine, read from its header only.

Reads the header of one or more GGUF files (shards of one model) without touching
the tensor data, so a 20 GB model is inventoried in seconds, and prints:

- the architecture, whether the engine's GGUF registry knows it and whether expert residency
  covers it, and every `<arch>.*` scalar hyperparameter;
- per tensor class (routed experts, shared experts, attention, linear-attention/SSM,
  embeddings and output, norms, other) the tensor count and bytes per GGML type;
- for every routed-expert projection, which engine path can run it;
- a verdict for the model.

The support sets below are copied from the engine sources named next to each one. They
describe this repository's engine tree; a lab build that differs must update them, and
the tool prints the sources it used so a reader can check.

Tensor sizes are computed from shape and type and checked against the offsets in the
header: overlapping tensors or a tensor past the end of its file is an error, not a guess.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

# (name, elements per block, bytes per block); mistralrs-quant/src/gguf/archive.rs `name`,
# `block_size`, `type_size`.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 142: ("PQ2_0", 128, 34), 143: ("PTQ1_0", 128, 28),
}

SOURCES = {
    "architectures": "vendor/mistralrs-strata/mistralrs-core/src/gguf/normal_registry.rs CanonicalGgufArchitecture",
    "engine_load": "vendor/mistralrs-strata/mistralrs-quant/src/gguf/archive.rs GgufDType::candle_dtype "
                   "(and weight_source.rs test unsupported_dtypes_are_rejected_at_construction)",
    "mxfp4_experts": "vendor/mistralrs-strata/mistralrs-core/src/gguf/normal_bindings.rs bind_block_tensor "
                     "(gpt-oss routed experts only)",
    "residency_architecture": "vendor/mistralrs-strata/mistralrs-core/src/pipeline/normal.rs residency pilot gate "
                              "(Qwen3MoeForCausalLM from a GGUF, no adapters, no ISQ)",
    "residency": "vendor/mistralrs-strata/mistralrs-core/src/moe/experts/residency.rs expert stack dtype check",
    "gnostral_kernel": "vendor/mistralrs-strata/mistralrs-core/src/moe/experts/residency.rs gnostral_host_ffn",
    "native_sizing": "patches/engine-program/0007 (Qwen3MoELoader::layer_sizes_in_bytes)",
}
ENGINE_ARCHS = {
    "llama", "mistral3", "gemma", "gemma2", "phi2", "phi3", "phimoe", "qwen2", "qwen3", "qwen3moe",
    "qwen3next", "qwen35", "qwen35moe", "starcoder2", "deepseek2", "glm4", "glm4moe", "smollm3", "granite",
    "granitemoe", "granitehybrid", "gpt-oss", "hunyuan-dense", "hunyuan-moe", "lfm2", "lfm2moe",
}
# Types `candle_dtype` maps; every other type fails the load. 142/143 are lab formats the public
# candle reconstruction refuses (BUILD-ONLY), so they are not counted as loadable here.
ENGINE_LOAD = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30}
# Extra types loadable for routed-expert tensors of one architecture, through a dedicated binding.
EXPERT_LOAD_BY_ARCH = {"gpt-oss": {39}}
RESIDENCY = {10, 11, 12, 13, 14, 2, 3, 6, 7, 8}
RESIDENCY_ARCHS = {"qwen3moe"}
GNOSTRAL_KERNEL = {10, 11}

VALUE_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
T_STRING, T_ARRAY = 8, 9
ARRAY_KEEP = 64
DEFAULT_ALIGN = 32
ROUTED = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.")
CLASSES = [
    ("routed_experts", re.compile(r"_exps\.")),
    ("shared_experts", re.compile(r"_shexp|shared_expert")),
    ("router", re.compile(r"\.ffn_gate_inp\.")),
    ("linear_attention_or_ssm", re.compile(r"\.ssm_|linear_attn|\.time_mix|\.delta")),
    ("norms", re.compile(r"norm")),
    ("attention", re.compile(r"\.attn_")),
    ("embeddings_and_output", re.compile(r"^(token_embd|output)\.")),
]


class InventoryError(RuntimeError):
    pass


class Header:
    """Sequential little-endian reads from a file handle."""

    def __init__(self, handle) -> None:
        self.h = handle
        self.pos = 0

    def take(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        raw = self.h.read(size)
        if len(raw) != size:
            raise InventoryError(f"header truncated at byte {self.pos}")
        self.pos += size
        values = struct.unpack("<" + fmt, raw)
        return values if len(values) > 1 else values[0]

    def string(self) -> str:
        n = self.take("Q")
        raw = self.h.read(n)
        if len(raw) != n:
            raise InventoryError(f"header truncated in a string at byte {self.pos}")
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def value(self, t: int):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et, n = self.take("IQ")
            kept = [self.value(et) for _ in range(min(n, ARRAY_KEEP))]
            for _ in range(n - len(kept)):
                self.value(et)
            return {"array_len": n, "head": kept} if n > ARRAY_KEEP else kept
        if t not in VALUE_FORMATS:
            raise InventoryError(f"unknown metadata value type {t} at byte {self.pos}")
        return self.take(VALUE_FORMATS[t])


def read_shard(path: Path) -> tuple[dict, list[dict]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        r = Header(handle)
        if r.take("4s") != b"GGUF":
            raise InventoryError(f"{path.name}: not a GGUF file")
        version = r.take("I")
        if version not in (2, 3):
            raise InventoryError(f"{path.name}: GGUF version {version} is not supported (2 or 3)")
        n_tensors, n_kv = r.take("QQ")
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.take("I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("Q") for _ in range(r.take("I"))]
            ggml_type, offset = r.take("IQ")
            tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
        align = kv.get("general.alignment", DEFAULT_ALIGN)
        data_start = r.pos + (-r.pos) % align
    for t in tensors:
        t["shard"] = path.name
        t["bytes"] = tensor_bytes(t)
    check_layout(path.name, tensors, size - data_start)
    return kv, tensors


def tensor_bytes(t: dict) -> int | None:
    spec = GGML_TYPES.get(t["type"])
    if spec is None:
        return None
    _, block, block_bytes = spec
    elements = 1
    for d in t["dims"]:
        elements *= d
    if t["dims"] and t["dims"][0] % block:
        raise InventoryError(f"{t['name']}: row of {t['dims'][0]} elements is not a multiple of {spec[0]} blocks")
    return elements // block * block_bytes


def check_layout(shard: str, tensors: list[dict], data_bytes: int) -> None:
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for cur, nxt in zip(ordered, ordered[1:] + [None]):
        if cur["bytes"] is None:
            continue
        end = cur["offset"] + cur["bytes"]
        limit = nxt["offset"] if nxt else data_bytes
        if end > limit:
            what = f"overlaps {nxt['name']}" if nxt else "runs past the end of the file"
            raise InventoryError(f"{shard}: {cur['name']} {what}")


def tensor_class(name: str) -> str:
    for label, pattern in CLASSES:
        if pattern.search(name):
            return label
    return "other"


def loadable(ggml_type: int, arch: str | None, routed_expert: bool) -> bool:
    return ggml_type in ENGINE_LOAD or (routed_expert and ggml_type in EXPERT_LOAD_BY_ARCH.get(arch, set()))


def expert_path(arch: str | None, gate: int | None, up: int | None, down: int | None) -> str:
    types = [t for t in (gate, up, down) if t is not None]
    if any(not loadable(t, arch, True) for t in types):
        return "refused_at_load"
    if arch not in RESIDENCY_ARCHS:
        return "loads_no_residency_for_architecture"
    if any(t not in RESIDENCY for t in types):
        return "loads_residency_refuses_types"
    if gate in GNOSTRAL_KERNEL and up == gate and down in GNOSTRAL_KERNEL:
        return "gnostral_kernel"
    return "residency_candle_kernel"


def inventory(paths: list[Path]) -> dict:
    kv, tensors = {}, []
    for path in paths:
        shard_kv, shard_tensors = read_shard(path)
        for key, value in shard_kv.items():
            kv.setdefault(key, value)
        tensors += shard_tensors
    arch = kv.get("general.architecture")
    by_class: dict = defaultdict(lambda: defaultdict(lambda: {"tensors": 0, "bytes": 0}))
    unknown_types = set()
    for t in tensors:
        name = GGML_TYPES.get(t["type"], (f"type{t['type']}",))[0]
        cell = by_class[tensor_class(t["name"])][name]
        cell["tensors"] += 1
        cell["bytes"] += t["bytes"] or 0
        if t["bytes"] is None:
            unknown_types.add(t["type"])
    layers: dict = defaultdict(dict)
    for t in tensors:
        m = ROUTED.match(t["name"])
        if m:
            layers[int(m.group(1))][m.group(2)] = t["type"]
    paths_by_layer = {}
    for layer, proj in sorted(layers.items()):
        if "gate_up" in proj:
            paths_by_layer[layer] = "fused_gate_up_unchecked"
        else:
            paths_by_layer[layer] = expert_path(arch, proj.get("gate"), proj.get("up"), proj.get("down"))
    verdicts = sorted(set(paths_by_layer.values()))
    load_refused = sorted({GGML_TYPES.get(t["type"], (f"type{t['type']}",))[0] for t in tensors
                           if not loadable(t["type"], arch, bool(ROUTED.match(t["name"])))})
    return {
        "schema": "gnostral.gguf-inventory/1",
        "files": [str(p) for p in paths],
        "architecture": arch,
        "name": kv.get("general.name"),
        "hyperparameters": {k: v for k, v in sorted(kv.items())
                            if arch and k.startswith(arch + ".") and not isinstance(v, (dict, list))},
        "tensors": len(tensors),
        "bytes_by_class": {c: dict(sorted(v.items())) for c, v in sorted(by_class.items())},
        "routed_expert_layers": len(paths_by_layer),
        "routed_expert_paths": {v: [l for l, p in paths_by_layer.items() if p == v] for v in verdicts},
        "types_refused_at_load": load_refused,
        "unknown_types": sorted(unknown_types),
        "architecture_known_to_engine": arch in ENGINE_ARCHS,
        "expert_residency_architecture": arch in RESIDENCY_ARCHS,
        "support_sources": SOURCES,
        "verdict": verdict(load_refused, verdicts, arch),
    }


def verdict(load_refused: list[str], paths: list[str], arch: str | None) -> str:
    if arch not in ENGINE_ARCHS:
        return f"UNKNOWN_ARCHITECTURE: the engine's native GGUF registry has no `{arch}`"
    if load_refused:
        return "REFUSED_AT_LOAD: this engine tree cannot load " + ", ".join(load_refused)
    if not paths:
        return "NO_ROUTED_EXPERTS: not a MoE layout this tool recognises"
    if paths == ["gnostral_kernel"]:
        return "GNOSTRAL_KERNEL: every expert layer runs on the Q-029 kernels"
    if paths == ["loads_no_residency_for_architecture"]:
        return (f"LOADS_WITHOUT_PROGRAM: `{arch}` loads, but expert residency, the Q-029 kernels and the 0007 "
                "sizing cover qwen3moe only")
    return "PARTIAL: expert paths " + ", ".join(paths)


def human(doc: dict) -> str:
    gib = 1 << 30
    lines = [f"architecture {doc['architecture']}  name {doc['name']}  tensors {doc['tensors']}"]
    for cls, types in doc["bytes_by_class"].items():
        parts = ", ".join(f"{t} {c['tensors']}x {c['bytes'] / gib:.2f} GiB" for t, c in types.items())
        lines.append(f"  {cls:24} {parts}")
    for path, layers in doc["routed_expert_paths"].items():
        lines.append(f"  experts {path}: {len(layers)} layers")
    lines.append(f"verdict {doc['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", nargs="+", help="GGUF file, or every shard of a split model")
    parser.add_argument("--json", action="store_true", help="print the full JSON document")
    args = parser.parse_args(argv)
    try:
        doc = inventory([Path(p) for p in args.gguf])
    except (InventoryError, OSError) as error:
        sys.stderr.write(f"gguf inventory: {error}\n")
        return 2
    print(json.dumps(doc, indent=2) if args.json else human(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
