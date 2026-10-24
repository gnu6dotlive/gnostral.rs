"""End-to-end smoke of the engine program on a synthetic Qwen3-MoE GGUF (CPU or CUDA).

Wires the real pieces together, nothing mocked:

1. `make_qwen3moe_gguf.py` writes the fixture model (no download).
2. `gnostral_capture` (mistralrs example, patched engine) captures the reference lane
   (`GNOSTRAL_MOE_EXECUTOR=per-route`) on a frozen Q-030 prompt set.
3. `gnostral-bench run` (Q-031) drives the lanes `per-route`, `grouped` and `gnostral`
   with identity pins, 1 warmup and interleaved repetitions. Each lane captures tokens,
   checks greedy parity against the reference with `quality_oracle.py` (quality before
   timing) and prints `GNOSTRAL_PARITY` plus the engine's `GNOSTRAL_METRICS`.
4. Plan v2 identity checks (docs/FINDINGS.md, F-010 to F-014): every capture runs with
   the prefix cache off and, on CUDA, an explicit `--gpu-layers` map; the lane forwards
   the engine's `GNOSTRAL_PLACEMENT`, which the bench compares with the pinned
   `expected_placement`. The warmup capture becomes the lane's own reference
   (`GNOSTRAL_SELF_REPRO REFERENCE`); every measured capture must match it in tokens
   and logprob bits (`GNOSTRAL_SELF_REPRO PASS|FAIL`).

All outputs land in `--out`. Numbers from this fixture are not engine performance: the
model is random, tiny and CPU-only. The run proves plumbing, parity and receipts.

Usage:
  python3 harness/synthetic-moe/run_e2e.py --capture-bin PATH --bench-bin PATH --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness" / "lanes"))
import lanelib  # noqa: E402
from lanelib import logprobs_path, manifest_hash, run_sampled, sha256_file  # noqa: E402

ORACLE = lanelib.ORACLE
LLAMA_LANE = ROOT / "harness" / "lanes" / "llama_server_lane.py"
MAKER = Path(__file__).resolve().parent / "make_qwen3moe_gguf.py"
DEFAULT_MANIFEST = ROOT / "harness" / "quality" / "prompts" / "quality-prompts-32-20261023.json"
LANES = {
    "per-route": {"GNOSTRAL_MOE_EXECUTOR": "per-route"},
    "grouped": {"GNOSTRAL_MOE_EXECUTOR": "grouped"},
    "gnostral": {"GNOSTRAL_MOE_EXECUTOR": "grouped", "GNOSTRAL_CPU_EXPERT_KERNEL": "gnostral"},
}


# Reads this boot's kernel log for NVIDIA Xid faults; UNKNOWN when it cannot (F-016).
DRIVER_PROBE = ROOT / "harness" / "bench" / "driver_probe.py"


def residency_env(cfg: dict, lane: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "GNOSTRAL_EXPERT_RESIDENCY_CPU": "1",
        "GNOSTRAL_EXPERT_CACHE_BYTES": str(cfg["cache_bytes"]),
        "GNOSTRAL_EXPERT_PROFILE": str(cfg["profile"]),
    })
    env.update(LANES[lane])
    env.update(extra or {})
    return env


def model_path(args: argparse.Namespace, out: Path) -> Path:
    return Path(args.model).resolve() if args.model else out / "synth.gguf"


def gpu_layers(args: argparse.Namespace) -> int | None:
    """Layers mapped to CUDA: explicit, all layers by default; never the automatic map (F-011)."""
    if args.device != "cuda":
        return None
    return args.gpu_layers if args.gpu_layers is not None else args.layers


def expected_placement(args: argparse.Namespace) -> str:
    """The canonical placement the engine must report for this run (engine patch 0008)."""
    last = args.layers - 1
    if args.device == "cpu":
        return f"0-{last}:cpu;nm=cpu"
    g = gpu_layers(args)
    if g >= args.layers:
        return f"0-{last}:cuda:0;nm=cuda:0"
    if g <= 0:
        return f"0-{last}:cpu;nm=cuda:0"
    return f"0-{g - 1}:cuda:0,{g}-{last}:cpu;nm=cuda:0"


def capture_cmd(args: argparse.Namespace, out: Path, lane: str, dest: Path) -> list[str]:
    model = model_path(args, out)
    cmd = [
        args.capture_bin, "--model-dir", str(model.parent), "--file", model.name,
        "--manifest", str(args.manifest), "--promptset-sha256", manifest_hash(args.manifest),
        "--engine", lane, "--binary-sha256", sha256_file(Path(args.capture_bin)),
        "--model-sha256", sha256_file(model), "--max-tokens", str(args.max_tokens),
        "--out", str(dest), "--no-prefix-cache", "--topn", "2",
        "--logprobs-out", str(logprobs_path(dest)),
    ]
    if args.device == "cpu":
        cmd.append("--cpu")
    else:
        cmd += ["--gpu-layers", str(gpu_layers(args))]
    return cmd


def capture(args: argparse.Namespace, out: Path, cfg: dict, lane: str, dest: Path,
            extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(capture_cmd(args, out, lane, dest), env=residency_env(cfg, lane, extra),
                          capture_output=True, text=True)


def lane_main(args: argparse.Namespace) -> int:
    """One lane run for gnostral-bench: capture, parity vs reference, then metrics."""
    out = Path(args.out)
    if (out / "PARITY_FAIL").exists():
        sys.stderr.write("skipped: an earlier lane failed parity\n")
        return 5
    cfg = {"cache_bytes": args.cache_bytes, "profile": args.profile}
    latest = out / f"capture-{args.config}-{args.lane}-latest.json"
    run = run_sampled(capture_cmd(args, out, args.lane, latest), residency_env(cfg, args.lane),
                      sample=args.device == "cuda")
    sys.stderr.write(run["stderr"][-4000:])
    if run["returncode"] != 0:
        return run["returncode"]
    if args.device == "cuda":
        print(lanelib.device_line(run))
        if run["pid_seen"] is False:
            sys.stderr.write("CUDA not proven: the capture pid never appeared as an nvidia-smi compute app\n")
            return 3
    for line in run["stdout"].splitlines():
        if line.startswith("GNOSTRAL_PLACEMENT "):
            print(line)
    print(lanelib.self_repro_line(args.manifest, self_ref(out, args.config, args.lane), latest))
    parity = subprocess.run(
        [sys.executable, "-I", str(ORACLE), "parity", "--manifest", str(args.manifest),
         "--reference", str(out / f"capture-{args.config}-reference.json"),
         "--candidate", str(latest)],
        capture_output=True, text=True)
    print("GNOSTRAL_PARITY " + ("PASS" if parity.returncode == 0 else "FAIL"))
    if parity.returncode != 0:
        (out / "PARITY_FAIL").write_text(f"{args.config}/{args.lane}\n{parity.stdout}{parity.stderr}")
    for line in run["stdout"].splitlines():
        if line.startswith("GNOSTRAL_METRICS "):
            metrics = json.loads(line[len("GNOSTRAL_METRICS "):])
            if args.device == "cuda":
                metrics["vram_used_mib"] = run["peak_card_mib"]
            print("GNOSTRAL_METRICS " + json.dumps(metrics))
    return 0


def self_ref(out: Path, config: str, lane: str) -> Path:
    return out / f"self-ref-{config}-{lane}.json"


def expert_bytes(geometry: dict) -> int:
    """Bytes of every expert stack: Q2_K gate/up (84 B per 256) and Q3_K down (110 B per 256)."""
    per_expert = geometry["moe_inter"] * geometry["hidden"] * (2 * 84 + 110) // 256
    return per_expert * geometry["experts"] * geometry["layers"]


def summarise(out: Path, config: str, cfg: dict, bench: subprocess.CompletedProcess) -> dict:
    """Receipt block for one configuration; overlapping lane ranges mean inconclusive."""
    bundle_path = out / f"bench-{config}" / "bundle.json"
    lanes: dict[str, dict] = {}
    if bundle_path.exists():
        for run in json.loads(bundle_path.read_text()).get("runs", []):
            if run["phase"] != "measured":
                continue
            lane = lanes.setdefault(run["lane"], {"decode_tok_s": [], "parity": [], "device": [], "vram_mib": [],
                                                  "placement": [], "self_reproducibility": []})
            lane["parity"].append(run["parity"])
            lane["placement"].append(run.get("placement"))
            lane["self_reproducibility"].append(run.get("self_reproducibility"))
            for line in run["stdout"].splitlines():
                if line.startswith("GNOSTRAL_METRICS "):
                    m = json.loads(line[len("GNOSTRAL_METRICS "):])
                    lane["decode_tok_s"].append(m["decode_tok_s"])
                    lane["vram_mib"].append(m.get("vram_used_mib"))
                elif line.startswith("GNOSTRAL_DEVICE "):
                    lane["device"].append(line)
            if run["exit"] != "OK":
                lane.setdefault("failed", []).append(run["exit"])
    order = list(lanes)
    for name, lane in lanes.items():
        v = sorted(lane["decode_tok_s"])
        if v:
            lane.update(min=v[0], median=v[len(v) // 2], max=v[-1])
    base = lanes.get(order[0]) if order else None
    for name, lane in lanes.items():
        if "min" not in lane or base is None or "min" not in base or lane is base:
            lane["verdict_vs_baseline"] = "baseline" if lane is base else "UNKNOWN"
        elif lane["min"] > base["max"]:
            lane["verdict_vs_baseline"] = "faster"
        elif lane["max"] < base["min"]:
            lane["verdict_vs_baseline"] = "slower"
        else:
            lane["verdict_vs_baseline"] = "inconclusive (ranges overlap)"
    return {"config": config, "profile_sample": cfg.get("profile_sample"), "cache_bytes": cfg["cache_bytes"], "profile": str(cfg["profile"]),
            "bench_exit": bench.returncode, "baseline": order[0] if order else None, "lanes": lanes}


def llama_lanes(args: argparse.Namespace, out: Path, config: str, model: Path, model_sha: str) -> list[dict]:
    """The optional llama.cpp lane (Q-025 three-way): another engine, gated on reproducing itself."""
    if not args.llama_server:
        return []
    server = Path(args.llama_server).resolve()
    return [{
        "engine": "llama.cpp",
        "binary_path": str(server),
        "expected_binary_sha256": sha256_file(server),
        "model_path": str(model),
        "expected_model_sha256": model_sha,
        "host_fingerprint": f"{args.device}:{os.uname().machine}:{os.cpu_count()}",
        "argv": [sys.executable, "-I", str(LLAMA_LANE), "--server-bin", str(server), "--model", str(model),
                 "--manifest", str(args.manifest), "--out", str(out), "--max-tokens", str(args.max_tokens),
                 f"--server-args={args.llama_args}", "--engine", "llama.cpp", "--config", config,
                 "--device", args.device, "--reference", str(out / f"capture-{config}-reference.json"),
                 "--stop-file", str(out / "PARITY_FAIL")],
        "expected_placement": args.llama_placement,
        "parity_class": "self",
    }]


def run_config(args: argparse.Namespace, out: Path, config: str, cfg: dict, model: Path, model_sha: str) -> dict:
    for lane in [*LANES, *(["llama.cpp"] if args.llama_server else [])]:
        for stale in (self_ref(out, config, lane), logprobs_path(self_ref(out, config, lane))):
            stale.unlink(missing_ok=True)
    reference = capture(args, out, cfg, "per-route", out / f"capture-{config}-reference.json")
    if reference.returncode != 0:
        sys.stderr.write(reference.stderr[-4000:])
        raise SystemExit(reference.returncode)
    lane_args = ["--capture-bin", args.capture_bin, "--bench-bin", args.bench_bin, "--out", str(out),
                 "--manifest", str(args.manifest), "--max-tokens", str(args.max_tokens),
                 "--device", args.device, "--config", config, "--layers", str(args.layers),
                 "--cache-bytes", str(cfg["cache_bytes"]), "--profile", str(cfg["profile"])]
    if args.gpu_layers is not None:
        lane_args += ["--gpu-layers", str(args.gpu_layers)]
    if args.model:
        lane_args += ["--model", args.model]
    plan = {
        "schema": "gnostral.rs/engine-benchmark-plan/v2",
        "run_id": f"synthetic-moe-e2e-{config}",
        "date": "2026-10-23",
        "protocol": {
            "id": f"synthetic-moe-{args.device}-{config}",
            "sha256": hashlib.sha256(json.dumps({"manifest": manifest_hash(args.manifest),
                                                 "max_tokens": args.max_tokens, "config": config,
                                                 "cache_bytes": cfg["cache_bytes"], "prefix_cache": "off",
                                                 "placement": expected_placement(args)}).encode()).hexdigest(),
            "repetitions": args.repetitions,
            "context_tokens": 4096,
            "prompt_sha256": manifest_hash(args.manifest),
            "threads": os.cpu_count() or 1,
            "sampling": {"temperature": 0, "seed": 0, "n_predict": args.max_tokens, "repeat_penalty": 1},
        },
        "warmup_runs": 1,
        "driver_probe": [sys.executable, "-I", str(DRIVER_PROBE)],
        "lanes": [
            {
                "engine": lane,
                "binary_path": str(Path(args.capture_bin).resolve()),
                "expected_binary_sha256": sha256_file(Path(args.capture_bin)),
                "model_path": str(model),
                "expected_model_sha256": model_sha,
                "host_fingerprint": f"{args.device}:{os.uname().machine}:{os.cpu_count()}",
                "argv": [sys.executable, "-I", str(Path(__file__).resolve()), "--lane", lane, *lane_args],
                "expected_placement": expected_placement(args),
                "parity_class": "exact",
            }
            for lane in LANES
        ] + llama_lanes(args, out, config, model, model_sha),
    }
    plan_path = out / f"plan-{config}.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    bench = subprocess.run([args.bench_bin, "run", str(plan_path), "--out", str(out / f"bench-{config}")],
                           capture_output=True, text=True)
    print(f"=== {config} (cache_bytes={cfg['cache_bytes']}) ===")
    print(bench.stdout)
    sys.stderr.write(bench.stderr)
    return summarise(out, config, cfg, bench)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-bin", required=True)
    parser.add_argument("--bench-bin", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", help="existing GGUF to run instead of the synthetic fixture "
                        "(give --layers/--experts/--moe-inter/--hidden of that model)")
    parser.add_argument("--cpu-experts-only", action="store_true",
                        help="skip the recording pass and the mixed configuration")
    parser.add_argument("--mixed-only", action="store_true", help="run only the mixed configuration")
    parser.add_argument("--record-manifest", type=Path,
                        help="prompt set for the recording pass (default: the frozen set, which is in-sample)")
    parser.add_argument("--record-max-tokens", type=int, default=64)
    parser.add_argument("--mixed-bytes", type=int, help="explicit expert budget for the mixed configuration")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--moe-inter", type=int, default=256)
    parser.add_argument("--gpu-layers", type=int,
                        help="layers mapped to CUDA (default: all); the automatic device map is never used")
    parser.add_argument("--llama-server", help="llama-server executable: adds a llama.cpp lane (parity class self)")
    parser.add_argument("--llama-args", default="",
                        help='llama-server arguments in one string, with the = form: '
                             '--llama-args="-ngl 99 --cpu-moe -t 8 -c 4096 --fit off"')
    parser.add_argument("--llama-placement",
                        help='placement the llama.cpp lane must report, e.g. "llama.cpp offloaded=49/49 cpu_moe=on"')
    parser.add_argument("--mixed-fraction", type=float, default=0.5,
                        help="share of expert bytes given to the card in the mixed configuration")
    parser.add_argument("--lane", choices=sorted(LANES), help=argparse.SUPPRESS)
    parser.add_argument("--config", default="cpu-experts", help=argparse.SUPPRESS)
    parser.add_argument("--cache-bytes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.lane:
        return lane_main(args)
    if args.llama_server and not (args.llama_args and args.llama_placement):
        parser.error("--llama-server needs --llama-args and --llama-placement (the placement is pinned, not guessed)")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = model_path(args, out)
    summary_path = out / "synth.json"
    if args.model:
        summary_path.write_text(json.dumps({"path": str(model), "layers": args.layers, "experts": args.experts,
                                            "hidden": args.hidden, "moe_inter": args.moe_inter}))
    elif not model.exists() or not summary_path.exists():
        made = subprocess.run(
            [sys.executable, "-I", str(MAKER), "--out", str(model), "--hidden", str(args.hidden),
             "--layers", str(args.layers), "--experts", str(args.experts), "--moe-inter", str(args.moe_inter)],
            check=True, capture_output=True, text=True).stdout
        summary_path.write_text(made)
    geometry = json.loads(summary_path.read_text())
    model_sha = sha256_file(model)
    profile_header = {"schema": 1, "model_sha256": model_sha, "layers": geometry["layers"],
                      "experts_per_layer": geometry["experts"]}
    empty = out / "profile-empty.json"
    empty.write_text(json.dumps({**profile_header, "ranking": []}))

    # Configuration 1: every expert on the CPU (budget of one byte, empty ranking).
    cpu_cfg = {"cache_bytes": 1, "profile": empty}
    configs = [("cpu-experts", cpu_cfg)]
    if args.cpu_experts_only and not args.mixed_only:
        return finish(args, out, geometry, model, model_sha, configs)
    # Recording pass: count observed expert demand on the frozen workload.
    recorded = out / "profile-recorded.json"
    record_args = args
    if args.record_manifest:
        record_args = argparse.Namespace(**{**vars(args), "manifest": args.record_manifest,
                                            "max_tokens": args.record_max_tokens})
    record = capture(record_args, out, cpu_cfg, "per-route", out / "capture-record.json",
                     {"GNOSTRAL_EXPERT_PROFILE_OUT": str(recorded)})
    if record.returncode != 0 or not recorded.exists():
        sys.stderr.write(record.stderr[-4000:])
        return record.returncode or 4
    # Configuration 2: the recorded profile with a real budget.
    mixed_cfg = {"cache_bytes": args.mixed_bytes or int(expert_bytes(geometry) * args.mixed_fraction), "profile": recorded}

    mixed_cfg["profile_sample"] = "out-of-sample" if args.record_manifest else "in-sample"
    if args.mixed_only:
        configs = []
    return finish(args, out, geometry, model, model_sha, configs + [("mixed-experts", mixed_cfg)])


def finish(args: argparse.Namespace, out: Path, geometry: dict, model: Path, model_sha: str,
           configs: list) -> int:
    blocks = [run_config(args, out, name, cfg, model, model_sha) for name, cfg in configs]
    receipt = {
        "schema": "gnostral.rs/synthetic-moe-e2e-device/1",
        "device": args.device,
        "model": geometry,
        "expert_bytes_total": expert_bytes(geometry),
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "note": "synthetic noise model: proves plumbing, not engine performance",
        "configs": blocks,
    }
    (out / "e2e-receipt.json").write_text(json.dumps(receipt, indent=2))
    for block in blocks:
        print(block["config"], {n: (l.get("min"), l.get("median"), l.get("max"), l["verdict_vs_baseline"])
                                for n, l in block["lanes"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
