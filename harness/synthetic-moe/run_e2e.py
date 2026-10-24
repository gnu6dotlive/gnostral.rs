"""End-to-end smoke of the engine program on a synthetic Qwen3-MoE GGUF (CPU or CUDA).

Wires the real pieces together, nothing mocked:

1. `make_qwen3moe_gguf.py` writes the fixture model (no download).
2. `gnostral_capture` (mistralrs example, patched engine) captures the reference lane
   (`GNOSTRAL_MOE_EXECUTOR=per-route`) on a frozen Q-030 prompt set.
3. `gnostral-bench run` (Q-031) drives the lanes `per-route`, `grouped` and `gnostral`
   with identity pins, 1 warmup and interleaved repetitions. Each lane captures tokens,
   checks greedy parity against the reference with `quality_oracle.py` (quality before
   timing) and prints `GNOSTRAL_PARITY` plus the engine's `GNOSTRAL_METRICS`.

All outputs land in `--out`. Numbers from this fixture are not engine performance: the
model is random, tiny and CPU-only. The run proves plumbing, parity and receipts.

Usage:
  python3 harness/synthetic-moe/run_e2e.py --capture-bin PATH --bench-bin PATH --out DIR
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "harness" / "quality" / "quality_oracle.py"
MAKER = Path(__file__).resolve().parent / "make_qwen3moe_gguf.py"
DEFAULT_MANIFEST = ROOT / "harness" / "quality" / "prompts" / "quality-prompts-32-20261023.json"
LANES = {
    "per-route": {"GNOSTRAL_MOE_EXECUTOR": "per-route"},
    "grouped": {"GNOSTRAL_MOE_EXECUTOR": "grouped"},
    "gnostral": {"GNOSTRAL_MOE_EXECUTOR": "grouped", "GNOSTRAL_CPU_EXPERT_KERNEL": "gnostral"},
}


# Kernel-log scan for NVIDIA Xid faults since boot; the bench refuses a GPU comparison otherwise.
DRIVER_PROBE = ("if journalctl -k -b --no-pager 2>/dev/null | grep -q 'NVRM: Xid'; "
                "then echo GNOSTRAL_DRIVER XID; else echo GNOSTRAL_DRIVER NO_XID; fi")


@functools.lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(manifest: Path) -> str:
    out = subprocess.run([sys.executable, "-I", str(ORACLE), "manifest-hash", str(manifest)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["manifest_sha256"]


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


def capture_cmd(args: argparse.Namespace, out: Path, lane: str, dest: Path) -> list[str]:
    model = model_path(args, out)
    cmd = [
        args.capture_bin, "--model-dir", str(model.parent), "--file", model.name,
        "--manifest", str(args.manifest), "--promptset-sha256", manifest_hash(args.manifest),
        "--engine", lane, "--binary-sha256", sha256_file(Path(args.capture_bin)),
        "--model-sha256", sha256_file(model), "--max-tokens", str(args.max_tokens),
        "--out", str(dest),
    ]
    if args.device == "cpu":
        cmd.append("--cpu")
    return cmd


def capture(args: argparse.Namespace, out: Path, cfg: dict, lane: str, dest: Path,
            extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(capture_cmd(args, out, lane, dest), env=residency_env(cfg, lane, extra),
                          capture_output=True, text=True)


def smi(*query: str) -> list[str] | None:
    """Rows of an nvidia-smi CSV query, or None when nvidia-smi is absent or fails."""
    try:
        run = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return run.stdout.strip().splitlines() if run.returncode == 0 else None


def card_used_mib() -> float | None:
    rows = smi("--query-gpu=memory.used")
    try:
        return float(rows[0]) if rows else None
    except ValueError:
        return None


def process_used_mib(pid: int) -> tuple[bool | None, float | None]:
    """(pid listed as a compute app, its VRAM); (None, None) when nvidia-smi cannot say."""
    rows = smi("--query-compute-apps=pid,used_memory")
    if rows is None:
        return None, None
    for row in rows:
        fields = [f.strip() for f in row.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return True, float(fields[1])
            except ValueError:
                return True, None
    return False, None


def run_sampled(cmd: list[str], env: dict[str, str]) -> dict:
    """Run a capture while sampling the card every 200 ms; no number is invented."""
    ambient = card_used_mib()
    with tempfile.TemporaryFile("w+") as out_f, tempfile.TemporaryFile("w+") as err_f:
        proc = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=err_f, text=True)
        peak_card: float | None = None
        peak_proc: float | None = None
        seen: bool | None = None
        while True:
            done = proc.poll() is not None
            card = card_used_mib()
            if card is not None:
                peak_card = card if peak_card is None else max(peak_card, card)
            listed, used = process_used_mib(proc.pid)
            if listed is not None:
                seen = bool(seen) or listed
            if used is not None:
                peak_proc = used if peak_proc is None else max(peak_proc, used)
            if done:
                break
            time.sleep(0.2)
        out_f.seek(0)
        err_f.seek(0)
        return {"returncode": proc.returncode, "stdout": out_f.read(), "stderr": err_f.read(),
                "ambient_mib": ambient, "peak_card_mib": peak_card, "pid_seen": seen,
                "peak_process_mib": peak_proc}


def fmt(v: float | bool | None) -> str:
    return "UNKNOWN" if v is None else str(round(v, 1) if isinstance(v, float) else v).lower()


def lane_main(args: argparse.Namespace) -> int:
    """One lane run for gnostral-bench: capture, parity vs reference, then metrics."""
    out = Path(args.out)
    cfg = {"cache_bytes": args.cache_bytes, "profile": args.profile}
    latest = out / f"capture-{args.config}-{args.lane}-latest.json"
    run = run_sampled(capture_cmd(args, out, args.lane, latest), residency_env(cfg, args.lane))
    sys.stderr.write(run["stderr"][-4000:])
    if run["returncode"] != 0:
        return run["returncode"]
    if args.device == "cuda":
        print(f"GNOSTRAL_DEVICE cuda pid_seen={fmt(run['pid_seen'])} "
              f"peak_process_mib={fmt(run['peak_process_mib'])} "
              f"peak_card_mib={fmt(run['peak_card_mib'])} ambient_mib={fmt(run['ambient_mib'])}")
        if run["pid_seen"] is False:
            sys.stderr.write("CUDA not proven: the capture pid never appeared as an nvidia-smi compute app\n")
            return 3
    parity = subprocess.run(
        [sys.executable, "-I", str(ORACLE), "parity", "--manifest", str(args.manifest),
         "--reference", str(out / f"capture-{args.config}-reference.json"),
         "--candidate", str(latest)],
        capture_output=True, text=True)
    print("GNOSTRAL_PARITY " + ("PASS" if parity.returncode == 0 else "FAIL"))
    for line in run["stdout"].splitlines():
        if line.startswith("GNOSTRAL_METRICS "):
            metrics = json.loads(line[len("GNOSTRAL_METRICS "):])
            if args.device == "cuda":
                metrics["vram_used_mib"] = run["peak_card_mib"]
            print("GNOSTRAL_METRICS " + json.dumps(metrics))
    return 0


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
            lane = lanes.setdefault(run["lane"], {"decode_tok_s": [], "parity": [], "device": [], "vram_mib": []})
            lane["parity"].append(run["parity"])
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
    return {"config": config, "cache_bytes": cfg["cache_bytes"], "profile": str(cfg["profile"]),
            "bench_exit": bench.returncode, "baseline": order[0] if order else None, "lanes": lanes}


def run_config(args: argparse.Namespace, out: Path, config: str, cfg: dict, model: Path, model_sha: str) -> dict:
    reference = capture(args, out, cfg, "per-route", out / f"capture-{config}-reference.json")
    if reference.returncode != 0:
        sys.stderr.write(reference.stderr[-4000:])
        raise SystemExit(reference.returncode)
    lane_args = ["--capture-bin", args.capture_bin, "--bench-bin", args.bench_bin, "--out", str(out),
                 "--manifest", str(args.manifest), "--max-tokens", str(args.max_tokens),
                 "--device", args.device, "--config", config,
                 "--cache-bytes", str(cfg["cache_bytes"]), "--profile", str(cfg["profile"])]
    plan = {
        "schema": "gnostral.rs/engine-benchmark-plan/v1",
        "run_id": f"synthetic-moe-e2e-{config}",
        "date": "2026-10-23",
        "protocol": {
            "id": f"synthetic-moe-{args.device}-{config}",
            "sha256": hashlib.sha256(json.dumps({"manifest": manifest_hash(args.manifest),
                                                 "max_tokens": args.max_tokens, "config": config,
                                                 "cache_bytes": cfg["cache_bytes"]}).encode()).hexdigest(),
            "repetitions": args.repetitions,
            "context_tokens": 4096,
            "prompt_sha256": manifest_hash(args.manifest),
            "threads": os.cpu_count() or 1,
            "sampling": {"temperature": 0, "seed": 0, "n_predict": args.max_tokens, "repeat_penalty": 1},
        },
        "warmup_runs": 1,
        "driver_probe": ["/bin/sh", "-c", DRIVER_PROBE],
        "lanes": [
            {
                "engine": lane,
                "binary_path": str(Path(args.capture_bin).resolve()),
                "expected_binary_sha256": sha256_file(Path(args.capture_bin)),
                "model_path": str(model),
                "expected_model_sha256": model_sha,
                "host_fingerprint": f"{args.device}:{os.uname().machine}:{os.cpu_count()}",
                "argv": [sys.executable, "-I", str(Path(__file__).resolve()), "--lane", lane, *lane_args],
            }
            for lane in LANES
        ],
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
    parser.add_argument("--mixed-bytes", type=int, help="explicit expert budget for the mixed configuration")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--moe-inter", type=int, default=256)
    parser.add_argument("--mixed-fraction", type=float, default=0.5,
                        help="share of expert bytes given to the card in the mixed configuration")
    parser.add_argument("--lane", choices=sorted(LANES), help=argparse.SUPPRESS)
    parser.add_argument("--config", default="cpu-experts", help=argparse.SUPPRESS)
    parser.add_argument("--cache-bytes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.lane:
        return lane_main(args)

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
    # Recording pass: count observed expert demand on the frozen workload.
    recorded = out / "profile-recorded.json"
    record = capture(args, out, cpu_cfg, "per-route", out / "capture-record.json",
                     {"GNOSTRAL_EXPERT_PROFILE_OUT": str(recorded)})
    if record.returncode != 0 or not recorded.exists():
        sys.stderr.write(record.stderr[-4000:])
        return record.returncode or 4
    # Configuration 2: the recorded profile with a real budget.
    mixed_cfg = {"cache_bytes": args.mixed_bytes or int(expert_bytes(geometry) * args.mixed_fraction), "profile": recorded}

    blocks = [run_config(args, out, name, cfg, model, model_sha)
              for name, cfg in (("cpu-experts", cpu_cfg), ("mixed-experts", mixed_cfg))]
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
