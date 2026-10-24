"""End-to-end CPU smoke of the engine program on a synthetic Qwen3-MoE GGUF.

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
import hashlib
import json
import os
import subprocess
import sys
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


def residency_env(out: Path, lane: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "GNOSTRAL_EXPERT_RESIDENCY_CPU": "1",
        "GNOSTRAL_EXPERT_CACHE_BYTES": "1",
        "GNOSTRAL_EXPERT_PROFILE": str(out / "profile.json"),
    })
    env.update(LANES[lane])
    return env


def capture(args: argparse.Namespace, out: Path, lane: str, dest: Path) -> subprocess.CompletedProcess:
    model = out / "synth.gguf"
    cmd = [
        args.capture_bin, "--model-dir", str(out), "--file", model.name,
        "--manifest", str(args.manifest), "--promptset-sha256", manifest_hash(args.manifest),
        "--engine", lane, "--binary-sha256", sha256_file(Path(args.capture_bin)),
        "--model-sha256", sha256_file(model), "--max-tokens", str(args.max_tokens),
        "--out", str(dest),
    ]
    if args.device == "cpu":
        cmd.append("--cpu")
    return subprocess.run(cmd, env=residency_env(out, lane), capture_output=True, text=True)


def lane_main(args: argparse.Namespace) -> int:
    """One lane run for gnostral-bench: capture, parity vs reference, then metrics."""
    out = Path(args.out)
    run = capture(args, out, args.lane, out / f"capture-{args.lane}-latest.json")
    sys.stderr.write(run.stderr[-4000:])
    if run.returncode != 0:
        return run.returncode
    parity = subprocess.run(
        [sys.executable, "-I", str(ORACLE), "parity", "--manifest", str(args.manifest),
         "--reference", str(out / "capture-reference.json"),
         "--candidate", str(out / f"capture-{args.lane}-latest.json")],
        capture_output=True, text=True)
    print("GNOSTRAL_PARITY " + ("PASS" if parity.returncode == 0 else "FAIL"))
    for line in run.stdout.splitlines():
        if line.startswith("GNOSTRAL_METRICS "):
            print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-bin", required=True)
    parser.add_argument("--bench-bin", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lane", choices=sorted(LANES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.lane:
        return lane_main(args)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = out / "synth.gguf"
    if not model.exists():
        subprocess.run([sys.executable, "-I", str(MAKER), "--out", str(model)], check=True)
    model_sha = sha256_file(model)
    (out / "profile.json").write_text(json.dumps({
        "schema": 1, "model_sha256": model_sha, "layers": 2, "experts_per_layer": 16, "ranking": [],
    }))

    reference = capture(args, out, "per-route", out / "capture-reference.json")
    if reference.returncode != 0:
        sys.stderr.write(reference.stderr[-4000:])
        return reference.returncode

    plan = {
        "schema": "gnostral.rs/engine-benchmark-plan/v1",
        "run_id": "synthetic-moe-e2e",
        "date": "2026-10-23",
        "protocol": {
            "id": f"synthetic-moe-{args.device}-smoke",
            "sha256": hashlib.sha256(json.dumps({"manifest": manifest_hash(args.manifest),
                                                 "max_tokens": args.max_tokens}).encode()).hexdigest(),
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
                "argv": [sys.executable, "-I", str(Path(__file__).resolve()), "--lane", lane,
                         "--capture-bin", args.capture_bin, "--bench-bin", args.bench_bin,
                         "--out", str(out), "--manifest", str(args.manifest),
                         "--max-tokens", str(args.max_tokens), "--device", args.device],
            }
            for lane in LANES
        ],
    }
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    bench = subprocess.run([args.bench_bin, "run", str(plan_path), "--out", str(out / "bench")],
                           capture_output=True, text=True)
    print(bench.stdout)
    sys.stderr.write(bench.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
