"""Does a lane reproduce itself? Capture the same prompts twice per configuration and diff.

Runs `gnostral_capture` (per-route lane, all experts on the CPU) twice for each configuration
with the top-2 logprobs recorded, then prints one row per (configuration, prompt): the first
token index where the two runs differ and the logprob gap between the two candidate tokens.
A tiny gap (~1e-3 or less) means a numerical near-tie; a large gap means a real defect.

Configurations:
  A  CUDA as is
  B  --cpu (everything on the processor)
  C  CUDA + CUBLAS_WORKSPACE_CONFIG=:4096:8 + prefix cache off   (C1 / C2 split the two knobs)
  D  CUDA + RAYON_NUM_THREADS=1

Usage:
  python3 harness/diag/repro_capture.py --capture-bin BIN --model GGUF --profile PROFILE.json \
      --out DIR [--configs A,B,C,D]
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
MANIFEST = ROOT / "harness" / "quality" / "prompts" / "quality-prompts-32-20261023.json"
DIVERGENT = ["q04-units", "q07-sql", "q09-bug", "q10-fr-explain", "q11-fr-translate",
             "q19-summary-tech", "q27-science", "q29-code-explain", "q32-refusal-boundary"]
CUBLAS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
CONFIGS = {
    "A": {"flags": [], "env": {}},
    "B": {"flags": ["--cpu"], "env": {}},
    "C": {"flags": ["--no-prefix-cache"], "env": CUBLAS},
    "C1": {"flags": [], "env": CUBLAS},
    "C2": {"flags": ["--no-prefix-cache"], "env": {}},
    "D": {"flags": [], "env": {"RAYON_NUM_THREADS": "1"}},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(args: argparse.Namespace, name: str, run: int, binary_sha: str, model_sha: str,
            promptset: str) -> Path:
    cfg = CONFIGS[name]
    dest = args.out / f"{name}-{run}.json"
    cmd = [
        args.capture_bin, "--model-dir", str(Path(args.model).parent), "--file", Path(args.model).name,
        "--manifest", str(MANIFEST), "--promptset-sha256", promptset, "--engine", "per-route",
        "--binary-sha256", binary_sha, "--model-sha256", model_sha, "--max-tokens", str(args.max_tokens),
        "--out", str(dest), "--only", ",".join(args.only), "--topn", "2",
        "--logprobs-out", str(args.out / f"{name}-{run}.logprobs.json"), *cfg["flags"],
    ]
    env = dict(os.environ)
    env.update({
        "GNOSTRAL_EXPERT_RESIDENCY_CPU": "1", "GNOSTRAL_EXPERT_CACHE_BYTES": "1",
        "GNOSTRAL_EXPERT_PROFILE": str(args.profile), "GNOSTRAL_MOE_EXECUTOR": "per-route",
    })
    env.update(cfg["env"])
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (args.out / f"{name}-{run}.stderr.txt").write_text(done.stderr[-8000:])
    if done.returncode != 0:
        sys.exit(f"capture {name}-{run} failed ({done.returncode}); see {name}-{run}.stderr.txt")
    return dest


def diff(args: argparse.Namespace, name: str) -> list[dict]:
    one = {c["prompt_id"]: c["token_ids"] for c in json.loads((args.out / f"{name}-1.json").read_text())["captures"]}
    two = {c["prompt_id"]: c["token_ids"] for c in json.loads((args.out / f"{name}-2.json").read_text())["captures"]}
    lp1 = {d["prompt_id"]: d["steps"] for d in json.loads((args.out / f"{name}-1.logprobs.json").read_text())}
    lp2 = {d["prompt_id"]: d["steps"] for d in json.loads((args.out / f"{name}-2.logprobs.json").read_text())}
    rows = []
    for pid in args.only:
        a, b = one[pid], two[pid]
        idx = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if idx is None and len(a) != len(b):
            idx = min(len(a), len(b))
        gap = None
        if idx is not None and idx < min(len(a), len(b)):
            def gap_in(steps: list, hi: int, lo: int) -> float | None:
                table = {t: lp for t, lp in steps[idx]}
                return table[hi] - table[lo] if hi in table and lo in table else None
            gaps = [g for g in (gap_in(lp1[pid], a[idx], b[idx]), gap_in(lp2[pid], b[idx], a[idx])) if g is not None]
            gap = min(gaps) if gaps else None
        rows.append({"config": name, "prompt_id": pid, "divergence_index": idx, "logprob_gap": gap})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", default="A,B,C,D")
    parser.add_argument("--only", default=",".join(DIVERGENT))
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    args.only = args.only.split(",")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    binary_sha = sha256_file(Path(args.capture_bin))
    model_sha = sha256_file(Path(args.model))
    promptset = json.loads(subprocess.run([sys.executable, "-I", str(ORACLE), "manifest-hash", str(MANIFEST)],
                                          check=True, capture_output=True, text=True).stdout)["manifest_sha256"]
    rows: list[dict] = []
    for name in args.configs.split(","):
        for run in (1, 2):
            capture(args, name, run, binary_sha, model_sha, promptset)
            print(f"captured {name}-{run}", flush=True)
        rows += diff(args, name)
    (args.out / "repro.json").write_text(json.dumps({"binary_sha256": binary_sha, "model_sha256": model_sha,
                                                     "max_tokens": args.max_tokens, "rows": rows}, indent=2))
    names = args.configs.split(",")
    print("| prompt | " + " | ".join(names) + " |")
    print("| --- |" + " --- |" * len(names))
    for pid in args.only:
        cells = []
        for name in names:
            r = next(x for x in rows if x["config"] == name and x["prompt_id"] == pid)
            cells.append("=" if r["divergence_index"] is None else
                         f"{r['divergence_index']} (gap {r['logprob_gap']:.2e})" if r["logprob_gap"] is not None
                         else f"{r['divergence_index']} (gap n/a)")
        print(f"| {pid} | " + " | ".join(cells) + " |")
    print("| diverged | " + " | ".join(
        f"{sum(1 for r in rows if r['config'] == n and r['divergence_index'] is not None)}/{len(args.only)}"
        for n in names) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
