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
import re
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
    (args.out / f"{name}-{run}.stdout.txt").write_text(done.stdout[-20000:])
    if done.returncode != 0:
        sys.exit(f"capture {name}-{run} failed ({done.returncode}); see {name}-{run}.stderr.txt")
    return dest


def mapping(args: argparse.Namespace, name: str, run: int) -> list[str]:
    """Layer-to-device lines the engine logged at load (the automatic device map)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", (args.out / f"{name}-{run}.stdout.txt").read_text())
    return [m.group(0) for m in re.finditer(r"Layers \d+-\d+: \w+(?:\[\d+\])?", text)]


def diff(args: argparse.Namespace, name: str) -> list[dict]:
    def load(run: int) -> tuple[dict, dict]:
        toks = {c["prompt_id"]: c["token_ids"]
                for c in json.loads((args.out / f"{name}-{run}.json").read_text())["captures"]}
        steps = {d["prompt_id"]: d["steps"]
                 for d in json.loads((args.out / f"{name}-{run}.logprobs.json").read_text())}
        return toks, steps
    (one, lp1), (two, lp2) = load(1), load(2)
    maps = [mapping(args, name, 1), mapping(args, name, 2)]
    rows = []
    for pid in args.only:
        a, b = one[pid], two[pid]
        n = min(len(a), len(b))
        tok_idx = next((i for i in range(n) if a[i] != b[i]), None)
        if tok_idx is None and len(a) != len(b):
            tok_idx = n
        chosen1 = [st[0][1] for st in lp1[pid]]
        chosen2 = [st[0][1] for st in lp2[pid]]
        lp_idx = next((i for i in range(n) if chosen1[i] != chosen2[i]), None)
        end = tok_idx if tok_idx is not None else n
        max_diff = max((abs(chosen1[i] - chosen2[i]) for i in range(min(end, n))), default=0.0)

        def top_gap(steps: list, i: int) -> float | None:
            return steps[i][0][1] - steps[i][1][1] if 0 <= i < len(steps) and len(steps[i]) > 1 else None

        first = min((i for i in (tok_idx, lp_idx) if i is not None), default=None)
        prev = None if first is None else first - 1
        at_div = None if tok_idx is None else tok_idx
        rows.append({
            "config": name, "prompt_id": pid,
            "first_logprob_diff_idx": lp_idx, "first_token_diff_idx": tok_idx,
            "gap_prev_run1": None if prev is None else top_gap(lp1[pid], prev),
            "gap_prev_run2": None if prev is None else top_gap(lp2[pid], prev),
            "gap_at_div_run1": None if at_div is None else top_gap(lp1[pid], at_div),
            "gap_at_div_run2": None if at_div is None else top_gap(lp2[pid], at_div),
            "max_logprob_diff_before_div": max_diff,
            "device_map_run1": maps[0], "device_map_run2": maps[1],
        })
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
    fmt = lambda v: "-" if v is None else (f"{v:.2e}" if isinstance(v, float) else str(v))
    lines = ["| config | prompt | lp idx | tok idx | gap prev r1 | gap prev r2 | gap@div r1 | gap@div r2 | max lp diff |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| " + " | ".join([r["config"], r["prompt_id"], fmt(r["first_logprob_diff_idx"]),
                     fmt(r["first_token_diff_idx"]), fmt(r["gap_prev_run1"]), fmt(r["gap_prev_run2"]),
                     fmt(r["gap_at_div_run1"]), fmt(r["gap_at_div_run2"]),
                     fmt(r["max_logprob_diff_before_div"])]) + " |")
    for n in names:
        mine = [r for r in rows if r["config"] == n]
        lines.append(f"\n{n}: tokens differ {sum(r['first_token_diff_idx'] is not None for r in mine)}/{len(mine)}, "
                     f"logprobs differ {sum(r['first_logprob_diff_idx'] is not None for r in mine)}/{len(mine)}, "
                     f"device map run1 {mine[0]['device_map_run1']} run2 {mine[0]['device_map_run2']}")
    (args.out / "repro.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
