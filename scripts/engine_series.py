#!/usr/bin/env python3
"""Tell how far the engine-program patch series is applied to an engine tree.

The engine tree (vendor/mistralrs-strata) is never pushed, and the lab's copy may not be a git
repository, so "which patches does this tree carry?" is answered from file contents:

    python3 scripts/engine_series.py check --tree vendor/mistralrs-strata
    # ENGINE_SERIES_STATE 0010 (all 17 files match)        exit 0
    # ENGINE_SERIES_STATE NONE + the files that differ       exit 1

`patches/engine-program/SERIES-STATES.json` holds, for every file the series touches, its SHA-256
after the baseline and after each patch. `check` reports the last state the whole set matches;
when none matches it names the closest state and its differing files (a missing patch, a local
edit, or a different baseline). Cargo.lock is not part of the series: cargo rewrites it.

`record` rebuilds the states file from a baseline tree by applying the series with `git apply` in a
scratch directory holding only the touched files (run it whenever a patch is added):

    python3 scripts/engine_series.py record --baseline-tree <tree at the baseline>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCH_DIR = REPO / "patches" / "engine-program"
STATES = PATCH_DIR / "SERIES-STATES.json"
PATCH_NAME = re.compile(r"^\d{4}-.*\.patch$")
TARGET = re.compile(r"^\+\+\+ b/(\S+)")


def series(patch_dir: Path) -> list[Path]:
    return sorted(p for p in patch_dir.iterdir() if PATCH_NAME.match(p.name))


def touched(patches: list[Path]) -> list[str]:
    files = set()
    for p in patches:
        for line in p.read_text(errors="replace").splitlines():
            m = TARGET.match(line)
            if m:
                files.add(m.group(1))
    return sorted(files)


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def snapshot(root: Path, files: list[str]) -> dict[str, str | None]:
    return {f: digest(root / f) for f in files}


def record(baseline: Path, out: Path, patch_dir: Path = PATCH_DIR) -> int:
    patches = series(patch_dir)
    files = touched(patches)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in files:
            if (baseline / f).is_file():
                (work / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline / f, work / f)
        states = [{"after": "baseline", "files": snapshot(work, files)}]
        for p in patches:
            run = subprocess.run(["git", "apply", "--whitespace=nowarn", str(p)], cwd=work,
                                 capture_output=True, text=True)
            if run.returncode:
                sys.stderr.write(f"{p.name} does not apply after the previous patches:\n{run.stderr}")
                return 1
            states.append({"after": p.name[:4], "patch": p.name, "patch_sha256": digest(p),
                           "files": snapshot(work, files)})
    doc = {"schema": "gnostral.engine-series-states/1",
           "baseline": "public mistral.rs 24dbf5c + patches/strata-residency/20261023-mistralrs-strata-integration.patch "
                       "(evidence/runs/mistralrs-strata-vendor-20261023.json)",
           "files": files, "states": states}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"recorded {len(states) - 1} patches over {len(files)} files -> {out}")
    return 0


def check(tree: Path, states_path: Path, patch_dir: Path = PATCH_DIR) -> int:
    doc = json.loads(states_path.read_text())
    files = doc["files"]
    have = snapshot(tree, files)
    stale = [s["patch"] for s in doc["states"][1:] if s["patch_sha256"] != digest(patch_dir / s["patch"])]
    if stale:
        print(f"WARNING: states file is stale for {', '.join(stale)}; re-run record", file=sys.stderr)
    scored = []
    for s in doc["states"]:
        diff = [f for f in files if s["files"][f] != have[f]]
        if not diff:
            print(f"ENGINE_SERIES_STATE {s["after"].upper()} (all {len(files)} files match)")
            later = [x["patch"] for x in doc["states"] if x["after"] > s["after"] and x["after"] != "baseline"]
            if s["after"] != "baseline" and later:
                print("not applied yet: " + ", ".join(later))
            return 0
        scored.append((len(diff), s["after"], diff))
    n, after, diff = min(scored)
    print(f"ENGINE_SERIES_STATE NONE: closest is {after} with {n} differing file(s):")
    for f in diff:
        print(f"  {f}: {'missing' if have[f] is None else 'differs'}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tree", type=Path, required=True)
    c.add_argument("--states", type=Path, default=STATES)
    c.add_argument("--patches", type=Path, default=PATCH_DIR)
    r = sub.add_parser("record")
    r.add_argument("--baseline-tree", type=Path, required=True)
    r.add_argument("--out", type=Path, default=STATES)
    r.add_argument("--patches", type=Path, default=PATCH_DIR)
    args = parser.parse_args(argv)
    if args.cmd == "record":
        return record(args.baseline_tree, args.out, args.patches)
    if not args.tree.is_dir():
        print(f"no such tree: {args.tree}", file=sys.stderr)
        return 2
    return check(args.tree, args.states, args.patches)


if __name__ == "__main__":
    sys.exit(main())
