#!/usr/bin/env python3
"""Replace personal home-directory paths with `~` in evidence before it is committed.

The repository is public. Receipts, plans and captured stdout record the commands that produced them,
and those commands carry absolute paths such as /home/<name>/... . Run this on a new evidence
directory before `git add`:

    python3 scripts/scrub_home_paths.py evidence/runs/<new-run>/
    python3 scripts/scrub_home_paths.py --check evidence/runs/<new-run>/   # exit 1 if any remain

It rewrites text files in place: every `/home/<name>` or `/Users/<name>` prefix becomes `~`. Binary
files (a NUL byte in the first 8 KiB) are skipped. Paths outside home directories (/mnt/models, /tmp)
are left alone. Scrubbing does not invalidate a bench receipt: the protocol hash covers the
protocol's content (prompts, token budget, configuration, placement), not paths. A scrubbed plan is a
record, not a command line to re-run verbatim, since `~` is not expanded outside a shell.

`scripts/check-public.sh` runs the same check over every tracked file except the ones listed in
`scripts/home-path-allowlist.txt` (evidence committed before this rule existed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HOME_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?=/|\b)")
SNIFF_BYTES = 8192


def is_text(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" not in handle.read(SNIFF_BYTES)


def files_under(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out.extend(sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p])
    return out


def scrub(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    new, count = HOME_PATH.subn("~", text)
    if count:
        path.write_text(new, encoding="utf-8", errors="surrogateescape")
    return count


def leaks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    return sorted(set(HOME_PATH.findall(text)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report leaks and exit 1, change nothing")
    args = parser.parse_args(argv)
    missing = [p for p in args.paths if not p.exists()]
    if missing:
        sys.stderr.write(f"no such path: {', '.join(map(str, missing))}\n")
        return 2
    found, replaced = 0, 0
    for f in files_under(args.paths):
        if not is_text(f):
            continue
        if args.check:
            hits = leaks(f)
            if hits:
                found += 1
                print(f"{f}: {', '.join(hits)}")
        else:
            n = scrub(f)
            if n:
                replaced += n
                print(f"{f}: {n} replaced")
    if args.check:
        return 1 if found else 0
    print(f"{replaced} home path(s) replaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
