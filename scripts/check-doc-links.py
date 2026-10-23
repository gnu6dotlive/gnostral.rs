#!/usr/bin/env python3
"""Static documentation gate for the active public entrypoints.

Checks local file links and documented status vocabulary. Does not claim
historical linked sources are complete or test live hosted pages.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

FILES = (
    "README.md", "QUESTLOG.md",
    "docs/START_HERE.md", "docs/CAPABILITY_MATRIX.md", "docs/ROADMAP.md",
    "docs/ARCHITECTURE.md", "docs/REPRODUCIBILITY.md", "docs/RELEASE.md",
    "docs/DEEPINFER_DONOR_ANALYSIS.md",
    "quests/Q-013-EVIDENCE-BOUND-ADMISSION.md",
    "quests/Q-013B-ISOLATED-FIXTURE.md", "quests/Q-013C-REAL-DENSE-PILOT.md",
    "quests/Q-017-DEEPINFER-DONOR-01.md", "quests/Q-025-ENGINE-DIRECTION.md",
)
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
FORBIDDEN_CLAIM = (
    ("README.md", "production-ready multi-model server.", False),  # appears as negation
)
REQUIRED = {
    "QUESTLOG.md": ("Q-010", "Q-011", "Q-012", "Q-013-A", "Q-013-B", "Q-013-C", "Q-013-D", "Q-017", "Q-025", "WakeKV"),
    "docs/RELEASE.md": ("v0.9.0-rc1", "38c4134"),
    "docs/DEEPINFER_DONOR_ANALYSIS.md": ("REFERENCE_ONLY", "Q-017"),
    "docs/CAPABILITY_MATRIX.md": ("NOT QUALIFIED", "FUNCTIONAL PASS", "Q013-C"),
    "docs/ROADMAP.md": ("Q-013D", "Q-014", "Q-015", "Q-016", "BURN-REMOTE-01"),
}
def validate(root: Path) -> list[str]:
    problems: list[str] = []
    for name in FILES:
        source = root / name
        if not source.is_file():
            problems.append(f"{name}: missing entrypoint")
            continue
        contents = source.read_text(encoding="utf-8")
        if "<<<<<<<" in contents or ">>>>>>>" in contents:
            problems.append(f"{name}: unresolved Git conflict marker")
        for term in REQUIRED.get(name, ()):
            if term not in contents:
                problems.append(f"{name}: missing status/index token {term}")
        for match in LINK.finditer(contents):
            target = match.group(1).strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0].split("?", 1)[0]
            if path.startswith("/"):
                problems.append(f"{name}: absolute filesystem link {path}")
                continue
            destination = (source.parent / path).resolve()
            if not destination.is_relative_to(root.resolve()):
                problems.append(f"{name}: link escapes repository {path}")
            elif not destination.exists():
                problems.append(f"{name}: broken local link {path}")
    return problems

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = validate(root)
    for problem in failures:
        print("DOCS_GATE_FAIL:", problem, file=sys.stderr)
    if failures:
        return 1
    print(f"PUBLIC_DOCS_LINKS_PASS {len(FILES)} entrypoints")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
