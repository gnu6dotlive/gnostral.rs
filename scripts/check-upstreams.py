#!/usr/bin/env python3
"""Validate manifests/upstreams.toml against docs/VENDORING_POLICY.md.

Checks the manifest shape and its agreement with the referenced evidence
inventories. It never requires the ignored vendor tree to be present; a fresh
clone has the manifest, the inventories and the exported patches, which is
exactly what a reviewer can verify offline.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

STATES = {"PINNED_UPSTREAM", "PATCHED_UPSTREAM", "REFERENCE_ONLY"}
REQUIRED_ALWAYS = ("name", "repository", "state")
# A vendored/forked tree must record its license; a reference-only study may not have one.
REQUIRED_BY_STATE_LICENSE = {"PINNED_UPSTREAM", "PATCHED_UPSTREAM"}
REQUIRED_BY_STATE = {
    "PINNED_UPSTREAM": ("revision", "source_path", "inventory"),
    "PATCHED_UPSTREAM": ("revision", "base_revision", "source_path", "inventory", "patches"),
    "REFERENCE_ONLY": (),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> list[str]:
    problems: list[str] = []
    manifest = root / "manifests" / "upstreams.toml"
    if not manifest.is_file():
        return [f"{manifest.relative_to(root)}: missing"]
    with manifest.open("rb") as handle:
        document = tomllib.load(handle)
    if document.get("schema") != 1:
        problems.append("manifests/upstreams.toml: schema must be 1")
    projects = document.get("project", [])
    if not isinstance(projects, list) or not projects:
        return problems + ["manifests/upstreams.toml: no [[project]] entries"]

    names: set[str] = set()
    for entry in projects:
        name = entry.get("name", "<unnamed>")
        where = f"manifests/upstreams.toml [{name}]"
        if name in names:
            problems.append(f"{where}: duplicate project name")
        names.add(name)
        state = entry.get("state")
        if state not in STATES:
            problems.append(f"{where}: state {state!r} not in {sorted(STATES)}")
            continue
        required = REQUIRED_ALWAYS + REQUIRED_BY_STATE[state]
        if state in REQUIRED_BY_STATE_LICENSE:
            required += ("license",)
        for field in required:
            if not entry.get(field):
                problems.append(f"{where}: missing required field {field!r} for {state}")
        if state == "PINNED_UPSTREAM" and (entry.get("patches") or entry.get("modified") is True):
            problems.append(f"{where}: PINNED_UPSTREAM must have patches=[] and modified=false")
        if state == "PATCHED_UPSTREAM":
            if entry.get("modified") is not True:
                problems.append(f"{where}: PATCHED_UPSTREAM must set modified=true")
            for patch in entry.get("patches", []):
                patch_path = root / patch
                if not patch_path.is_file():
                    problems.append(f"{where}: patch {patch!r} does not exist")
        inventory_ref = entry.get("inventory")
        if not inventory_ref:
            continue
        inventory_path = root / inventory_ref
        if not inventory_path.is_file():
            problems.append(f"{where}: inventory {inventory_ref!r} does not exist")
            continue
        try:
            inventory = json.loads(inventory_path.read_text())
        except json.JSONDecodeError as error:
            problems.append(f"{where}: inventory is not JSON: {error}")
            continue
        if inventory.get("state") != state:
            problems.append(
                f"{where}: inventory state {inventory.get('state')!r} != manifest state {state!r}"
            )
        if state == "PATCHED_UPSTREAM":
            patch = inventory.get("patch") or {}
            patch_path = root / patch.get("path", "")
            if patch_path.is_file() and patch.get("sha256"):
                if sha256(patch_path) != patch["sha256"]:
                    problems.append(f"{where}: patch digest does not match inventory")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems = validate(root)
    for problem in problems:
        print(f"UPSTREAMS_GATE_FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("UPSTREAMS_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
