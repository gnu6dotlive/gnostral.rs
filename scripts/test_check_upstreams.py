"""Tests for the upstreams manifest gate (scripts/check-upstreams.py)."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check-upstreams.py")
spec = importlib.util.spec_from_file_location("check_upstreams", SCRIPT)
gate = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(gate)

INVENTORY = {"schema": 1, "state": "PATCHED_UPSTREAM", "patch": {"path": "p.patch", "sha256": "0" * 64}}


def write_project(root: Path, body: str) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "upstreams.toml").write_text(body)


class TestUpstreamsGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_reference_only_is_accepted(self):
        write_project(
            self.root,
            'schema = 1\n[[project]]\nname = "x"\nrepository = "https://example/x"\n'
            'license = "MIT"\nstate = "REFERENCE_ONLY"\n',
        )
        self.assertEqual(gate.validate(self.root), [])

    def test_patched_requires_base_revision(self):
        write_project(
            self.root,
            'schema = 1\n[[project]]\nname = "x"\nrepository = "https://example/x"\n'
            'license = "MIT"\nstate = "PATCHED_UPSTREAM"\nrevision = "a"\nsource_path = "vendor/x"\n'
            'inventory = "i.json"\n',
        )
        problems = gate.validate(self.root)
        self.assertTrue(any("base_revision" in p for p in problems), problems)

    def test_inventory_state_must_agree(self):
        (self.root / "i.json").write_text(json.dumps({"state": "REFERENCE_ONLY"}))
        write_project(
            self.root,
            'schema = 1\n[[project]]\nname = "x"\nrepository = "https://example/x"\n'
            'license = "MIT"\nstate = "PINNED_UPSTREAM"\nrevision = "a"\nsource_path = "vendor/x"\n'
            'inventory = "i.json"\n',
        )
        problems = gate.validate(self.root)
        self.assertTrue(any("inventory state" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
