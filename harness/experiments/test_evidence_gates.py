"""The audit gates must not disappear under python -O (PYTHONOPTIMIZE).

`assert` is removed by the optimizer; a receipt must never be written by a gate
that silently vanished. These tests execute the *optimized* interpreter and
scan the audit scripts for reintroduced bare asserts.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GATE_FILES = (
    "q010_hotpath_ab_audit_20261010.py",
    "q010_hotpath_ab_retest_20261010.py",
    "q012_capability_audit.py",
    "q013c_real_dense.py",
)
PROBE = (
    "import sys\n"
    f"sys.path.insert(0, {str(HERE)!r})\n"
    "from gates import demand, EvidenceError\n"
    "try:\n"
    "    demand(False, 'must fail closed')\n"
    "except EvidenceError:\n"
    "    sys.exit(0)\n"
    "sys.exit(3)\n"
)


class TestFailClosedGates(unittest.TestCase):
    def test_demand_raises_in_normal_mode(self):
        proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_demand_raises_under_python_optimize(self):
        proc = subprocess.run([sys.executable, "-O", "-c", PROBE], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_audit_scripts_have_no_bare_asserts(self):
        pattern = re.compile(r"^\s*assert\b", re.MULTILINE)
        for name in GATE_FILES:
            source = (HERE / name).read_text()
            matches = pattern.findall(source)
            self.assertEqual(matches, [], f"{name} reintroduced bare assert(s)")
            self.assertIn("from gates import demand", source, name)

    def test_optimized_audit_script_fails_before_writing_a_receipt(self):
        # q012's parser fails on a missing session before any output is written.
        # The point is that the *optimized* interpreter still exits nonzero and
        # leaves no receipt behind.
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "should-not-exist.json"
            env = dict(os.environ, GNOSTRAL_Q012_EVIDENCE_ROOT=str(Path(tmp) / "missing"))
            proc = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(HERE / "q012_capability_audit.py"),
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
            )
            self.assertNotEqual(proc.returncode, 0, "audit script passed with no evidence")
            self.assertFalse(out.exists(), "audit script wrote a receipt with no evidence")


if __name__ == "__main__":
    unittest.main()
