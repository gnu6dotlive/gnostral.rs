"""The driver probe says NO_XID only after actually reading this boot's kernel log."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import driver_probe as dp


def runner(journal: tuple[int, str], dmesg: tuple[int, str] = (1, "")):
    def run(argv: list[str]) -> tuple[int, str]:
        return journal if argv[0] == "journalctl" else dmesg
    return run


JOURNAL = "Linux version 6.8\nnvidia: loading out-of-tree module\n"


class DriverProbeTests(unittest.TestCase):
    def test_clean_journal_is_no_xid(self) -> None:
        status, detail = dp.probe(runner((0, JOURNAL)), lambda: True)
        self.assertEqual(status, "NO_XID")
        self.assertIn("journalctl lines=2", detail)

    def test_xid_line_is_xid(self) -> None:
        log = JOURNAL + "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n"
        status, detail = dp.probe(runner((0, log)), lambda: True)
        self.assertEqual(status, "XID")
        self.assertIn("xid_lines=1", detail)

    def test_empty_or_forbidden_journal_is_unknown_not_no_xid(self) -> None:
        # The F-016 case: "No journal files were found." and an empty boot log.
        status, detail = dp.probe(runner((0, "-- No entries --\n")), lambda: True)
        self.assertEqual(status, "UNKNOWN")
        self.assertIn("not readable", detail)
        status, _ = dp.probe(runner((1, "")), lambda: True)
        self.assertEqual(status, "UNKNOWN")

    def test_dmesg_is_a_fallback(self) -> None:
        status, detail = dp.probe(runner((1, ""), (0, JOURNAL)), lambda: True)
        self.assertEqual(status, "NO_XID")
        self.assertIn("dmesg", detail)

    def test_no_nvidia_driver_is_unknown(self) -> None:
        status, detail = dp.probe(runner((0, JOURNAL)), lambda: False)
        self.assertEqual(status, "UNKNOWN")
        self.assertIn("no NVIDIA", detail)

    def test_cli_prints_exactly_one_status_line(self) -> None:
        script = Path(dp.__file__).resolve()
        done = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0)
        status_lines = [l for l in done.stdout.splitlines() if l.startswith("GNOSTRAL_DRIVER ")]
        self.assertEqual(len(status_lines), 1)
        self.assertIn(status_lines[0].split()[1], {"NO_XID", "XID", "UNKNOWN"})


if __name__ == "__main__":
    unittest.main()
