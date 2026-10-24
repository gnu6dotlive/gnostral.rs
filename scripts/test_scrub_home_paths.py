"""The home-path scrubber rewrites personal prefixes only, and its check mode agrees with it."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import scrub_home_paths as scrub

# Built at run time so this file never contains a literal home path itself.
LAB = "/" + "home" + "/someone_lab"
MAC = "/" + "Users" + "/Some.One"


class ScrubTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = scrub.main(list(args))
        return code, out.getvalue()

    def test_personal_prefixes_become_tilde_and_others_stay(self) -> None:
        f = self.dir / "plan.json"
        f.write_text(f'{{"cmd": "{LAB}/Projects/gnostral.rs/bin --model /mnt/models/x.gguf", '
                     f'"log": "{MAC}/tmp/a", "tmp": "/tmp/claude-1000/x", "bare": "{LAB}"}}')
        code, _ = self.run_main(str(self.dir))
        self.assertEqual(code, 0)
        text = f.read_text()
        self.assertIn('"~/Projects/gnostral.rs/bin --model /mnt/models/x.gguf"', text)
        self.assertIn('"~/tmp/a"', text)
        self.assertIn('"/tmp/claude-1000/x"', text)
        self.assertIn('"bare": "~"', text)
        self.assertNotIn("someone_lab", text)

    def test_check_mode_reports_and_changes_nothing(self) -> None:
        f = self.dir / "run.stdout"
        f.write_text(f"Generated: {LAB}/Projects/r/x.nsys-rep\n")
        code, out = self.run_main("--check", str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("someone_lab", out)
        self.assertIn(LAB, f.read_text())
        self.run_main(str(self.dir))
        self.assertEqual(self.run_main("--check", str(self.dir))[0], 0)

    def test_binary_files_are_skipped(self) -> None:
        f = self.dir / "blob.bin"
        payload = b"\0\1" + LAB.encode() + b"/x"
        f.write_bytes(payload)
        self.run_main(str(self.dir))
        self.assertEqual(f.read_bytes(), payload)

    def test_missing_path_is_an_error(self) -> None:
        self.assertEqual(scrub.main([str(self.dir / "nope")]), 2)


if __name__ == "__main__":
    unittest.main()
