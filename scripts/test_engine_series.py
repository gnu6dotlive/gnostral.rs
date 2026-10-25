"""The series checker names the last applied patch, and refuses a tree that matches no state."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import engine_series

BASE = "one\ntwo\nthree\n"
P1 = """--- a/src/a.txt
+++ b/src/a.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
"""
P2 = """--- /dev/null
+++ b/src/new.txt
@@ -0,0 +1 @@
+fresh
"""


class SeriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.patches, self.tree, self.states = root / "patches", root / "tree", root / "states.json"
        self.patches.mkdir()
        (self.patches / "0001-edit.patch").write_text(P1)
        (self.patches / "0002-add.patch").write_text(P2)
        (self.patches / "README.md").write_text("not a patch")
        (self.tree / "src").mkdir(parents=True)
        (self.tree / "src" / "a.txt").write_text(BASE)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(engine_series.record(self.tree, self.states, self.patches), 0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check(self) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = engine_series.check(self.tree, self.states, self.patches)
        return code, out.getvalue()

    def test_each_prefix_of_the_series_is_named(self) -> None:
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn("ENGINE_SERIES_STATE BASELINE", out)
        (self.tree / "src" / "a.txt").write_text(BASE.replace("two", "TWO"))
        code, out = self.check()
        self.assertEqual(code, 0)
        self.assertIn("ENGINE_SERIES_STATE 0001", out)
        self.assertIn("not applied yet: 0002-add.patch", out)
        (self.tree / "src" / "new.txt").write_text("fresh\n")
        self.assertIn("ENGINE_SERIES_STATE 0002", self.check()[1])

    def test_a_local_edit_matches_no_state(self) -> None:
        (self.tree / "src" / "a.txt").write_text(BASE.replace("two", "TWO"))
        (self.tree / "src" / "new.txt").write_text("edited locally\n")
        code, out = self.check()
        self.assertEqual(code, 1)
        self.assertIn("ENGINE_SERIES_STATE NONE", out)
        self.assertIn("src/new.txt: differs", out)


if __name__ == "__main__":
    unittest.main()
