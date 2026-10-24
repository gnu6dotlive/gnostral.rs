"""The llama.cpp lane against a fake server: identity, self-reproduction, honest logprobs."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import llama_server_lane as lane

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "harness" / "quality" / "prompts" / "quality-prompts-20261023.json"


class LlamaLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.server = self.dir / "llama-server"
        self.server.write_text(f"#!/bin/sh\nexec {sys.executable} -I {HERE / 'fake_llama_server.py'} \"$@\"\n")
        self.server.chmod(self.server.stat().st_mode | stat.S_IEXEC)
        self.model = self.dir / "m.gguf"
        self.model.write_bytes(b"model")

    def run_lane(self, extra_env: dict | None = None, server_args: str = "-ngl 3 --cpu-moe",
                 *more: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, **(extra_env or {}))
        return subprocess.run(
            [sys.executable, "-I", str(HERE / "llama_server_lane.py"), "--server-bin", str(self.server),
             "--model", str(self.model), "--manifest", str(MANIFEST), "--out", str(self.dir),
             "--max-tokens", "8", f"--server-args={server_args}", *more],
            capture_output=True, text=True, env=env, timeout=120)

    def lines(self, done: subprocess.CompletedProcess, prefix: str) -> list[str]:
        return [l[len(prefix):] for l in done.stdout.splitlines() if l.startswith(prefix)]

    def test_reference_then_pass_with_pinned_placement_and_metrics(self) -> None:
        first = self.run_lane()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self.lines(first, "GNOSTRAL_PLACEMENT "), ["llama.cpp offloaded=3/3 cpu_moe=on"])
        self.assertEqual(self.lines(first, "GNOSTRAL_SELF_REPRO "), ["REFERENCE"])
        metrics = json.loads(self.lines(first, "GNOSTRAL_METRICS ")[0])
        self.assertAlmostEqual(metrics["decode_tok_s"], 500.0)  # 8 tokens per 16 ms in the fake
        self.assertAlmostEqual(metrics["prefill_tok_s"], 10000.0)
        second = self.run_lane()
        self.assertEqual(self.lines(second, "GNOSTRAL_SELF_REPRO "), ["PASS"])

    def test_one_ulp_drift_fails_self_reproduction(self) -> None:
        self.assertEqual(self.run_lane().returncode, 0)
        drifted = self.run_lane({"FAKE_LLAMA_DRIFT": "1"})
        verdict = self.lines(drifted, "GNOSTRAL_SELF_REPRO ")[0]
        self.assertTrue(verdict.startswith("FAIL"), verdict)
        self.assertIn("kind=logprob", verdict)

    def test_entries_for_incomplete_utf8_are_recorded_as_reported(self) -> None:
        self.assertEqual(self.run_lane().returncode, 0)
        docs = json.loads((self.dir / "capture-cpu-experts-llama.cpp-latest.logprobs.json").read_text())
        capture = json.loads((self.dir / "capture-cpu-experts-llama.cpp-latest.json").read_text())
        tokens = {c["prompt_id"]: c["token_ids"] for c in capture["captures"]}
        for doc in docs:
            kept = [t for t in tokens[doc["prompt_id"]] if t not in {200, 210}]
            self.assertEqual(doc["entry_token_ids"], kept)
            self.assertEqual(len(doc["steps"]), len(kept))

    def test_automatic_layer_count_is_refused(self) -> None:
        done = self.run_lane(None, "--cpu-moe")
        self.assertEqual(done.returncode, 4)
        self.assertIn("-ngl", done.stderr)

    def test_stop_file_skips_the_run(self) -> None:
        stop = self.dir / "PARITY_FAIL"
        stop.write_text("per-route\n")
        done = self.run_lane(None, "-ngl 3", "--stop-file", str(stop))
        self.assertEqual(done.returncode, 5)

    def test_helpers(self) -> None:
        self.assertTrue(lane.is_subsequence([1, 3], [1, 2, 3]))
        self.assertFalse(lane.is_subsequence([3, 1], [1, 2, 3]))
        log = "load_tensors: offloaded 49/49 layers to GPU\n"
        self.assertEqual(lane.placement_from_log(log, ["-ngl", "99", "--cpu-moe"]),
                         "llama.cpp offloaded=49/49 cpu_moe=on")
        self.assertIsNone(lane.placement_from_log(log + "load_tensors: offloaded 20/49 layers to GPU\n", []))
        self.assertEqual(lane.placement_from_log("", ["-ngl", "0"]), "llama.cpp offloaded=none cpu_moe=off")


if __name__ == "__main__":
    unittest.main()
