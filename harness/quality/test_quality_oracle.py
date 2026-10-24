"""Tests for the Q-030 quality oracle. Deterministic, offline, no model involved."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import quality_oracle as oracle

MANIFEST = Path(__file__).resolve().parent / "prompts" / "quality-prompts-20261023.json"
BINARY = "a" * 64
MODEL = "b" * 64


def _capture_document(manifest: oracle.PromptSet, tokens: dict[str, list[int]],
                      binary: str | None = BINARY, model: str | None = MODEL,
                      temperature: float = 0.0) -> dict:
    known = manifest.by_id()
    return {
        "schema": oracle.SCHEMA_CAPTURE,
        "promptset_sha256": manifest.manifest_sha256,
        "lane": {"engine": "test-engine", "binary_sha256": binary, "model_sha256": model},
        "sampling": {"temperature": temperature, "seed": 0},
        "captures": [
            {"prompt_id": pid, "prompt_sha256": known[pid].sha256, "token_ids": ids}
            for pid, ids in tokens.items()
        ],
    }


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = oracle.load_manifest(MANIFEST)

    def test_manifest_is_frozen_and_hashed(self) -> None:
        self.assertEqual(self.manifest.name, "six-check-20261023")
        self.assertEqual(len(self.manifest.prompts), 6)
        self.assertEqual(len(self.manifest.manifest_sha256), 64)

    def test_every_prompt_hash_matches_its_text(self) -> None:
        for prompt in self.manifest.prompts:
            self.assertEqual(prompt.sha256, oracle.sha256_text(prompt.text))

    def test_tampered_prompt_text_is_rejected(self) -> None:
        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["prompts"][0]["text"] += " x"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(raw, handle)
            path = handle.name
        with self.assertRaises(oracle.OracleError):
            oracle.load_manifest(path)

    def test_duplicate_prompt_id_is_rejected(self) -> None:
        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["prompts"][1]["id"] = raw["prompts"][0]["id"]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(raw, handle)
            path = handle.name
        with self.assertRaises(oracle.OracleError):
            oracle.load_manifest(path)


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = oracle.load_manifest(MANIFEST)
        self.ids = [prompt.id for prompt in self.manifest.prompts]
        self.tokens = {pid: [1, 2, 3] for pid in self.ids}

    def _write(self, document: dict) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(document, handle)
            return handle.name

    def test_valid_capture_loads(self) -> None:
        capture = oracle.load_capture(self._write(_capture_document(self.manifest, self.tokens)), self.manifest)
        self.assertTrue(capture.lane.pinned)
        self.assertEqual(len(capture.captures), 6)

    def test_wrong_promptset_hash_is_rejected(self) -> None:
        document = _capture_document(self.manifest, self.tokens)
        document["promptset_sha256"] = "c" * 64
        with self.assertRaises(oracle.OracleError):
            oracle.load_capture(self._write(document), self.manifest)

    def test_incomplete_capture_is_rejected(self) -> None:
        partial = dict(self.tokens)
        partial.pop(self.ids[0])
        with self.assertRaises(oracle.OracleError):
            oracle.load_capture(self._write(_capture_document(self.manifest, partial)), self.manifest)

    def test_unknown_prompt_is_rejected(self) -> None:
        document = _capture_document(self.manifest, self.tokens)
        document["captures"].append(
            {"prompt_id": "p9-invented", "prompt_sha256": "d" * 64, "token_ids": [1]}
        )
        with self.assertRaises(oracle.OracleError):
            oracle.load_capture(self._write(document), self.manifest)

    def test_non_integer_token_is_rejected(self) -> None:
        tokens = dict(self.tokens)
        tokens[self.ids[0]] = [1, 2.5, 3]
        with self.assertRaises(oracle.OracleError):
            oracle.load_capture(self._write(_capture_document(self.manifest, tokens)), self.manifest)

    def test_unpinned_lane_loads_but_is_blocked(self) -> None:
        document = _capture_document(self.manifest, self.tokens, binary=None, model=None)
        capture = oracle.load_capture(self._write(document), self.manifest)
        self.assertFalse(capture.lane.pinned)


class ParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = oracle.load_manifest(MANIFEST)
        self.ids = [prompt.id for prompt in self.manifest.prompts]
        self.tokens = {pid: [10, 11, 12, 13] for pid in self.ids}

    def _capture(self, tokens: dict[str, list[int]] | None = None, **kwargs) -> oracle.LaneCapture:
        tokens = tokens or self.tokens
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(_capture_document(self.manifest, tokens, **kwargs), handle)
            path = handle.name
        return oracle.load_capture(path, self.manifest)

    def test_identical_streams_pass(self) -> None:
        result = oracle.greedy_parity(self._capture(), self._capture(), self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_PASS)
        self.assertIsNone(result.divergence_index)
        self.assertEqual(result.matched_prefix, 4)

    def test_divergence_index_is_reported(self) -> None:
        changed = dict(self.tokens)
        changed[self.ids[0]] = [10, 11, 99, 13]
        result = oracle.greedy_parity(self._capture(), self._capture(changed), self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_DIVERGED)
        self.assertEqual(result.divergence_index, 2)
        self.assertEqual(result.matched_prefix, 2)

    def test_truncated_stream_diverges_at_end(self) -> None:
        changed = dict(self.tokens)
        changed[self.ids[0]] = [10, 11]
        result = oracle.greedy_parity(self._capture(), self._capture(changed), self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_DIVERGED)
        self.assertEqual(result.divergence_index, 2)

    def test_different_model_file_is_blocked(self) -> None:
        other = self._capture(model="c" * 64)
        result = oracle.greedy_parity(self._capture(), other, self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_BLOCKED)

    def test_unpinned_lane_is_blocked(self) -> None:
        unpinned = self._capture(binary=None, model=None)
        result = oracle.greedy_parity(self._capture(), unpinned, self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_BLOCKED)

    def test_sampling_lane_is_blocked_for_greedy_parity(self) -> None:
        sampled = self._capture(temperature=0.7)
        result = oracle.greedy_parity(self._capture(), sampled, self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_BLOCKED)

    def test_batch_invariance_relabels_divergence(self) -> None:
        changed = dict(self.tokens)
        changed[self.ids[0]] = [10, 42, 12, 13]
        result = oracle.batch_invariance(self._capture(), self._capture(changed), self.ids[0])
        self.assertEqual(result.verdict, oracle.VERDICT_DIVERGED)
        self.assertIn("batch invariance failed", result.detail)


class DistributionTests(unittest.TestCase):
    def test_kl_of_identical_distributions_is_zero(self) -> None:
        self.assertAlmostEqual(oracle.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, places=12)

    def test_kl_known_value(self) -> None:
        value = oracle.kl_divergence([0.5, 0.5], [0.25, 0.75])
        self.assertAlmostEqual(value, 0.1438410362258904, places=6)

    def test_kl_rejects_unnormalised_input(self) -> None:
        with self.assertRaises(oracle.OracleError):
            oracle.kl_divergence([0.5, 0.6], [0.5, 0.5])

    def test_kl_rejects_negative_probability(self) -> None:
        with self.assertRaises(oracle.OracleError):
            oracle.kl_divergence([1.5, -0.5], [0.5, 0.5])

    def test_kl_rejects_length_mismatch(self) -> None:
        with self.assertRaises(oracle.OracleError):
            oracle.kl_divergence([0.5, 0.5], [0.5, 0.25, 0.25])

    def test_kl_epsilon_floor_is_finite_and_declared(self) -> None:
        value = oracle.kl_divergence([0.5, 0.5], [1.0, 0.0], epsilon=1e-6)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_ks_accepts_uniform_pit(self) -> None:
        samples = [(index + 0.5) / 32 for index in range(32)]
        result = oracle.uniformity_verdict(samples)
        self.assertEqual(result["verdict"], "PASS")
        self.assertGreater(result["p_value"], 0.05)

    def test_ks_rejects_clustered_pit(self) -> None:
        samples = [0.02 + 0.01 * index for index in range(16)] + [0.9] * 16
        result = oracle.uniformity_verdict(samples)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertLess(result["p_value"], 0.01)

    def test_ks_rejects_out_of_range_samples(self) -> None:
        with self.assertRaises(oracle.OracleError):
            oracle.pit_ks([1.2] * 16)

    def test_ks_rejects_tiny_sample(self) -> None:
        with self.assertRaises(oracle.OracleError):
            oracle.pit_ks([0.5])


class CliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "quality_oracle.py"), *args],
            capture_output=True, text=True, check=False,
        )

    def test_manifest_hash_command(self) -> None:
        completed = self._run("manifest-hash", str(MANIFEST))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["prompts"], 6)
        self.assertEqual(len(payload["manifest_sha256"]), 64)

    def test_parity_exit_code_on_divergence(self) -> None:
        manifest = oracle.load_manifest(MANIFEST)
        ids = [prompt.id for prompt in manifest.prompts]
        left = {pid: [1, 2, 3] for pid in ids}
        right = dict(left)
        right[ids[0]] = [1, 9, 3]
        paths = []
        for tokens in (left, right):
            handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(_capture_document(manifest, tokens), handle)
            handle.close()
            paths.append(handle.name)
        completed = self._run(
            "parity", "--manifest", str(MANIFEST),
            "--reference", paths[0], "--candidate", paths[1],
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["results"][0]["verdict"], oracle.VERDICT_DIVERGED)

    def test_parity_exit_code_on_unpinned_lane(self) -> None:
        manifest = oracle.load_manifest(MANIFEST)
        ids = [prompt.id for prompt in manifest.prompts]
        tokens = {pid: [1, 2, 3] for pid in ids}
        paths = []
        for document in (
            _capture_document(manifest, tokens),
            _capture_document(manifest, tokens, binary=None, model=None),
        ):
            handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(document, handle)
            handle.close()
            paths.append(handle.name)
        completed = self._run(
            "parity", "--manifest", str(MANIFEST),
            "--reference", paths[0], "--candidate", paths[1],
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)


if __name__ == "__main__":
    unittest.main()


class SelfReproductionTests(unittest.TestCase):
    """One lane captured twice must match itself bit for bit (docs/FINDINGS.md, F-010)."""

    def setUp(self) -> None:
        self.manifest = oracle.load_manifest(MANIFEST)
        self.ids = [prompt.id for prompt in self.manifest.prompts]
        self.dir = Path(tempfile.mkdtemp())

    def _pair(self, name: str, tokens: dict[str, list[int]], logprob: float = -0.25,
              placement: str | None = "0-47:cuda:0;nm=cuda:0", **kwargs) -> tuple[str, str]:
        document = _capture_document(self.manifest, tokens, **kwargs)
        document["lane"]["placement"] = placement
        capture = self.dir / f"{name}.json"
        capture.write_text(json.dumps(document), encoding="utf-8")
        steps = [{"prompt_id": pid, "steps": [[[t, logprob], [t + 1, logprob - 0.25]] for t in ids]}
                 for pid, ids in tokens.items()]
        logprobs = self.dir / f"{name}.logprobs.json"
        logprobs.write_text(json.dumps(steps), encoding="utf-8")
        return str(capture), str(logprobs)

    def _run(self, a: tuple[str, str], b: tuple[str, str]) -> dict:
        return oracle.self_reproduction(self.manifest, a[0], b[0], a[1], b[1])

    def test_identical_runs_pass(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        result = self._run(self._pair("a", tokens), self._pair("b", tokens))
        self.assertEqual(result["verdict"], "PASS")
        self.assertIsNone(result["first_difference"])

    def test_one_ulp_logprob_change_fails_before_any_token_changes(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        nudged = math.nextafter(-0.25, 0.0)
        result = self._run(self._pair("a", tokens), self._pair("b", tokens, logprob=nudged))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["first_difference"]["kind"], "logprob")
        self.assertEqual(result["first_difference"]["index"], 0)

    def test_token_divergence_is_located(self) -> None:
        tokens = {pid: [1, 2, 3, 4] for pid in self.ids}
        other = dict(tokens)
        other[self.ids[2]] = [1, 2, 9, 4]
        result = self._run(self._pair("a", tokens), self._pair("b", other))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["first_difference"]["prompt_id"], self.ids[2])
        self.assertEqual(result["first_difference"]["index"], 2)

    def test_placement_change_fails(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        result = self._run(self._pair("a", tokens),
                           self._pair("b", tokens, placement="0-15:cuda:0,16-47:cpu;nm=cuda:0"))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["first_difference"]["kind"], "placement")

    def test_different_lane_or_unpinned_lane_is_blocked(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        result = self._run(self._pair("a", tokens), self._pair("b", tokens, model="c" * 64))
        self.assertEqual(result["verdict"], oracle.VERDICT_BLOCKED)
        result = self._run(self._pair("c", tokens, binary=None), self._pair("d", tokens, binary=None))
        self.assertEqual(result["verdict"], oracle.VERDICT_BLOCKED)

    def test_any_other_lane_identity_field_must_match(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        a, b = self._pair("a", tokens), self._pair("b", tokens)
        doc = json.loads(Path(b[0]).read_text(encoding="utf-8"))
        doc["lane"]["build_tree_sha256"] = "f" * 64
        Path(b[0]).write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self._run(a, b)["verdict"], oracle.VERDICT_BLOCKED)

    def test_cli_exit_codes(self) -> None:
        tokens = {pid: [1, 2, 3] for pid in self.ids}
        a, b = self._pair("a", tokens), self._pair("b", tokens, logprob=-0.5)
        script = Path(oracle.__file__).resolve()
        for second, code in ((self._pair("same", tokens), 0), (b, 1)):
            done = subprocess.run([sys.executable, "-I", str(script), "self-repro", "--manifest", str(MANIFEST),
                                   "--first", a[0], "--second", second[0],
                                   "--first-logprobs", a[1], "--second-logprobs", second[1]],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, code, done.stderr)
