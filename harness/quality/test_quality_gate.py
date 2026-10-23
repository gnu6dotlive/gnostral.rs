"""Q-030 full gate: frozen thresholds, 32-prompt set, divergence ledger, batch and tier checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import quality_oracle as oracle

HERE = Path(__file__).resolve().parent
MANIFEST_32 = HERE / "prompts" / "quality-prompts-32-20261023.json"
MANIFEST_6 = HERE / "prompts" / "quality-prompts-20261023.json"
GATES = HERE / "gates" / "q030-gates-20261023.json"
BINARY = "a" * 64
MODEL = "b" * 64


def _write(document: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(document, handle)
        return handle.name


def _capture(manifest: oracle.PromptSet, tokens: dict[str, list[int]], engine: str = "ref",
             binary: str | None = BINARY) -> oracle.LaneCapture:
    known = manifest.by_id()
    doc = {
        "schema": oracle.SCHEMA_CAPTURE,
        "promptset_sha256": manifest.manifest_sha256,
        "lane": {"engine": engine, "binary_sha256": binary, "model_sha256": MODEL},
        "sampling": {"temperature": 0, "seed": 0},
        "captures": [
            {"prompt_id": pid, "prompt_sha256": known[pid].sha256, "token_ids": ids}
            for pid, ids in tokens.items()
        ],
    }
    return oracle.load_capture(_write(doc), manifest)


class FrozenInputsTests(unittest.TestCase):
    def test_32_prompt_set_is_frozen_with_its_published_hash(self) -> None:
        manifest = oracle.load_manifest(MANIFEST_32)
        self.assertEqual(len(manifest.prompts), 32)
        self.assertEqual(
            manifest.manifest_sha256,
            "c573a8d81684b4ce7630bf585e95860600f5d1ec7cbb469c9de0e7604cd1787e",
        )

    def test_gates_match_the_program_scorecard(self) -> None:
        gates = oracle.load_gates(GATES)
        self.assertEqual(gates.min_prompts, 32)
        self.assertEqual(gates.min_greedy_agreement, 0.995)
        self.assertEqual(gates.max_kl_nats, 1e-3)
        self.assertEqual(gates.batch_sizes, (1, 2, 4))
        self.assertEqual(gates.long_context_tiers, (1024, 8192, 32768))
        self.assertEqual(len(gates.sha256), 64)

    def test_kl_cli_default_is_the_program_threshold(self) -> None:
        args = oracle.build_parser().parse_args(["kl", "--reference", "p", "--candidate", "q"])
        self.assertEqual(args.max_kl, 1e-3)


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = oracle.load_manifest(MANIFEST_32)
        self.gates = oracle.load_gates(GATES)
        self.ids = [p.id for p in self.manifest.prompts]
        self.base = {pid: list(range(1, 101)) for pid in self.ids}
        self.reference = _capture(self.manifest, self.base)

    def _candidate(self, diverge: dict[str, int] | None = None) -> oracle.LaneCapture:
        tokens = {pid: list(ids) for pid, ids in self.base.items()}
        for pid, index in (diverge or {}).items():
            tokens[pid][index] = 999
        return _capture(self.manifest, tokens, engine="cand", binary="c" * 64)

    def test_identical_lanes_with_all_checks_pass(self) -> None:
        cand = self._candidate()
        receipt = oracle.gate_receipt(
            self.manifest, self.gates, self.reference, cand, {},
            {2: cand, 4: cand}, {1024: cand, 8192: cand, 32768: cand},
        )
        self.assertEqual(receipt["verdict"], "PASS", receipt)
        self.assertEqual(receipt["greedy_agreement"], 1.0)
        self.assertEqual(receipt["gates"]["sha256"], self.gates.sha256)

    def test_missing_batch_or_tier_captures_are_incomplete_not_pass(self) -> None:
        receipt = oracle.gate_receipt(self.manifest, self.gates, self.reference, self._candidate())
        self.assertEqual(receipt["verdict"], "INCOMPLETE")
        self.assertIn("batch 2", receipt["unknown"])
        self.assertIn("tier 32768", receipt["unknown"])

    def test_unexplained_divergence_is_a_hard_fail_even_above_threshold(self) -> None:
        cand = self._candidate({"q05-rust-fn": 99})
        receipt = oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, {},
                                      {2: cand, 4: cand}, {t: cand for t in (1024, 8192, 32768)})
        self.assertGreater(receipt["greedy_agreement"], 0.995)
        self.assertEqual(receipt["verdict"], "FAIL")
        self.assertTrue(any("unexplained" in f for f in receipt["failed"]))

    def test_ledger_explains_a_divergence_only_at_its_exact_index(self) -> None:
        cand = self._candidate({"q05-rust-fn": 99})
        entry = {"prompt_id": "q05-rust-fn", "category": "kernel_reduction_order",
                 "explanation": "Q3_K dot keeps one i32 per block instead of 8 lanes",
                 "evidence_ref": "evidence/runs/fixture"}
        good = oracle.load_ledger(_write({"schema": oracle.SCHEMA_LEDGER, "entries": [{**entry, "divergence_index": 99}]}), self.gates)
        wrong = oracle.load_ledger(_write({"schema": oracle.SCHEMA_LEDGER, "entries": [{**entry, "divergence_index": 98}]}), self.gates)
        complete = ({2: cand, 4: cand}, {t: cand for t in (1024, 8192, 32768)})
        self.assertEqual(oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, good, *complete)["verdict"], "PASS")
        self.assertEqual(oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, wrong, *complete)["verdict"], "FAIL")
        with self.assertRaises(oracle.OracleError):
            oracle.load_ledger(_write({"schema": oracle.SCHEMA_LEDGER, "entries": [{**entry, "divergence_index": 1, "category": "felt_right"}]}), self.gates)

    def test_low_agreement_fails_even_when_explained(self) -> None:
        diverge = {pid: 0 for pid in self.ids[:2]}
        cand = self._candidate(diverge)
        ledger = {(pid, 0): oracle.LedgerEntry(pid, 0, "block_order", "x", "y") for pid in diverge}
        receipt = oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, ledger,
                                      {2: cand, 4: cand}, {t: cand for t in (1024, 8192, 32768)})
        self.assertEqual(receipt["verdict"], "FAIL")
        self.assertTrue(any("agreement" in f for f in receipt["failed"]))

    def test_batch_and_long_context_drift_fail(self) -> None:
        cand = self._candidate()
        drift = self._candidate({"q10-fr-explain": 50})
        r = oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, {},
                                {2: cand, 4: drift}, {t: cand for t in (1024, 8192, 32768)})
        self.assertIn("batch invariance failed at batch 4", r["failed"])
        r = oracle.gate_receipt(self.manifest, self.gates, self.reference, cand, {},
                                {2: cand, 4: cand}, {1024: cand, 8192: cand, 32768: drift})
        self.assertTrue(any("32768" in f for f in r["failed"]))

    def test_small_prompt_set_or_unpinned_lane_blocks(self) -> None:
        small = oracle.load_manifest(MANIFEST_6)
        tokens = {p.id: [1, 2] for p in small.prompts}
        r = oracle.gate_receipt(small, self.gates, _capture(small, tokens), _capture(small, tokens))
        self.assertEqual(r["verdict"], "BLOCKED")
        unpinned = _capture(self.manifest, self.base, binary=None)
        r = oracle.gate_receipt(self.manifest, self.gates, self.reference, unpinned)
        self.assertEqual(r["verdict"], "BLOCKED")

    def test_cli_gate_exit_codes(self) -> None:
        known = self.manifest.by_id()

        def doc(tokens: dict[str, list[int]], binary: str) -> str:
            return _write({
                "schema": oracle.SCHEMA_CAPTURE,
                "promptset_sha256": self.manifest.manifest_sha256,
                "lane": {"engine": "x", "binary_sha256": binary, "model_sha256": MODEL},
                "sampling": {"temperature": 0, "seed": 0},
                "captures": [{"prompt_id": p, "prompt_sha256": known[p].sha256, "token_ids": t} for p, t in tokens.items()],
            })

        ref = doc(self.base, BINARY)
        cand = doc(self.base, "c" * 64)
        base = [sys.executable, "-I", str(HERE / "quality_oracle.py"), "gate", "--manifest", str(MANIFEST_32),
                "--gates", str(GATES), "--reference", ref, "--candidate", cand]
        full = base + ["--batch", f"2={cand}", "--batch", f"4={cand}"]
        full += [arg for t in (1024, 8192, 32768) for arg in ("--long-context", f"{t}={cand}")]
        self.assertEqual(subprocess.run(full, capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(base, capture_output=True).returncode, 1)
        bad = subprocess.run(base + ["--batch", "two=x"], capture_output=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
