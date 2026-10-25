"""The A/B schedule alternates, gates refuse, and only disjoint ranges rank."""

from __future__ import annotations

import unittest

import binary_ab


def run(label: str, tok_s: float, rnd: int = 1, gates: list[str] | None = None) -> dict:
    return {"label": label, "phase": "measured", "round": rnd, "metrics": {"decode_tok_s": tok_s},
            "gate_failures": gates or []}


class AbTests(unittest.TestCase):
    def test_schedule_warms_both_then_alternates(self) -> None:
        self.assertEqual(binary_ab.schedule(["a", "b"], 3), [
            ("a", "warmup", 0), ("b", "warmup", 0),
            ("a", "measured", 1), ("b", "measured", 1),
            ("b", "measured", 2), ("a", "measured", 2),
            ("a", "measured", 3), ("b", "measured", 3)])

    def test_overlap_is_inconclusive_and_disjoint_ranks(self) -> None:
        runs = [run("a", 10), run("a", 12), run("a", 11), run("b", 11.5), run("b", 13), run("b", 14)]
        self.assertEqual(binary_ab.verdict(runs, ["a", "b"])["result"], "inconclusive (ranges overlap)")
        runs = [run("a", 10), run("a", 12), run("a", 11), run("b", 12.5), run("b", 13), run("b", 14)]
        v = binary_ab.verdict(runs, ["a", "b"])
        self.assertEqual(v["result"], "b faster")
        self.assertAlmostEqual(v["chance_of_separated_ranges_if_equal"], 0.1)
        runs = [run("a", 20), run("a", 21), run("a", 22), run("b", 12.5), run("b", 13), run("b", 14)]
        self.assertEqual(binary_ab.verdict(runs, ["a", "b"])["result"], "b slower")

    def test_any_failed_gate_refuses(self) -> None:
        runs = [run("a", 10), run("a", 11), run("a", 12),
                run("b", 30), run("b", 31), run("b", 32, gates=["GNOSTRAL_SELF_REPRO FAIL x"])]
        v = binary_ab.verdict(runs, ["a", "b"])
        self.assertEqual(v["comparison"], "REFUSED")
        self.assertIn("b round 1: GNOSTRAL_SELF_REPRO FAIL x", v["reasons"])

    def test_placement_strings_match_the_engine_report(self) -> None:
        self.assertEqual(binary_ab.expected_placement("cuda", 48, None), "0-47:cuda:0;nm=cuda:0")
        self.assertEqual(binary_ab.expected_placement("cuda", 48, 40), "0-39:cuda:0,40-47:cpu;nm=cuda:0")
        self.assertEqual(binary_ab.expected_placement("cpu", 4, None), "0-3:cpu;nm=cpu")

    def test_chance_of_separation(self) -> None:
        self.assertAlmostEqual(binary_ab.chance_of_separation(2, 2), 1 / 3)
        self.assertLess(binary_ab.chance_of_separation(5, 5), 0.01)


if __name__ == "__main__":
    unittest.main()
