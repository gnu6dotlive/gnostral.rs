"""Q-047 speech oracle: normalisation, alignment counts and the BLOCKED rules."""

from __future__ import annotations

import unittest

import speech_oracle as so


def clips() -> dict:
    return {
        "schema": so.SCHEMA_CLIPS,
        "clips": [
            {"id": "c1", "audio_sha256": "a" * 64, "language": "fr", "transcript": "Il est trois heures à Montréal."},
            {"id": "c2", "audio_sha256": "b" * 64, "language": "en", "transcript": "Turn the lights off, please."},
        ],
    }


def transcripts(doc: dict, texts: dict[str, str], **lane) -> dict:
    pins = {"engine": "stt-lane", "binary_sha256": "c" * 64, "model_sha256": "d" * 64}
    pins.update(lane)
    return {
        "schema": so.SCHEMA_TRANSCRIPTS,
        "clips_sha256": so.canonical_sha256(doc),
        "lane": pins,
        "transcripts": [
            {"clip_id": c["id"], "audio_sha256": c["audio_sha256"], "text": texts[c["id"]]}
            for c in doc["clips"]
            if c["id"] in texts
        ],
    }


class SpeechOracleTests(unittest.TestCase):
    def test_normalisation_keeps_accents_and_inner_apostrophes(self) -> None:
        self.assertEqual(so.normalize("L’heure, c'est « à » Montréal !"), ["l'heure", "c'est", "à", "montréal"])
        self.assertEqual(so.normalize("well-known -- yes."), ["well-known", "yes"])
        self.assertNotEqual(so.normalize("a"), so.normalize("à"))

    def test_alignment_counts(self) -> None:
        e = so.align(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
        self.assertEqual((e.substitutions, e.deletions, e.insertions), (1, 0, 1))
        self.assertAlmostEqual(e.wer, 0.5)
        e = so.align(["a", "b", "c"], ["c"])
        self.assertEqual((e.substitutions, e.deletions, e.insertions), (0, 2, 0))
        self.assertEqual(so.wer("Bonjour.", "bonjour"), 0.0)
        with self.assertRaises(ValueError):
            so.align([], ["x"]).wer

    def test_score_passes_and_fails_on_threshold(self) -> None:
        doc = clips()
        good = transcripts(doc, {"c1": "il est trois heures a montréal", "c2": "turn the lights off please"})
        r = so.score(doc, good, max_wer=0.1)
        self.assertEqual(r["verdict"], so.VERDICT_PASS)
        self.assertAlmostEqual(r["corpus_wer"], 1 / 11)
        self.assertEqual(so.score(doc, good, max_wer=0.05)["verdict"], so.VERDICT_FAIL)

    def test_missing_pins_hashes_or_clips_block(self) -> None:
        doc = clips()
        texts = {"c1": "x", "c2": "y"}
        self.assertEqual(so.score(doc, transcripts(doc, texts, model_sha256=""), 1.0)["verdict"], so.VERDICT_BLOCKED)
        self.assertEqual(so.score(doc, transcripts(doc, {"c1": "x"}), 1.0)["verdict"], so.VERDICT_BLOCKED)
        other = transcripts(doc, texts)
        other["clips_sha256"] = "0" * 64
        r = so.score(doc, other, 1.0)
        self.assertEqual(r["verdict"], so.VERDICT_BLOCKED)
        self.assertIsNone(r["corpus_wer"])
        swapped = transcripts(doc, texts)
        swapped["transcripts"][0]["audio_sha256"] = "e" * 64
        self.assertEqual(so.score(doc, swapped, 1.0)["verdict"], so.VERDICT_BLOCKED)


if __name__ == "__main__":
    unittest.main()
