"""Q-047 speech oracle — word error rate gates for speech-to-text and text-to-speech.

CPU-only, like the Q-030 token oracle: it never runs a model. It compares
transcripts against a frozen clip manifest.

- Speech-to-text: the candidate transcript of each frozen clip against the
  clip's reference transcript.
- Text-to-speech: a round trip. The engine speaks a frozen sentence, a pinned
  speech-to-text lane transcribes the audio, and the transcript is scored
  against the sentence. This measures intelligibility, not voice quality.

Rules shared with the token oracle: frozen inputs (one SHA-256 per clip and one
per manifest), no silent defaults, and `UNKNOWN` stays a verdict. Normalisation
is deliberately small and documented: Unicode NFC, lower case, punctuation
removed except apostrophes and hyphens inside words, whitespace collapsed.
Accents are kept: "a" and "à" are different French words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCHEMA_CLIPS = "gnostral.speech.clips/1"
SCHEMA_TRANSCRIPTS = "gnostral.speech.transcripts/1"
SCHEMA_RECEIPT = "gnostral.speech.receipt/1"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_BLOCKED = "BLOCKED"


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "'-’":
            out.append("'" if ch == "’" else ch)
        elif cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    words = []
    for w in "".join(out).split():
        w = w.strip("'-")
        if w:
            words.append(w)
    return words


@dataclass(frozen=True)
class Errors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("empty reference transcript")
        return (self.substitutions + self.deletions + self.insertions) / self.reference_words


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Errors:
    """Minimum edit alignment; ties prefer substitution, then deletion, then insertion."""
    n, m = len(reference), len(hypothesis)
    # cost, substitutions, deletions, insertions
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            c, s, d, ins = prev[j - 1]
            options = [(c + (0 if same else 1), s + (0 if same else 1), d, ins)]
            c, s, d, ins = prev[j]
            options.append((c + 1, s, d + 1, ins))
            c, s, d, ins = cur[j - 1]
            options.append((c + 1, s, d, ins + 1))
            cur.append(min(options, key=lambda o: o[0]))
        prev = cur
    _, s, d, ins = prev[m]
    return Errors(s, d, ins, n)


def wer(reference: str, hypothesis: str) -> float:
    return align(normalize(reference), normalize(hypothesis)).wer


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(raw)


def load_clips(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA_CLIPS:
        raise ValueError(f"expected schema {SCHEMA_CLIPS}")
    clips = doc.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest has no clips")
    seen = set()
    for c in clips:
        for field in ("id", "audio_sha256", "transcript", "language"):
            if not isinstance(c.get(field), str) or not c[field]:
                raise ValueError(f"clip missing {field}")
        if c["id"] in seen:
            raise ValueError(f"duplicate clip {c['id']}")
        seen.add(c["id"])
        if not normalize(c["transcript"]):
            raise ValueError(f"clip {c['id']} has an empty transcript")
    return doc


def score(clips_doc: dict[str, Any], transcripts_doc: dict[str, Any], max_wer: float) -> dict[str, Any]:
    """Per-clip and corpus WER; `BLOCKED` when the transcripts do not name this manifest or a lane pin is missing."""
    manifest_sha = canonical_sha256(clips_doc)
    reasons = []
    if transcripts_doc.get("schema") != SCHEMA_TRANSCRIPTS:
        reasons.append(f"transcripts schema is not {SCHEMA_TRANSCRIPTS}")
    if transcripts_doc.get("clips_sha256") != manifest_sha:
        reasons.append("transcripts name a different clip manifest")
    lane = transcripts_doc.get("lane") or {}
    for pin in ("engine", "binary_sha256", "model_sha256"):
        if not lane.get(pin):
            reasons.append(f"lane pin {pin} missing")
    by_id = {t.get("clip_id"): t for t in transcripts_doc.get("transcripts", [])}
    results = []
    total = Errors(0, 0, 0, 0)
    for clip in clips_doc["clips"]:
        t = by_id.get(clip["id"])
        if t is None or not isinstance(t.get("text"), str):
            reasons.append(f"clip {clip['id']} has no transcript")
            continue
        if t.get("audio_sha256") != clip["audio_sha256"]:
            reasons.append(f"clip {clip['id']} audio hash differs")
            continue
        e = align(normalize(clip["transcript"]), normalize(t["text"]))
        total = Errors(total.substitutions + e.substitutions, total.deletions + e.deletions,
                       total.insertions + e.insertions, total.reference_words + e.reference_words)
        results.append({"clip_id": clip["id"], "wer": e.wer, "substitutions": e.substitutions,
                        "deletions": e.deletions, "insertions": e.insertions,
                        "reference_words": e.reference_words})
    receipt: dict[str, Any] = {
        "schema": SCHEMA_RECEIPT,
        "clips_sha256": manifest_sha,
        "lane": lane,
        "max_wer": max_wer,
        "results": results,
    }
    if reasons:
        receipt.update(verdict=VERDICT_BLOCKED, corpus_wer=None, reasons=reasons)
        return receipt
    corpus = total.wer
    receipt.update(corpus_wer=corpus, verdict=VERDICT_PASS if corpus <= max_wer else VERDICT_FAIL, reasons=[])
    return receipt


def _cmd_wer(args: argparse.Namespace) -> int:
    print(json.dumps({"wer": wer(args.reference, args.hypothesis)}))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    clips = load_clips(args.clips)
    transcripts = json.loads(Path(args.transcripts).read_text(encoding="utf-8"))
    receipt = score(clips, transcripts, args.max_wer)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["verdict"] == VERDICT_PASS else 1


def _cmd_manifest_hash(args: argparse.Namespace) -> int:
    print(json.dumps({"clips_sha256": canonical_sha256(load_clips(args.clips))}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("wer", help="WER of one hypothesis against one reference")
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.set_defaults(func=_cmd_wer)
    p = sub.add_parser("manifest-hash", help="print the frozen clip manifest hash")
    p.add_argument("clips")
    p.set_defaults(func=_cmd_manifest_hash)
    p = sub.add_parser("score", help="corpus WER gate for one lane's transcripts")
    p.add_argument("--clips", required=True)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--max-wer", type=float, required=True)
    p.set_defaults(func=_cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
