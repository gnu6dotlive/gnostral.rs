# Q-047 — SPEECH-MODELS-01

Review date: 2026-10-24
State: **PROPOSED** — WER oracle landed (`harness/quality/speech_oracle.py`, 4 tests); no clip set frozen, no speech model run.
Authorization: choose and pin speech models and a frozen, consented clip set; run them through the
base engine. No scraped or personal audio; no voice cloning of real people.
Workstream: W7 · Milestone: M7 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Which speech-to-text and text-to-speech models run in the engine next to the language model, how
large are they on the card, and how good are they, measured the same way every time?

## What the base engine already has

The vendored mistral.rs carries a text-to-speech family (`speech_models/dia`) and audio-input
multimodal models (Voxtral under `vision_models/voxtral`). Which of them fit the M7 envelope, and
whether a dedicated speech-to-text model is needed, is this quest's first answer.

## Deliverable and file seams

- `harness/quality/speech_oracle.py`: documented normalisation (NFC, lower case, punctuation out,
  accents kept), word alignment with substitution/deletion/insertion counts, corpus WER gate with
  `BLOCKED` on missing pins, a different clip manifest or a changed audio hash.
- A frozen clip manifest (`gnostral.speech.clips/1`): French and English clips, each with audio
  SHA-256 and reference transcript.
- Text-to-speech is gated by round trip: speak frozen sentences, transcribe with a pinned
  speech-to-text lane, score with the same oracle.
- A size and latency row per model (Q-031 runner) for the co-hosting broker's inputs.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Pinned | model, binary and clip manifest pinned by SHA-256 | any pin missing (BLOCKED) |
| STT quality | corpus WER within the frozen threshold, per language | threshold missed |
| TTS intelligibility | round-trip WER within the frozen threshold | threshold missed |
| Footprint | measured resident bytes on the card, used as broker input | estimated bytes |

## Non-claims

- No WER, size or latency figure exists yet.
- Round-trip WER measures intelligibility, not naturalness.

## Definition of done

1. Frozen clip set and thresholds committed with hashes.
2. Receipts for at least one speech-to-text and one text-to-speech model in the engine.
