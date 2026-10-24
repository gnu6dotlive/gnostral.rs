# Q-044 — VOICE-TURN-01

Review date: 2026-10-24
State: **PROPOSED** — turn state machine, barge-in, speakable chunking and first-audio budget landed offline (`crates/gnostral-assistant`); no audio, no model.
Authorization: offline policy and simulation. No microphone capture, recording or audio storage is
authorized by this quest; real audio tests use frozen, consented clips (Q-047).
Workstream: W7 · Milestone: M7 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can one engine run a full voice turn (listen, transcribe, think, speak) with speech starting at the
first sentence of the answer, and stop instantly when the person talks over it?

## Deliverable and file seams

- `gnostral-assistant::turn`: `Idle -> Listening -> Transcribing -> Thinking -> Speaking -> Idle`
  with barge-in from any busy state (`CancelOutput`), typed errors for out-of-order events and a
  monotonic clock; every transition logged.
- `gnostral-assistant::chunk`: streamed text cut into speakable pieces at sentence ends (not inside
  `3.14`), or at the last space once a length cap is reached; lossless.
- `LatencyBudget`: first audio = speech-to-text final + prefill + tokens to first piece x TPOT +
  text-to-speech first piece. Each term comes from Q-042 timers.
- Engine side (later): one request type carrying audio in and audio out through the supervisor.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| First audio | p95 time from end of speech to first audio within the M7 target, all samples kept | target missed or samples dropped |
| Barge-in | output stops within one audio chunk of detected speech; no queued audio plays after | audio continues after cancel |
| Chunking | the spoken text equals the generated text (round trip via Q-047) | lost or duplicated text |
| Timers | each budget term measured, not estimated | any term missing |

## Non-claims

- No latency number exists. The example budget in tests (642 ms) is arithmetic on invented inputs.
- No voice activity detection quality claim; VAD is an input event here.

## Definition of done

1. Engine request path for a voice turn behind the supervisor.
2. Repeated measured turns on frozen clips with first-audio p50/p95 and barge-in timing receipts.
