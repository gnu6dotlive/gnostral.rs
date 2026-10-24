# Q-045 — GPU-SHARE-01

Review date: 2026-10-24
State: **PROPOSED** — admission test, realtime-first policy and schedule simulator landed offline (`gnostral-assistant::share`); no device run.
Authorization: offline policy and simulation; engine hooks only behind a flag with Q-030 parity.
Workstream: W7 · Milestone: M7 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

When speech models and a language model share one GPU, can audio meet every deadline while text
keeps most of the throughput, given that a launched language-model step cannot be interrupted?

## Approach

Audio work is periodic with deadlines (a speech-to-text chunk every 20 ms, a text-to-speech chunk
ahead of playback). Language-model decode steps are long and non-preemptible. Earliest-deadline-
first among audio jobs, language model only when no audio is ready, and a cap on the language
model's step length while someone is talking. The cap comes from the processor-demand test for
non-preemptive EDF with blocking: at every deadline `t`, demand plus the longest blocking job with a
later deadline must fit in `t`.

## Verified offline

- 1,500 random stream sets (over 500 admitted): every admitted set ran 2 s of simulated schedule
  with **zero** audio misses under realtime-first; arrival-order scheduling missed on some of the
  same sets.
- The first, simpler bound (`U + B / P_min <= 1`) was **rejected by the random test**: it ignored
  long audio jobs blocking short ones. The demand test replaced it.
- Example: 20 ms / 3 ms speech-to-text plus 40 ms / 8 ms text-to-speech admits language-model steps
  up to 17 ms; at 30 ms audio is late.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Audio deadlines | zero missed audio chunks over the frozen voice protocol | any underrun or late chunk |
| Text throughput | decode tok/s with speech resident reported against solo, all samples | unreported loss |
| Step cap | engine decode step (batch x tokens) bounded by the admitted cap while audio is active | uncapped steps |
| Quality | tokens identical with and without audio sharing (Q-030) | any change |

## Non-claims

- Costs in tests are invented; real costs come from Q-042 timers on the card.
- GPU streams can overlap in hardware; the simulator models one queue, which is conservative.

## Definition of done

1. Step cap enforced by the engine scheduler while audio tenants are active.
2. Measured voice protocol: zero misses, throughput loss reported, tokens unchanged.
