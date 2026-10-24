# Q-046 — DRAFT-SPEC-01

Review date: 2026-10-24
State: **PROPOSED** — greedy speculation oracle and draft-length policy landed offline (`crates/gnostral-speculative`, 5 tests); no draft model run.
Authorization: offline oracle and policy; any engine speculation must pass Q-030 parity against
plain greedy and follow Q-018 (sampler determinism) before Q-019 (MTP width).
Workstream: W7 (with W3) · Milestone: M7 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Does a small draft model (about 1 GB) earn its memory on a shared desktop card: more tokens per
large-model pass at identical output, and parked by the broker (Q-043) when it does not pay?

## Deliverable and file seams

- `gnostral-speculative::reference`: draft/verify loop over abstract models. Each emitted token is
  the target's greedy choice, so output equals plain greedy for any draft; tested over 40 targets x
  4 draft accuracies x 4 draft lengths, plus an adaptive length.
- `gnostral-speculative::policy`: acceptance tracker (EWMA) and the cost model
  `E(k) / (1 + k c)`; `k = 0` when no length pays; `keep_draft` answers whether the broker should
  keep the draft resident.
- The vendored engine already has MTP, DFlash and draft proposers/verifiers (`speculative/`); this
  quest gates and steers them, it does not replace them.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Exactness | greedy output identical with speculation on and off on the Q-030 prompt set | any token change |
| Gain | measured tok/s gain over the draft's own memory cost, all samples | gain claimed from the model alone |
| Self-disable | policy turns speculation off where acceptance is low, with receipts | speculation kept at a loss |

## Non-claims

- No acceptance rate or speedup has been measured; test acceptances are synthetic.
- The independence assumption in the cost model is a simplification for choosing `k`.

## Definition of done

1. Engine speculation path checked against the oracle on the frozen prompt set.
2. Measured acceptance and speedup per draft model, and the broker's park decision driven by it.
