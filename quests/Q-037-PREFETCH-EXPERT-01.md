# Q-037 — PREFETCH-EXPERT-01

Review date: 2026-10-23
State: **PROPOSED** — stretch goal; no measurement exists; no code exists.
Authorization: only after a profile shows upload stalls dominate batch decode, and only for the
router-logit-based next-step prefetch policy. This quest does not authorize speculative execution of
an expert whose weights are absent, a new upload architecture, or an early start before Q-026/Q-029
land.
Workstream: W1 (stretch) · Milestone: M6 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

In batch decode, can the router's top-k logits predict the next step's experts well enough that
prefetching them during the current step removes an upload stall that is actually there?

The precondition is the question. If profiling does not show upload stalls dominating, this quest
does not start. A negative result is the expected outcome and is a valid completion.

## Deliverable and file seams

- Profile first: per-step upload time from Q-042 timers on the reference workload; a recorded
  decision to proceed or stop.
- If proceeding: a prefetch policy consuming the current step's router logits, issuing upload tickets
  for the predicted set on the copy stream, with lease and eviction rules from Q-027.
- No change to routing: predictions affect transfer timing only, never which experts compute.
- Benchmarks replay the frozen protocol; evidence under `evidence/runs/`.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Precondition | Q-042 evidence shows upload stalls are a material share of step time | prefetch written before the profile exists |
| Semantic neutrality | predicted prefetch changes no token and no routing decision | any token difference |
| Prediction receipt | hit rate of predicted experts recorded per step | policy applied without measuring its accuracy |
| Net effect | batch-decode TPOT improves beyond the noise band, with VRAM inside the envelope | a regression, or a gain only in a synthetic harness |
| Waste accounting | prefetched-but-unused bytes reported | hidden bandwidth cost |
| Stop rule | if prediction accuracy or gain is poor, publish the negative and leave prefetch off | enabling it anyway |

No numeric threshold is asserted here beyond the envelope: the M1/M2 gates own the numbers, and this
quest may not restate them as results.

## Non-claims

- No prefetch gain, hit-rate or bandwidth number exists.
- Prediction accuracy from one model/route distribution does not transfer to another model.
- Prefetch is not speculation: no draft tokens, no verification, no acceptance policy.
- A negative result is not a failure of the program; it closes the hypothesis.

## Definition of done

1. A profile receipt that either stops the quest or justifies it.
2. If justified, a measured batch-decode comparison with prediction hit rate, wasted bytes and VRAM,
   all samples retained.
3. Either a gain inside the envelope, or a published negative with the feature off by default.
