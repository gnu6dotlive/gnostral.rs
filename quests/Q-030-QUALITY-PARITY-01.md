# Q-030 — QUALITY-PARITY-01

Review date: 2026-10-23
State: **PROPOSED** — oracle, frozen 32-prompt set, frozen thresholds, divergence ledger and gate receipt landed; capture tool `gnostral_capture` (engine patches 0002, 0005) exercised on the synthetic fixture and on Qwen3-30B-A3B for diagnostics, which showed the engine did not reproduce itself across loads ([F-009, F-010](../docs/FINDINGS.md)); no gate receipt on a real model exists.
Authorization: a new harness under `harness/bench/` with schemas under `schemas/` and receipts under
`evidence/runs/`; one GPU session to capture reference tokens. This quest does not authorize changing
sampling, routing, quantization or the tokenizer to make a gate pass.
Workstream: W5 · Milestone: M0 (scaffold) → M1+ (gate) · [Program](../docs/ENGINE_PROGRAM.md)

## Question

What is the smallest quality oracle that can block an engine change **before** its performance is
recorded: greedy token parity on fixed prompts, KL/perplexity where logits are comparable,
batch-invariance and sampler checks, and long-context parity — against a reference engine on
identical artifacts?

This is the gate every engine PR must pass. It is built before the executor work lands.

## Deliverable and file seams

- `harness/bench/quality/` — prompt-set freeze by hash, reference token capture, comparison and
  divergence ledger.
- `schemas/` — receipt schema for parity runs; `evidence/runs/` — retained receipts.
- Reference lane: llama.cpp on the same GGUF bytes (a reference engine, not a target).
- Batch-invariance and PIT/KS checks (sampler gates) usable without a GPU where the sampler permits.
- Long-context parity: same tokens as the short-context baseline at the frozen tiers.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Prompt freeze | ≥ 32 prompts, exact bytes hashed before any run; set may not change after results are seen | prompts added/removed to move a number |
| Greedy parity | token agreement ≥ 99.5 % vs the reference run | below the threshold |
| KL gate | KL ≤ 1e-3 where logits are comparable | above the threshold, or "not comparable" used to skip it |
| Divergence ledger | every divergent position explained (documented kernel or block-order difference) | any unexplained divergence — hard fail |
| Batch invariance | same tokens for batch 1/2/4 neighbours, greedy and one sampled setting | any difference |
| Sampler gate | randomized PIT + KS against U(0,1) on sampler probabilities | gate fails |
| Long context | exact same tokens as the short-context baseline | any drift |
| Harness honesty | thresholds and prompt set are frozen before the first run and recorded in the receipt | thresholds tuned after seeing results |

## Non-claims

- No quality result exists; the thresholds are gates copied from the program scorecard, not findings.
- Agreement with one reference engine on one GGUF is not general quality parity.
- Passing this harness does not prove the engine is stable, fast or production-ready.
- The harness cannot detect a quality loss that both engines share (for example a bad prompt set);
  the prompt-set freeze limits, but does not remove, that risk.

## Definition of done

1. Prompt set and thresholds frozen by hash and recorded before the first comparison.
2. Reference tokens captured once and stored with the exact binary and model identity.
3. One real engine comparison with a divergence ledger, all samples and all failures retained.
4. The gate is wired so an engine PR that changes tokens without an explanation fails.
