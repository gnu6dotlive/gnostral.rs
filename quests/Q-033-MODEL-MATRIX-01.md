# Q-033 — MODEL-MATRIX-01

Review date: 2026-10-23
State: **PROPOSED** — no measurement exists; no receipt exists.
Authorization: loader/format work and functional qualification runs on the lab host. This quest does
not authorize a new model architecture before the quality gate passes, a multimodal feature, or any
performance claim from a functional pass.
Workstream: W6 · Milestone: M5 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

After the engine is fast and correct on the reference MoE model, which model families and formats can
it serve with an explicit, per-family functional receipt: dense, embeddings, Qwen3 MoE,
Qwen3.8-Flash-Next/GDN, and Q4/IQ2/IQ3 quantization?

Functional support is not production readiness, and one family's pass does not transfer to another.

## Deliverable and file seams

- Loader/format support for each family; per-family qualification profile with exact binaries,
  weights, tokenizer and prompt hashes.
- ASCII/unicode-safe sentinel oracle per family, plus a dense embedding triple, reusing the Q-012
  pattern.
- `evidence/runs/` receipts per family and format; `docs/CAPABILITY_MATRIX.md` updated only after a
  receipt exists.
- GDN/hybrid attention wiring where the fork supports it; multimodal only if it does not compromise
  the core.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Per-family function | exact oracle output for each family, with independent hashes | a family claimed from a different family's run |
| Format matrix | each claimed quantisation loads and produces the oracle output | format claimed from a load-only test |
| Sequential honesty | families tested in separate processes unless co-residency is itself measured | implying simultaneous residency |
| Tokenizer parity | tokenizer identity hashed and checked | unverified tokenizer |
| Unsupported is explicit | an unsupported family or format fails closed with a typed error | silent fallback to another model |
| No performance inheritance | functional passes carry no throughput claim | any tok/s claim from this quest |
| Quality gate | Q-030 stays green for every family added | adding a family while tokens diverge |

## Non-claims

- No co-residency, hot-swap, multi-model serving or production stability claim.
- No performance claim of any kind; functional pass ≠ fast.
- No claim about GPUs, drivers or platforms other than the recorded host.
- Multimodal support is not promised; it is conditional and may be refused.

## Definition of done

1. One receipt per claimed family and format, each replayable from pinned identities.
2. A capability table that lists exactly the qualified combinations and nothing more.
3. Explicit typed refusal for everything not qualified.
