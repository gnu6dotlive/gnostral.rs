# Q-034 — PACKAGING-01

Review date: 2026-10-23
State: **PROPOSED** — no measurement exists; no binary exists.
Authorization: a new `crates/gnostral-serve/` (or a fork of server-core), CLI work, build
reproducibility work and docs. This quest does not authorize a hosted service, telemetry, a
distribution claim, or packaging before the M1–M3 gates.
Workstream: W6 · Milestone: M5 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can the engine ship as **one binary** — `gnostral serve`, OpenAI-compatible, with a model manager,
config profiles, the desktop reserve and reproducible builds — without smuggling in a product claim
the evidence does not support?

Packaging is where an unmeasured engine most easily becomes an unearned claim. The rule is that the
binary reports what it observed and refuses what it cannot verify.

## Deliverable and file seams

- `crates/gnostral-serve/` or a pinned server-core fork; CLI subcommands for serve, model list,
  profile selection, doctor.
- OpenAI-compatible surface limited to what is implemented; unknown endpoints fail closed.
- Model manager: exact paths and hashes, no implicit download, no silent substitution.
- Config profiles derived from measured envelopes, with `UNKNOWN` where a fact is missing.
- Desktop reserve hook from Q-041; Linux target, Windows only if the fork supports it.
- Reproducible build recipe and a recorded build identity.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Single artifact | one command starts the server with a pinned model identity | a runtime dependency fetched at startup |
| Contract honesty | every exposed endpoint is implemented and documented | an endpoint returning placeholders |
| Identity reporting | `/health`-style output separates observed facts from `UNKNOWN` | plausible defaults presented as facts |
| Refusal | missing/unknown model, hash mismatch or unsupported format fails closed | silent fallback to another model |
| Reproducibility | two clean builds produce identical binary hashes, or every difference is explained | unreproducible artifact claimed as reproducible |
| No telemetry | no network egress without an explicit, documented opt-in | silent reporting |
| Measured envelopes only | defaults come from a receipt, not from a guess | a default tuned by feel |
| Portability | claimed only for platforms actually tested | untested platform listed |

## Non-claims

- Packaging does not make the engine fast, stable or production-ready.
- No claim of OpenAI API completeness, multi-tenant serving, authentication or TLS.
- No distribution claim over model weights; the binary ships without weights.
- No Windows, macOS or container support claim unless separately qualified.

## Definition of done

1. One binary builds from a clean tree and serves one pinned model with a documented command.
2. A build-identity receipt and a startup smoke test are retained.
3. The documented surface equals the implemented surface, checked by a test.
