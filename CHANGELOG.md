# Changelog

## 0.9.0-rc1 - 2026-10-23

release-identity and evidence-gate hardening; no inference claim changes.

- publish an exact release identity for the public tree (`docs/RELEASE.md`, tag `v0.9.0-rc1`);
- make the documented offline evaluation runnable: both harness crates are standalone Cargo
  workspaces, so `cargo test --manifest-path ...` works from a fresh clone;
- enforce the offline gates in CI: Rust tests, clippy `-D warnings`, Python unit suites (normal
  and `-O`), documentation links and public-artifact hygiene;
- make audit gates fail closed under `python -O`: all bare `assert` gates in the Q-010/Q-012/Q-013-C
  audit scripts are replaced by `harness/experiments/gates.py::demand`, with a regression test;
- bind `IndependentWitness.observed_digest_sha256` to the SHA-256 of the exact witnessed output
  bytes instead of shape-checking a digest-looking string;
- register the lab `mistralrs-strata` engine fork as `PATCHED_UPSTREAM` with a file inventory and
  exported residency patch (Q-009/Q-012 evidence source);
- record DeepInfer as a `REFERENCE_ONLY` donor (Q-017) with a content digest pin, donor analysis
  and a quest that gates any future port; no donor code is built, imported or vendored;
- add the additive `EngineProvider v1` draft contract and reference module (Q-024a): provider
  registry, typed capability negotiation, start/poll streaming, explicit error taxonomy and
  admission facts, with the v0 trait unchanged;
- freeze the three-way engine-direction benchmark protocol (Q-025) before further engine investment.

## 0.8.0 - 2026-10-07

modeld promotion boundary ratification.

- close Q-008 as OBSERVABILITY_PROMOTED / MATERIALIZATION_REJECTED;
- promote cross-engine runtime observability as an implementation contract;
- promote EngineProvider lifecycle and resource-envelope intent only to a non-production experimental implementation program;
- reject generic supervisor authority over CPU/GPU layers, experts, KV blocks, virtual pages, token batches, and kernel selection;
- ratify semantic readiness, provenance-bearing observations, unload/reclaim separation, and driver-health adjudication as supervisor responsibilities;
- close the original Q-001 through Q-008 research sequence.

## 0.7.0 - 2026-10-07

EngineProvider v0 contract qualification.

- close Q-007 as CONTRACT_V0_QUALIFIED;
- publish an explicit provider lifecycle and semantic-readiness state model;
- separate declared capabilities from evidence-qualified capabilities;
- make UNKNOWN a first-class operational fact with observation provenance;
- separate engine unload acknowledgement from physical reclaim qualification;
- allow supervisor resource envelopes while forbidding generic layer/page/KV placement commands;
- publish a dependency-free Rust reference contract with 7/7 local tests covering mistral.rs/Bonsai, Pulsar, zLLM, and one-shot CLI behavior.

## 0.6.0 - 2026-10-07

UniLLM donor architecture analysis.

- close Q-006 as DONOR_ANALYSIS_COMPLETE;
- validate unillm-kv 15/15 and unillm-scheduler 7/7 unit tests;
- adopt/adapt registry, streaming runner, request state, and strengthened lifecycle/health concepts;
- keep device placement, KV materialization, radix/paged mutation, and token-level batching engine-internal;
- reject current hard-coded/mock operational telemetry as provider facts;
- require UNKNOWN instead of plausible default values when runtime observation is unavailable;
- require semantic-ready and physical reclaim receipts in the future EngineProvider contract.

## 0.5.0 - 2026-10-07

zLLM CUDA placement intake.

- close Q-005 as BLOCKED_MODEL_INVENTORY;
- clean-build zLLM 0.8.11 against local CUDA 13.4 without a compatibility shim;
- qualify Gemma4 E4B semantic readiness on the official standalone HTTP CUDA runtime;
- retain model-registered readiness as stronger than socket/HTTP readiness;
- qualify bounded reclaim and no observed Xid/MMU fault;
- preserve the Qwen3.6/3.8 adaptive-placement claim as untested;
- reject a 16.8–19 GB checkpoint download because it would consume nearly half of the remaining workbench free-space margin.

## 0.4.0 - 2026-10-07

xInfer challenger intake.

- close Q-004 as NOT_QUALIFIED / RUNTIME_BLOCKED;
- preserve clean CUDA 13.4 build failure from the pinned cudarc gate;
- record a separate diagnostic compatibility-shim build that produced real sm_86 kernels without patching xInfer source;
- retain Qwen3.5-9B and Qwen3-8B GGUF load OOMs;
- retain the Phi-4-mini GGUF merged-quantized-weight implementation gap;
- explicitly leave TurboQuant KV, continuous batching, prefix cache, and scheduler quality unproven locally;
- reject xInfer adoption for now rather than converting the research lab into an upstream repair fork.

## 0.3.0 - 2026-10-07

Elastic residency observability qualification.

- close Q-003 with an observability contract supported across three mechanism classes;
- qualify dense mistral.rs/LFM2.5 residency and reclaim under CONTROL;
- reproduce Pulsar Qwen3-30B-A3B Q2_K >VRAM execution with warm host + VRAM cache;
- qualify CUDA VMM stable-address remapping and pagewise map/unmap/release;
- preserve non-comparability of throughput across workload classes;
- reject promotion to a generic materialization contract or new modeld authority.

## 0.2.0 - 2026-10-07

Runtime lifecycle qualification.

- close Q-002 as QUALIFIED_WITH_LIMITS;
- establish HTTP-ready vs semantic-ready;
- qualify CONTROL C1/C4 and retain the C>4 partial-failure boundary;
- qualify streaming TTFT/TPOT under the frozen reference configuration;
- qualify graceful and abrupt process reclaim/restart behavior;
- retain 384-token success and 512+ context OOM as bounded negative evidence;
- preserve DESKTOP-PRESSURE OOM evidence separately from CONTROL.

## 0.1.0 - 2026-10-07

Initial public research baseline.

- establish evidence vocabulary and research method;
- publish the quest-driven execution plan;
- record the qualified Bonsai FG-02 baseline at 35.2 tok/s;
- explicitly reject the unsupported public 40 tok/s FG-03 claim;
- summarize bounded elastic-memory findings;
- establish vendoring/provenance policy;
- establish DCO-style contribution requirements.
