# Start here · gnostral.rs

**Rust-first inference research for the GPU you actually own.**
*Provenance-aware runtimes, resource-constrained inference, low-bit kernels, and local supervision.*

> **FR — En une phrase :** gnostral.rs explore comment faire fonctionner des modèles modernes sur un GPU contraint, en Rust, avec des preuves auditables de performance, de disponibilité, de placement et de récupération des ressources. C'est un **laboratoire ouvert**, pas encore un service multi-modèle prêt à installer.

> **EN — In one sentence:** gnostral.rs researches how to run modern models on constrained hardware through Rust-first execution paths and auditable evidence for performance, placement, lifecycle, and resource reclaim. It is an **open engineering lab**, not a production multi-model server.

## Navigate by intent

| I want to… | Read |
| --- | --- |
| Understand what works right now | [Capability matrix](CAPABILITY_MATRIX.md) |
| See the full engineering sequence | [QUESTLOG](../QUESTLOG.md) |
| Know the next implementation milestones | [Roadmap](ROADMAP.md) |
| Build the best engine for constrained hardware | [Engine program](ENGINE_PROGRAM.md) |
| Understand the boundary between engines and supervisor | [Architecture and authority](ARCHITECTURE.md) |
| Run tests without a GPU | [Reproducibility guide](REPRODUCIBILITY.md) |
| Inspect primary measurements and negative results | [Evidence index](../evidence/README.md) |
| See current upstream signals | [Rust inference watch](../research/runtime/RUST-INFERENCE-WATCH-20261010.md) |
| Compare the three native model families | [Q-012 qualification](../research/runtime/GNOSTRAL-Q012-NATIVE-CAPABILITIES-20261010.md) |

## Four observed milestones

| Milestone | Observation | Boundary |
| --- | --- | --- |
| Native MoE CPU fast path | 9.51 → 20.18 tokens/s, median, 3 paired runs per lane | Only the compared Qwen3/Strata workload, not all models |
| Load-time memory-aware mapping | 22/21/21 GPU layers baseline; 13/12/13 under VRAM reservation | No layer migration while serving |
| Same CUDA executable, three families | Dense + embeddings + MoE functionally qualified in three **sequential** server processes | No co-residency or production stability |
| Bounded real dense process | Exact output `GNOSTRAL`; engine and weights rehashed; GPU memory after stop +8 MiB ambient | One model, one short-lived server, not an arbitrary serving API |

Every number has a separate report, protocol, and explicit limitations. Source code may demonstrate a mechanism without proving its system-wide benefit.

## What code is actually here?

- A lab-owned Rust expert-residency crate and audited experimental patches for the local Q2_K/Q3_K MoE path.
- An engine-neutral **EngineProvider v0** contract for capability, readiness, lifecycle, unload and reclaim.
- An evidence-bound selector that produces SHA-pinned **plans, never execution grants**.
- A bounded Rust subprocess **fixture** harness with exact executable identity, cgroup verification, timeout and group cleanup.
- A restricted Python **real-engine dense pilot** and an independent auditor.
- Python/Rust test oracles, source-provenance manifests, benchmark reports and machine-readable public receipts.

**Not bundled:** upstream inference runtimes, third-party model weights, local credentials, CUDA binaries, production services, a generic control plane, or automatic multi-model hot swap.

## Thirty-second evaluation (no GPU, no deployment)

```bash
git clone https://github.com/tension-atoi/gnostral.rs.git
cd gnostral.rs
# No GPU or weights needed:
cargo test --offline --locked --manifest-path harness/engine-provider-contract/Cargo.toml
python3 -m unittest discover -s harness/experiments -p 'test_q013c*.py'
```

Offline Rust tests require their dependencies to be cached. See [reproducibility](REPRODUCIBILITY.md) for safe fallbacks, environment and qualification boundaries. **Do not run a real CUDA experiment without isolated local resource limits.**

## Quality bar

Negative runs are kept. Upstream commit messages are not treated as benchmarks. A PASS is valid only for the stated hardware, model, exact binary, frozen inputs, observed lifecycle and independent gate. Unmeasured quantities stay UNKNOWN.

Built independently with upstream attribution. See [licensing/provenance](../THIRD_PARTY.md) and [contribution rules](../CONTRIBUTING.md).
