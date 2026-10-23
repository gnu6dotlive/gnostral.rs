# Third-party material

gnostral.rs does not claim ownership of upstream projects studied by the lab.

| Project | Role in research | Local status |
|---|---|---|
| mistral.rs | Rust inference/runtime reference and Bonsai integration target | upstream plus experimental local patches |
| Candle | tensor/kernel substrate used by local experiments | upstream plus experimental local patches |
| candle-vLLM | Rust serving/runtime candidate | upstream probe / donor analysis |
| Pulsar | oversubscribed MoE and residency reference | upstream experimental track |
| xInfer | scheduler/KV compression challenger | planned evaluation |
| zLLM | placement/runtime challenger | planned evaluation |
| UniLLM | unified runtime/model abstraction donor | planned evaluation |
| Strata (Niko1221) | expert residency, CPU/GPU MoE split and chunked prefill donor for the Rust engine | pinned local source snapshot; no local build or inference qualification |
| mistralrs-strata | lab engine fork (mistral.rs 0.9.3 + Bonsai/lab commits); source behind Q-010/Q-012/Q-013-C | patched upstream; public base plus exported patch; local-only vendor tree |
| DeepInfer (DeepSeek-AI) | scheduling, sampling-determinism, confidence-scheduled MTP verification and structured-output design donor; datacenter reference, not a runtime candidate | reference-only source analysis; not vendored, not built, not imported, no inference |

Actual vendored code must carry its upstream license in-tree and be registered in manifests/upstreams.toml.

The root MIT license applies only to original gnostral.rs material unless a file or subtree states otherwise.

Strata is pinned to `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` in ignored
`vendor/strata/`. Its MIT license remains verbatim at `vendor/strata/LICENSE`;
dependency notices, including `third_party/ggml/LICENSE` and SYCL notices, are
retained. The snapshot excludes the upstream experimental speed-projection
GGUF. Its exact file inventory, exclusions and source archive digest are in
`evidence/runs/strata-vendor-20261008.json`. Ported or copied code must retain
the applicable Strata and dependency notices; no port has landed yet.

`mistralrs-strata` is the lab's mistral.rs fork (workspace version 0.9.3,
MIT, copyright Eric Buehler) based on public revision
`24dbf5c256f232176ee5949485ba264049407fbe`, with six local commits and an
uncommitted 16-file residency integration on top. It lives only in the ignored
`vendor/gnostral-strata-worktree/` and is **not** published. The exact
base→working-tree delta is exported as
`patches/strata-residency/20261023-mistralrs-strata-integration.patch`; applying
it to a clean checkout at the public base reproduces the tree whose per-file
SHA-256 list is in `evidence/runs/mistralrs-strata-vendor-20261023.json`
(verified identical). Nothing here is upstream-accepted: do not describe the
lab commits as mistral.rs releases, and retain the MIT license and notices when
copying code.

DeepInfer is recorded as a **reference donor only** — no source tree is vendored.
Its public archive is pinned by content, not by version control: the snapshot
carries no Git history, so its revision is recorded as `UNRESOLVED` and the bytes
are pinned by archive sha256
`a2f6f0a42e7bb213bb9e30fe997b6778a467829a4d71b94e783d0e6f6fa8af3d` plus a
deterministic tree digest, listed in
`evidence/runs/deepinfer-reference-20261023.json`. Apache-2.0 LICENSE is retained
verbatim where any excerpt is retained; the archive contains no NOTICE file and
no root-level copyright header, and two source files carry third-party
vLLM/EleutherAI/HuggingFace copyright lines. Six upstream Git submodules
(DeepGEMM, DeepEP, FlashMLA, DeepSelect and their Ascend variants) are **absent**
from the archive, so no complete dependency provenance can be claimed.
DeepInfer cannot run on this project's reference hardware (Python 3.12 + torch +
CUDA 13, NVIDIA SM100 / Ascend 950, DeepSeek-V4.1-Flash only), and its C++ sources
are a torch extension, not a portable C++ engine. Any algorithm ported from it is
gnostral-owned Rust code carrying an explicit `Derived from deepinfer/<file>,
Apache-2.0, (c) DeepSeek-AI` header and is **not** relicensed under the root MIT
license. No port has landed yet; the root MIT license does not cover DeepInfer
material.
