# Public evidence

Only evidence suitable for public release belongs here.

Do not commit credentials or tokens, user data, private host inventory unrelated to the result, raw model weights, PID/socket state, or giant profiler traces without a publication need.

Prefer compact machine-readable summaries plus checksums for externally hosted or reproducible large artifacts.

Q-012's sanitized, SHA-256-attributed [three-family functional qualification](runs/gnostral-q012-native-three-families-20261010.json) includes the negative MoE feature-omission witness, per-family bounded memory observations, and the digest of each retained private source receipt. Its raw logs and embedding vectors are intentionally not committed.

Q-013-A [evidence-bound probe plans](runs/gnostral-q013-exact-probe-plans-20261010.json) are derived from the pinned Q-012 public receipt; every plan has `execution_authorized=false`. They are offline contract fixtures, not new runtime qualifications or executable access grants.

Q-013B's [CPU fixture containment receipt](runs/gnostral-q013b-fixture-process-20261010.json) contains eight bounded-process observations, an actual negative outside-scope check, and digests of private source logs. It does not qualify GPU reclaim, real inference, arbitrary executable admission or production supervision.

Q-013C's [real-dense single-server receipt](runs/gnostral-q013c-real-dense-single-server-20261010.json) is from an actual bounded Qwen2.5-Coder inference and an independent local rehash/oracle audit. It excludes the privately retained engine log and full response. It does not qualify multi-family routing, authenticated third-party semantic witnesses, or production.

## Engine program (from 2026-10-23)

Context and interpretation for each of these is in [docs/FINDINGS.md](../docs/FINDINGS.md); a receipt here does not by itself support a performance claim.

- [Vendored engine inventory](runs/mistralrs-strata-vendor-20261023.json): the engine tree the patch series applies to (public mistral.rs `24dbf5c` plus the strata integration patch, 1,889 files byte-identical).
- [Synthetic end to end, CPU](runs/synthetic-moe-e2e-20261023.json): random 2-layer Qwen3-MoE fixture; executor and kernel token parity, trace neutrality, bench plumbing (F-001, F-002). No timing.
- [Synthetic end to end, CUDA](runs/synthetic-moe-e2e-gpu-20261023.json): random 8-layer fixture on the lab RTX 3070; parity on every run, whole-card VRAM, range-based lane ordering on a noise model (F-007). Layer placement was not recorded.
- [Reproducibility matrix, Qwen3-30B-A3B](runs/m1-repro-diag2-20261024/repro.md): same prompts captured twice per configuration; divergence follows the number of layers placed on CUDA (F-010).
- [Kernel microbenchmarks against ggml, cloud VM](runs/kernel-microbench-cloud-20261024/summary.json): expert FFN per layer, thread-pool dispatch and per-block dot costs next to llama.cpp `test-backend-ops` at the Qwen3-30B-A3B geometry (F-021 to F-023). Not the lab host.
- [Four-lane plan v2 run, CPU](runs/synthetic-moe-four-lane-cpu-20261024/summary.json): `per-route`, `grouped`, `gnostral` and llama.cpp on the synthetic fixture; every run kept its pinned placement and reproduced its lane's warmup; comparison refused for driver `UNKNOWN` (F-016, F-020). The fixture itself is not retained; it regenerates from the recorded generator and seed.
