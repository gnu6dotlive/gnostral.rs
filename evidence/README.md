# Public evidence

Only evidence suitable for public release belongs here.

Do not commit credentials or tokens, user data, private host inventory unrelated to the result, raw model weights, PID/socket state, or giant profiler traces without a publication need.

Prefer compact machine-readable summaries plus checksums for externally hosted or reproducible large artifacts.

The repository is public: before `git add`, run `python3 scripts/scrub_home_paths.py evidence/runs/<new-run>/` so receipts, plans and captured output carry `~` instead of a personal home path. `scripts/check-public.sh` fails on any new tracked file that still holds one; the files committed before this rule are listed in `scripts/home-path-allowlist.txt` and left as they are.

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
- [M1 cpu-experts comparison, lab](runs/m1-qwen3-30b-cpu-experts-v3-20261024/bench-cpu-experts/pareto.md): first accepted M1 comparison; per-route, grouped, gnostral and llama.cpp `--cpu-moe`, placement pinned, each lane self-reproducing (F-032).
- [M1 decode profile, lab](runs/m1-profile-20261024/trace-gnostral.json): per-phase timeline of one traced gnostral request (F-033).
- [Placement and reproducibility with 0007, lab](runs/m1-repro-diag3-20261024/A-plain/repro.md): 48/48 layers on the card, 0/9 divergences, unchanged under 1.5 GB of VRAM pressure (F-031).
- [M1 mixed configuration, lab](runs/m1-qwen3-30b-mixed-20261024/bench-mixed-experts/pareto.md): 2 GiB expert cache from an out-of-sample profile, with the llama.cpp lane (F-034).
- [Per-layer attention sync A/B, lab](runs/m1-sdpa-sync-ab-20261024/): six interleaved captures with and without the naive-attention sync, byte-identical (F-035).
- [qwen35moe fixture judged against a float64 reference](runs/qwen35moe-fixture-oracle-20261024/summary.json): self-consistency of both engines, teacher-forced judgment of the engine and llama.cpp on F32, Q2_K/Q3_K and split-expert variants, with and without llama.cpp repacking (F-028 to F-030).
- [Vectorised `CandleCompat`, before and after, cloud VM](runs/kernel-candlecompat-vectorised-20261024/summary.json): per-block dot and per-layer expert FFN, both binaries interleaved in one session, plus the end-to-end bit-identity check against the four-lane run (F-024, F-025). Not the lab host.
- [Four-lane plan v2 run, CPU](runs/synthetic-moe-four-lane-cpu-20261024/summary.json): `per-route`, `grouped`, `gnostral` and llama.cpp on the synthetic fixture; every run kept its pinned placement and reproduced its lane's warmup; comparison refused for driver `UNKNOWN` (F-016, F-020). The fixture itself is not retained; it regenerates from the recorded generator and seed.
