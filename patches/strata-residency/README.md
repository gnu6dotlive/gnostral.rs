# Native expert-residency patches

Research quests: Q-009 (residency), Q-010 (borrowed CPU expert fast path), Q-012 (residency feature evidence).
These are local research patches; none is submitted upstream or claimed upstream-accepted.

## Bases

- Candle base: `f0f334aa82dd808973fd325040baa82053275b22` (MIT / Apache-2.0).
- mistral.rs lab revision: `0bed3b4ce4f490131696ea268e133f952425bcc0` (MIT), branch `lab03-bonsai`.
- mistral.rs **public** base: `24dbf5c256f232176ee5949485ba264049407fbe` (mistral.rs 0.9.3).
- Strata donor: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (MIT), referenced only as ignored `vendor/strata`.

## Patches in this directory

| Patch | Base | Contents |
| --- | --- | --- |
| `candle-q010-cpu-expert-borrowed.patch` | Candle `f0f334aa…` | Q-010 borrowed Q2_K/Q3_K CPU expert blocks (`cpu_expert_matmul_k`) |
| `mistralrs-q010-cpu-expert-borrowed.patch` | mistral.rs `0bed3b4ce…` | Q-010 `compute_host` fast path over the Q-009 residency prototype |
| `20261023-mistralrs-strata-integration.patch` | mistral.rs public `24dbf5c2…` | Complete base→working-tree delta: six lab commits plus the uncommitted 16-file residency integration (1,325 insertions / 10 deletions over the lab revision) |

`20261023-mistralrs-strata-integration.patch` is the reviewable export of the engine tree that produced
the Q-010/Q-012/Q-013-C evidence. Applying it to a clean mistral.rs checkout at
`24dbf5c256f232176ee5949485ba264049407fbe` reproduces the working tree whose per-file SHA-256 list
is recorded in `evidence/runs/mistralrs-strata-vendor-20261023.json` (1,889 files, verified identical;
`git apply` tolerates the provenance preamble at the top of the patch).

The patch is a provenance artefact, not a distribution or a support claim:

- It does not prove numerical parity, speed, reclaim or portability.
- It is not upstream-accepted. Do not report it as merged, and do not rebase it blindly onto a
  different mistral.rs revision.
- Keep the MIT licenses and third-party notices when copying any code. Candle and mistral.rs retain
  their own licenses.
- The build used `CUDA_COMPUTE_CAP=86` for the RTX 3070 under CUDA 13.4; that is a hardware fact,
  not a qualification of an optimized distribution.
