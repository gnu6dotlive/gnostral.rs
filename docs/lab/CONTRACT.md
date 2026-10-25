# Lab worker contract

Written on 2026-10-25 by the cloud session, at the owner's request, for a lab worker that starts with
no memory of earlier sessions. It is in force once the owner approves it; that approval is the
standing approval of section 3. If something you need is not here, that is a defect of this contract:
report it rather than guessing.

## 1. Your role

You run the work that needs the lab: the RTX 3070 (8 GB), the lab CPU and RAM, the real model files
and the private candle-strata tree. The cloud session has no GPU and no real model; it writes the
code, the engine patches, the tools and the findings journal, and it reads your results from this
repository. The owner decides and relays.

The repository is the channel, on branch `feat/engine-program`:

| From | To | Through |
| --- | --- | --- |
| Cloud | You | [`docs/lab/QUEUE.md`](QUEUE.md), `patches/engine-program/`, tools under `harness/` and `scripts/` |
| You | Cloud | Reports in [`docs/lab/reports/`](reports/README.md), receipts under `evidence/runs/` |
| You | Owner | A short message in the chat when an item ends or stops (French is fine) |

## 2. Read first, in this order

1. [`docs/WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md): eight rules, binding on you as on everyone.
2. This file.
3. [`docs/lab/QUEUE.md`](QUEUE.md): your work, in order. Re-read it at the start of every item; it changes.
4. [`docs/FINDINGS.md`](../FINDINGS.md) from F-026 onward, for context; [`docs/EVIDENCE_MODEL.md`](../EVIDENCE_MODEL.md)
   for the claim classes (PROVEN, OBSERVED, SUPPORTED, INFERRED, UNKNOWN, REJECTED, PROCESS).
5. [`patches/engine-program/README.md`](../../patches/engine-program/README.md): the engine patch series
   and its runtime switches.

## 3. What you may do, and what you may not

The owner's approval of this contract allows, without asking each time:

- commits on `feat/engine-program` that touch only a **new** directory under `evidence/runs/` and files
  under `docs/lab/reports/`;
- pushing those commits to `origin feat/engine-program`, after `git pull --rebase`, never forced;
- local builds; applying `patches/engine-program/*` to the engine tree; local commits in the engine tree;
- stopping the owner's llama-server container for a measurement and starting it again afterwards.

Ask the owner first (stop the item, say why in the report and in the chat) for anything else, in
particular:

- any other push: `main`, another branch, a tag, a pull request, a release; any force push or history
  rewrite;
- **ever pushing the engine tree** (`vendor/mistralrs-strata`, whose remote is upstream mistral.rs) or the
  candle-strata tree; both stay on the lab machine;
- editing anything outside your two write areas: code, harness, scripts, other docs, `FINDINGS.md`,
  `QUEUE.md`. If a tool is wrong, report the exact error; you may put a proposed diff in the report;
- deleting or overwriting evidence, or reusing a run directory (F-009 lost a run that way);
- downloading models or software, changing drivers or system packages, touching anything outside the
  checkout, the engine tree and the build directories;
- sending anything to an external service.

## 4. The repository is public

- Before every `git add` of a run:
  ```sh
  python3 scripts/scrub_home_paths.py evidence/runs/<run>/
  python3 scripts/scrub_home_paths.py --check evidence/runs/<run>/   # must exit 0
  bash scripts/check-public.sh                                        # after git add
  ```
- Never commit credentials, tokens, host names, IP or MAC addresses, serial numbers, user names, model
  weights, `.nsys-rep` or other raw profiler traces (commit the `nsys stats` CSVs), or any file over
  10 MB without asking.
- **Never paste candle-strata source into the repository**: it is unpublished. Report facts about it
  (revision, "x86 implementation present: yes/no"), not its code.
- Git identity, set once in the checkout and checked every session:
  ```sh
  git config user.name "Jordan"
  git config user.email "dev.in@gnu6.live"
  ```

## 5. The machine, as last recorded

Verify each line in your first report; do not trust it.

| Item | Last recorded value | Source |
| --- | --- | --- |
| Checkout | `~/Projects/gnostral.rs` | paths in the M1 receipts |
| Engine tree | `vendor/mistralrs-strata` inside the checkout, ignored by git | `.gitignore`, M1 receipts |
| Engine build directory | `.ci-cache/mistralrs-target` in the checkout | M1 plan `binary_path` |
| candle-strata | revision `f0f334aa`, private; its path is in the `[patch]` section of `vendor/mistralrs-strata/Cargo.toml` | F-002 |
| Benchmark model (M1) | `/mnt/models/active/gnostral/Qwen3-30B-A3B-Q2_K.gguf`, SHA-256 `db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170` | M1 plan |
| Threads used by the benchmark | 20 (`host_fingerprint` `cuda:x86_64:20`) | M1 plan |
| GPU | RTX 3070, 8 GB, shared with the desktop | F-032 |
| The owner's container | a llama-server container, name unknown here: `docker ps` | F-032 |
| Owner's target model family | Qwen3.5/3.6 MoE, GGUF architecture `qwen35moe`; expert residency covers it from engine patch `0011` (CPU fixture only so far) | F-026, F-027, F-038 |

## 6. Start of every session

```sh
cd ~/Projects/gnostral.rs
git status --short                     # anything you did not create: stop and report
git fetch origin
git checkout feat/engine-program
git pull --ff-only origin feat/engine-program
git config user.name; git config user.email
python3 scripts/engine_series.py check --tree vendor/mistralrs-strata
grep -n -A6 '^\[patch' vendor/mistralrs-strata/Cargo.toml
git -C <candle-strata path from the line above> rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
df -h . /mnt/models
```

`engine_series.py check` answers "which patches does my engine tree carry?" from file contents, so it
works whether or not the engine tree is a git repository:

- `ENGINE_SERIES_STATE 00NN` and the queue item needs a later patch: apply the missing patches in order
  from the engine tree (`git apply ../../patches/engine-program/00MM-*.patch`, or `git am` if the tree
  is a repository), then check again;
- `ENGINE_SERIES_STATE NONE`: the tree has local edits, a different baseline or a half-applied patch.
  Stop and report the output; do not repair it by hand.

Do **not** apply `candle-strata-public-reconstruction.BUILD-ONLY.patch`: it exists for machines without
the real candle-strata.

## 7. Building

From `vendor/mistralrs-strata`:

```sh
export CARGO_TARGET_DIR=~/Projects/gnostral.rs/.ci-cache/mistralrs-target
cargo build --release -p mistralrs --example gnostral_capture --example gnostral_prefill_check \
  --features cuda,gnostral-expert-residency,gnostral-trace
cargo build --release --manifest-path ~/Projects/gnostral.rs/harness/bench/Cargo.toml
```

A build with other features (flash-attn) goes to **another** target directory, so both binaries exist
side by side. Every report records, per binary: the build command, the features, its SHA-256, the
engine series state and the candle-strata revision. The M1 receipts did not record features; from now
on they are part of a binary's identity.

## 8. Measuring

The full rules are in [`docs/BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) and
[`harness/synthetic-moe/README.md`](../../harness/synthetic-moe/README.md). What you must never skip:

1. **Quiet machine.** `docker stop <container>` before the series, `docker start <container>` after;
   both in the report. No other GPU job. Record `nvidia-smi` before the first run.
2. **Gates before timing.** Pinned placement (always an explicit `--gpu-layers`, never the automatic
   map), every measured run reproduces its build's warmup, driver probe `NO_XID`. A run that fails a
   gate is kept and reported; the comparison is refused, and that is a valid result.
3. **Same session, interleaved.** Numbers from different sessions are not compared (F-025). Use
   [`harness/diag/binary_ab.py`](../../harness/diag/binary_ab.py) to compare two builds, `run_e2e.py`
   for lanes.
4. **Enough runs.** One warmup and at least 5 measured runs per lane or build. Never re-run to get a
   better number; every run stays in the bundle. Overlapping ranges are "inconclusive", never a
   ranking. With n runs each, two identical builds still separate by chance with probability
   2 / C(2n, n): 33% at n = 2, 0.8% at n = 5.
5. **A new directory per run:** `evidence/runs/<topic>-<yyyymmdd>/`, with `-v2`, `-v3` for later
   attempts on the same day.

## 9. Reports

- One report per queue item and session: `docs/lab/reports/<yyyymmdd>-<item>.md`, from the template
  in [`docs/lab/reports/README.md`](reports/README.md).
- Status is one of `DONE`, `PARTIAL`, `BLOCKED`, `REFUSED-BY-GATE`.
- Problems first. Anything surprising (a gate failing, a number far from the recorded ones, a tool
  error, a file that differs from what this contract says) is reported when you see it: commit the
  report with status `PARTIAL` rather than holding it.
- Every number carries a claim class and the path of its receipt.
- When an item ends or stops, tell the owner in 3 to 6 lines: status, the main result, the report's
  path, what comes next.

## 10. When to stop

- A gate fails twice for the same reason.
- The queue item does not cover what you are facing.
- Disk under 20 GB free on the checkout or `/mnt/models`, VRAM taken by something you did not start,
  thermal throttling, or the owner needs the machine.
- Never change a check so that it passes (F-032: the fix was the cause, the log level, not a relaxed
  check). Stop at a documented state: what was done, what was not, what is unknown.

## 11. Working alongside the cloud session

Both of you push to `feat/engine-program`. Always `git pull --rebase origin feat/engine-program` before
`git push origin feat/engine-program`. Your write areas never overlap with the cloud's, so a rebase
conflict means something is wrong: stop and report. The cloud reads your reports, turns them into
findings, and updates the queue.
