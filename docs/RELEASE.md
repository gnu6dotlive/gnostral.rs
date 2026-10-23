# Release identity

**Current candidate:** `v0.9.0-rc1` · **Date:** 2026-10-23 · **Base commit:** `38c4134`
(merge of `research/upstream-intake-20261021` into `main`).

This page is the canonical answer to "which exact tree is this?". It exists because the
repository's claims are only as good as the artifact a reader can obtain.

## Canonical repository

Both URLs below resolve to the same refs (checked with `git ls-remote` on 2026-10-23):

- `https://github.com/tension-atoi/gnostral.rs` (used in `README.md` and `CITATION.cff`)
- `https://github.com/gnu6dotlive/gnostral.rs` (working remote in the lab checkout)

If they ever diverge, this page must name one canonical URL and the other a mirror.

## Obtain exactly this tree

```bash
git clone https://github.com/tension-atoi/gnostral.rs.git
cd gnostral.rs
git fetch --tags
git checkout v0.9.0-rc1
git rev-parse HEAD        # must equal the tag target recorded below
```

`main` at the time of this note is `38c4134a3a398b5b83d34326826362cf4dcc7336`. The tag is created on
the release commit after this page; `git rev-parse v0.9.0-rc1^{commit}` is authoritative. The
attachment exported by GitHub on 2026-10-21 carries embed comment `38c4134…` and its tree equals
`main@38c4134` (verified with `git archive` + `diff -rq`); it is **not** a branch export.

## What the tag is checked by

`scripts/check-release-identity.sh` fails unless all of these hold:

1. the checkout is a Git repository with a clean working tree;
2. `HEAD` is exactly tagged, and the tag equals `v` + `CITATION.cff.version`;
3. the entrypoint files that make the release usable exist:
   `docs/CAPABILITY_MATRIX.md`, `docs/START_HERE.md`, `harness/q013b-local-adapter/`,
   `evidence/runs/gnostral-q012-native-three-families-20261010.json`,
   `evidence/runs/gnostral-q013c-real-dense-single-server-20261010.json`;
4. the offline evaluation commands in `README.md` name the same crates that CI runs.

It does **not** re-run the GPU experiments. A tag is a pointer to bytes, not a proof of performance;
the evidence rules in `docs/EVIDENCE_MODEL.md` are unchanged.

## Known provenance states at this release

| Item | State |
| --- | --- |
| Public source of the Q-010 fast path (`67040aa`) | **not pushed** at the time of writing; exported as a bundle with this release, or push `feat/q010-cpu-moe-hotpath-20261010` |
| Lab engine fork `mistralrs-strata` | registered in `manifests/upstreams.toml` as `PATCHED_UPSTREAM`; tree stays in ignored `vendor/` |
| DeepInfer | `REFERENCE_ONLY` content pin (`revision = UNRESOLVED`, archive + tree digests); no build, no import |
| Local checkout `main` diverged from GitHub `main` | stale local ref; fast-forward after `git fetch`. The release branch is based on GitHub `main`, not on the stale ref |

## Tagging procedure (maintainer)

```bash
# after the release commits are green locally and reviewed
git checkout main
git merge --ff-only <release-branch>     # the release branch is based on GitHub main
git tag -s v0.9.0-rc1 -m "gnostral.rs v0.9.0-rc1"
git push origin main v0.9.0-rc1
```

Do not use GitHub Actions to discover whether the tree is green. Run the local gate suite first:
`cargo test`/`clippy` for the two standalone harness crates, the Python suites in normal and `-O`
mode, `scripts/check-doc-links.py`, and `scripts/check-public.sh`.
