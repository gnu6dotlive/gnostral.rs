#!/usr/bin/env bash
# Fail closed when the public artifact is not the exact tagged release it claims to be.
# This gate checks bytes and identity; it never re-runs a GPU experiment.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail=0
say() { printf '%s\n' "$*"; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  say "RELEASE_GATE_FAIL: not a Git checkout (a source export cannot prove release identity)"
  exit 1
fi

version="$(awk '/^version:/ {print $2; exit}' CITATION.cff)"
[ -n "$version" ] || { say "RELEASE_GATE_FAIL: CITATION.cff has no version"; exit 1; }
expected_tag="v${version}"

if ! tag="$(git describe --tags --exact-match HEAD 2>/dev/null)"; then
  say "RELEASE_GATE_FAIL: HEAD is not exactly tagged (expected ${expected_tag})"
  fail=1
elif [ "$tag" != "$expected_tag" ]; then
  say "RELEASE_GATE_FAIL: HEAD tag ${tag} != CITATION version ${expected_tag}"
  fail=1
fi

if [ -n "$(git status --porcelain)" ]; then
  say "RELEASE_GATE_FAIL: working tree is dirty; a release tag points at committed bytes only"
  fail=1
fi

for f in \
  docs/CAPABILITY_MATRIX.md \
  docs/START_HERE.md \
  docs/RELEASE.md \
  harness/q013b-local-adapter/Cargo.toml \
  evidence/runs/gnostral-q012-native-three-families-20261010.json \
  evidence/runs/gnostral-q013c-real-dense-single-server-20261010.json; do
  [ -f "$f" ] || { say "RELEASE_GATE_FAIL: missing ${f}"; fail=1; }
done

if ! grep -q "harness/engine-provider-contract/Cargo.toml" README.md; then
  say "RELEASE_GATE_FAIL: README does not document the offline contract test path"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
say "RELEASE_IDENTITY_PASS ${expected_tag} $(git rev-parse HEAD)"
