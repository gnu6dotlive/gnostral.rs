#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail=0

if git grep -nE '(gho_|ghp_|github_pat_|AKIA[0-9A-Z]{16})' -- ':!scripts/check-public.sh'; then
  echo "ERROR: possible secret-like token found" >&2
  fail=1
fi

# Inspect the Git index, not local untracked vendored dependencies or model fixtures.
# CI checks exactly the files being published; a forced git add is still detected.
indexed_files="$(git ls-files --cached)" || { echo "ERROR: cannot inspect Git index" >&2; exit 1; }
if grep -E '\.(gguf|safetensors|nsys-rep|pid)$' <<<"$indexed_files"; then
  echo "ERROR: forbidden public artifact type staged or tracked" >&2
  fail=1
fi

if ! grep -q '35.2 tok/s' README.md; then
  echo "ERROR: qualified public baseline missing" >&2
  fail=1
fi

# Personal home-directory paths: the repository is public. Files committed before the rule are listed
# in scripts/home-path-allowlist.txt; everything else must go through scripts/scrub_home_paths.py.
home_leaks="$(git grep -lE '/(home|Users)/[A-Za-z0-9_.-]+/' -- . || true)"
if [[ -n "$home_leaks" ]]; then
  new_home_leaks="$(grep -vxF -f <(grep -v '^#' scripts/home-path-allowlist.txt) <<<"$home_leaks" || true)"
  if [[ -n "$new_home_leaks" ]]; then
    echo "$new_home_leaks"
    echo "ERROR: personal home path in tracked files; run scripts/scrub_home_paths.py on them" >&2
    fail=1
  fi
fi

if ! python3 scripts/check-doc-links.py; then
  echo "ERROR: active public documentation links/status index failed" >&2
  fail=1
fi

exit "$fail"
