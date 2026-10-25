#!/usr/bin/env bash
# Bring a lab checkout to exactly origin/feat/engine-program, keeping everything that is not on GitHub.
#
#   bash scripts/lab_sync_checkout.sh          # preview: reports what would change, modifies nothing
#   bash scripts/lab_sync_checkout.sh --yes    # backup, then overwrite the checkout with the remote
#
# Kept in place, never touched: vendor/ (the lab engine tree with the private candle-strata),
# .ci-cache/ (builds), evidence/runs/ (possibly uncommitted evidence). Everything else that differs
# from the remote (local commits, edits, untracked files) is first saved under ~/gnostral-backup-<time>/
# (git bundle, binary diff, tarball), then discarded. The log of the run is saved there too; its
# content goes into the L-001 report.
set -euo pipefail

BRANCH=feat/engine-program
REMOTE_PATTERN="${GNOSTRAL_REMOTE_PATTERN:-gnu6dotlive/gnostral.rs}"
KEEP=(/.ci-cache/ /vendor/ /evidence/runs/)

apply=0
case "${1:-}" in
  --yes) apply=1 ;;
  "") ;;
  *) echo "usage: $0 [--yes]" >&2; exit 64 ;;
esac

top="$(git rev-parse --show-toplevel)"
cd "$top"
url="$(git remote get-url origin)"
if [[ "$url" != *"$REMOTE_PATTERN"* ]]; then
  echo "STOP: origin is $url, not $REMOTE_PATTERN" >&2
  exit 2
fi

stamp="$(date +%Y%m%d-%H%M%S)"
backup="$HOME/gnostral-backup-$stamp"
log="$(mktemp)"
say() { printf '%s\n' "$*" | tee -a "$log"; }
run() { say "\$ $*"; "$@" 2>&1 | tee -a "$log"; }

clean_args=()
for k in "${KEEP[@]}"; do clean_args+=(-e "$k"); done

say "== lab_sync_checkout $([[ $apply == 1 ]] && echo APPLY || echo PREVIEW) $stamp in $top"
run git fetch origin --prune
say "remote $BRANCH: $(git log --oneline -1 "origin/$BRANCH")"
say "local HEAD: $(git log --oneline -1 HEAD) on $(git rev-parse --abbrev-ref HEAD)"
say "-- local commits not on the remote:"
run git log --oneline "origin/$BRANCH..HEAD"
say "-- status:"
run git status --short
say "-- kept in place:"
for d in vendor .ci-cache evidence/runs; do
  [[ -e $d ]] && say "$(du -sh "$d" | cut -f1) $d" || say "absent: $d"
done
[[ -d vendor/mistralrs-strata ]] || say "WARNING: vendor/mistralrs-strata is absent; report it, the queue needs it"

if [[ $apply == 0 ]]; then
  say "-- untracked files that --yes would remove after backup (tracked files are reset to the remote):"
  git clean -nd "${clean_args[@]}" | tee -a "$log"
  say "PREVIEW only: nothing changed. Run again with --yes."
  rm -f "$log"
  exit 0
fi

mkdir -p "$backup"
run git bundle create "$backup/local-refs.bundle" --all
git diff --binary HEAD > "$backup/uncommitted.diff"
git ls-files --others --exclude-standard -z -- . ':(exclude).ci-cache' ':(exclude)vendor' \
  | tar --null -T - -czf "$backup/untracked.tgz"
git clean -nd "${clean_args[@]}" > "$backup/removed-untracked.txt"

run git checkout -f -B "$BRANCH" "origin/$BRANCH"
run git branch --set-upstream-to="origin/$BRANCH"
run git clean -fd "${clean_args[@]}"
git config user.name "Jordan"
git config user.email "dev.in@gnu6.live"

say "-- result:"
say "HEAD: $(git log --oneline -1 HEAD)"
say "-- left untracked (only under kept directories; list them in the L-001 report):"
run git status --short
if [[ -d vendor/mistralrs-strata ]]; then
  run python3 scripts/engine_series.py check --tree vendor/mistralrs-strata || true
fi
say "backup: $backup"
for f in "$backup"/*; do say "$(stat -c %s "$f") bytes  $(basename "$f")"; done
cp "$log" "$backup/sync.log"
rm -f "$log"
say "DONE. Read docs/WORKING_AGREEMENT.md, docs/lab/CONTRACT.md, docs/lab/QUEUE.md, then start L-001."
