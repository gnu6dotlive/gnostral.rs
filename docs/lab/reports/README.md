# Lab reports

One file per queue item and session, named `<yyyymmdd>-<item>.md` (for example `20261025-L-001.md`).
Written by the lab worker under [`../CONTRACT.md`](../CONTRACT.md); read by the cloud session, which
journals them in [`docs/FINDINGS.md`](../../FINDINGS.md). A report is never edited after it is pushed;
a correction is a new report that names the one it corrects.

Copy this template:

```markdown
# <item> · <yyyymmdd> · <one line: what happened>

- **Status:** DONE | PARTIAL | BLOCKED | REFUSED-BY-GATE
- **Repository:** feat/engine-program at <commit you started from>
- **Engine tree:** <output line of scripts/engine_series.py check>
- **candle-strata:** <revision>
- **Binaries:** <name, build command, features, SHA-256> for each one used
- **Machine state:** container stopped at <time> / started at <time>; nvidia-smi before the first run: <used/total MiB>; free disk

## Problems and surprises

Anything that went wrong or differs from the contract, the queue or earlier findings, first. "None" if none.

## Results

Each number with its claim class (OBSERVED, ...) and the path of its receipt under evidence/runs/.

## Commands

The commands as run, in order (home paths as ~).

## Left undone or unknown

What the item asked that this session did not do, and why.
```
