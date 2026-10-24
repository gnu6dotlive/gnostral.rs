"""NVIDIA driver-health probe for `gnostral-bench` plans (`driver_probe`).

Prints exactly one `GNOSTRAL_DRIVER NO_XID|XID|UNKNOWN` line, then one
`GNOSTRAL_DRIVER_DETAIL ...` line saying what the verdict rests on.

`NO_XID` is a claim that the kernel log of the current boot was read and holds no
`NVRM: Xid` line. It is only printed when that is true:

- no NVIDIA kernel driver loaded (`/proc/driver/nvidia/version` absent) -> `UNKNOWN`;
- no readable kernel log for this boot (journal empty or not permitted, `dmesg`
  restricted) -> `UNKNOWN`;
- an `NVRM: Xid` line in the log -> `XID`.

An earlier shell one-liner printed `NO_XID` whenever `grep` found nothing, including
when the journal could not be read at all (docs/FINDINGS.md, F-016).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

NVIDIA_VERSION = Path("/proc/driver/nvidia/version")
XID_MARKER = "NVRM: Xid"

# (argv) -> (returncode, stdout); injected by tests.
Runner = Callable[[list[str]], tuple[int, str]]


def _run(argv: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return done.returncode, done.stdout


def kernel_log(run: Runner) -> tuple[str | None, str]:
    """The current boot's kernel log and its source, or (None, why it is unreadable)."""
    reasons = []
    code, out = run(["journalctl", "-k", "-b", "--no-pager", "-o", "cat"])
    lines = [line for line in out.splitlines() if line.strip() and not line.startswith("-- ")]
    if code == 0 and lines:
        return out, f"journalctl lines={len(lines)}"
    reasons.append(f"journalctl exit={code} lines={len(lines)}")
    code, out = run(["dmesg"])
    if code == 0 and out.strip():
        return out, f"dmesg lines={len(out.splitlines())}"
    reasons.append(f"dmesg exit={code}")
    return None, "; ".join(reasons)


def probe(run: Runner = _run, nvidia_present: Callable[[], bool] = NVIDIA_VERSION.exists) -> tuple[str, str]:
    if not nvidia_present():
        return "UNKNOWN", "no NVIDIA kernel driver loaded"
    log, source = kernel_log(run)
    if log is None:
        return "UNKNOWN", f"kernel log of this boot not readable ({source})"
    xids = [line for line in log.splitlines() if XID_MARKER in line]
    if xids:
        return "XID", f"{source} xid_lines={len(xids)} first={xids[0].strip()[:160]!r}"
    return "NO_XID", f"{source} xid_lines=0"


def main() -> int:
    status, detail = probe()
    print(f"GNOSTRAL_DRIVER {status}")
    print(f"GNOSTRAL_DRIVER_DETAIL {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
