"""Fail-closed evidence gates for the gnostral.rs audit scripts.

`assert` statements disappear when Python runs with -O (PYTHONOPTIMIZE); a
qualification receipt must never be produced by a gate that silently vanished.
These helpers raise unconditionally, so the same code path fails closed in both
normal and optimized runs.

House style: every demand carries the reason the evidence would be rejected.
"""
from __future__ import annotations


class EvidenceError(AssertionError):
    """Raised when retained evidence does not satisfy a declared gate."""


def demand(condition: bool, why: str) -> None:
    """Raise EvidenceError(why) unless condition is truthy."""
    if not condition:
        raise EvidenceError(why)
