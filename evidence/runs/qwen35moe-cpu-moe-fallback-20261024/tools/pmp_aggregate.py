#!/usr/bin/env python3
"""Aggregate poor-man's-profiler samples (gdb `thread apply all bt`, see pmp.sh) into thread states,
top frames, first engine frame and the engine call chains under the copy and dequantize leaves."""

from __future__ import annotations

import collections
import re
import sys

IDLE = ("futex", "pthread_cond", "epoll_wait", "nanosleep", "clock_nanosleep", "__GI___poll", "wait4",
        "read", "__GI___libc_read", "syscall")
ENGINE = r"mistralrs|candle|gnostral|gdn"
LEAVES = ("memcpy", "to_float", "munmap", "memset")


def stacks_of(text: str) -> list[list[str]]:
    stacks, cur = [], None
    for line in text.splitlines():
        if line.startswith("Thread"):
            if cur:
                stacks.append(cur)
            cur = []
        elif line.startswith("#") and cur is not None:
            m = re.search(r" in (.+?) \(", line) or re.search(r"#\d+\s+(\S+) \(", line)
            cur.append(m.group(1) if m else line)
        elif line.startswith("==="):
            if cur:
                stacks.append(cur)
            cur = None
    if cur:
        stacks.append(cur)
    return stacks


def clean(frame: str) -> str:
    for _ in range(4):
        frame = re.sub(r"<[^<>]*>", "", frame)
    return frame


def short(frame: str) -> str:
    parts = clean(frame).split("::")
    return "::".join(parts[-2:]) if len(parts) > 1 else parts[0]


def main(path: str) -> None:
    text = open(path).read()
    samples = text.count("=== sample")
    state, top, first, chains = (collections.Counter() for _ in range(4))
    for st in stacks_of(text):
        if not st:
            continue
        f0 = st[0]
        if any(k in f0 for k in IDLE) and not any("sched_yield" in f for f in st[:2]):
            state["idle"] += 1
            continue
        state["spin(sched_yield)" if "sched_yield" in f0 else "running"] += 1
        top[short(f0)] += 1
        first[short(next((f for f in st if re.search(ENGINE, f)), "?"))] += 1
        if any(k in f0 for k in LEAVES):
            frames = [clean(f) for f in st if re.search(ENGINE + "|" + "|".join(LEAVES), f)]
            chains[" <- ".join(frames[:9])] += 1
    print(f"samples: {samples}")
    print("thread states:", dict(state))
    print("\ntop frame (non-idle):")
    for k, n in top.most_common(15):
        print(f"  {n:4d}  {k}")
    print("\nfirst engine frame (non-idle):")
    for k, n in first.most_common(20):
        print(f"  {n:4d}  {k}")
    print("\nengine chains under copy/dequantize leaves:")
    for k, n in chains.most_common(8):
        print(f"[{n}] {k}\n")


if __name__ == "__main__":
    main(sys.argv[1])
