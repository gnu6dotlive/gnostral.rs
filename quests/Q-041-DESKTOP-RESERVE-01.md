# Q-041 — DESKTOP-RESERVE-01

Review date: 2026-10-23
State: **PROPOSED** — reserve state machine and envelope arithmetic landed offline (`gnostral-memory-plan::reserve`); no observer or pressure test exists.
Authorization: elastic reserve and pressure-observer work after Q-027/Q-028, with the desktop session
protected. This quest does not authorize taking VRAM from the desktop by force, a driver setting
change, or a "desktop survives" claim based on a single unlucky run.
Workstream: W4 · Milestone: M4 (after M2 memory work) · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can the engine hold an explicit **desktop reserve** and hand memory back when the graphical session
shows pressure — elastic VRAM segments returning on demand and re-acquiring afterwards — without an
OOM, a session loss or a driver fault?

The 8 GiB card shares memory with a live desktop. The program's envelope says whole-card peak
≤ 7.0 GiB with ≥ 1.0 GiB free; this quest is how that envelope is enforced at runtime rather than
asserted before launch.

## Deliverable and file seams

- Reserve handshake between the memory manager (Q-027 segmented allocator) and a desktop-pressure
  observer; thresholds recorded in a profile, not hard-coded guesses.
- Explicit states: `reserved`, `released`, `reacquiring`; every transition receipted.
- Interaction with admission (Q-015/Q-039): a refused re-acquire is a typed rejection, not a crash.
- Driver health check (`NO_XID`) before and after each pressure test.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| Reserve honoured | the reserve stays free while the engine serves under normal load | reserve consumed by engine allocations |
| Pressure response | induced desktop pressure triggers release within the recorded profile | observer misses pressure or release is partial |
| Desktop survival | the graphical session is usable throughout the test | any session loss, compositor restart or Xid |
| Re-acquire | after pressure clears, memory is re-acquired with a receipt or refused explicitly | silent starvation or OOM |
| Driver health | `NO_XID` before and after every transition | any Xid, driver reset or unaccounted device fault |
| Reclaim | released memory is observed as returned, not merely requested | "released" claimed without observation |
| No force | no driver or compositor setting is changed to make the test pass | test environment altered |

The M1 envelope number (≤ 7.0 GiB whole-card with ≥ 1.0 GiB free) is measured there; this quest may
use it as a target, not report it as its own result.

## Non-claims

- No reserve, reclaim or survival number exists.
- A single successful pressure test does not prove desktop safety in general; the protocol repeats it.
- No claim of hard VRAM caps: NVIDIA does not expose one, and TOCTOU limits remain.
- No claim that the observer sees every pressure source (for example another GPU client).

## Definition of done

1. Pressure observer and reserve transitions unit-tested offline where possible.
2. A repeated pressure test with `NO_XID`, session health, release/re-acquire receipts and all
   samples retained.
3. A documented limitation list, including what the observer cannot see.
