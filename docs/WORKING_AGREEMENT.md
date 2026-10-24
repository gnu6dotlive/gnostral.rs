# Working agreement

How work on this program is done, by people and by agents. Set by the owner on 2026-10-24.

1. **Truth before velocity.** Work is judged on understanding the problem's complexity, on a plan that
   loses nothing over time, and on transparency. It is not judged on how fast something ships.
2. **Say it when it goes wrong.** A problem, a failed fix or an earlier claim that turns out wrong is
   reported as soon as it is known, with its evidence, and recorded in `docs/FINDINGS.md`. Reporting a
   problem is never penalised; hiding one is the only failure.
3. **No undocumented partial state.** Unfinished work is either finished or left with its state written
   down: what works, what does not, what is unknown. Asking for more time beats delivering something that
   works only in theory.
4. **Measure before optimising.** A change is chosen from a profile or a measurement, and its result is
   measured the same way, before and after, in the same session.
5. **Every claim has a class and a receipt.** Claim classes come from `docs/EVIDENCE_MODEL.md`. No number
   without a receipt in the repository, and nothing is called faster while the ranges overlap.
6. **Correctness gates come before timing:** parity, self-reproduction, pinned placement and driver health.
7. **Nothing leaves the machine without the owner's approval of that action.**
8. **When resources run out** (time, tokens, hardware), stop at a documented state and wait. Do not cut
   corners to finish.
