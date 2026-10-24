# Q-043 — COHOST-01

Review date: 2026-10-24
State: **PROPOSED** — broker policy landed offline (`crates/gnostral-cohost`, 7 tests); no engine hook, no device run.
Authorization: offline policy and simulation; engine wiring only after Q-027 elastic expert budget and
Q-041 reserve are integrated. This quest does not authorize loading two models on the card by force,
taking memory from the desktop, or a "fits on 8 GiB" claim without a measured whole-card peak.
Workstream: W7 · Milestone: M7 · [Program](../docs/ENGINE_PROGRAM.md)

## Question

Can several models (a MoE language model, speech-to-text, text-to-speech, a small draft model)
share one 8 GiB desktop card, with the language model's expert cache acting as the elastic part
that shrinks when speech arrives and grows back when it leaves, without ever overbooking the card?

## Why the MoE engine makes this possible

A dense model has a fixed size: either it fits or it does not. A MoE model served with expert
residency (Q-027) has a **core** (attention, embeddings, router, KV floor) and an **elastic** expert
cache: fewer resident experts cost speed, not correctness. That elastic part is the memory the other
models borrow.

## Deliverable and file seams

- `crates/gnostral-cohost`: tenants declare `core_bytes`, an optional elastic `min..max`, a class
  (`Realtime`, `Interactive`, `Background`) and whether they may be parked. The broker turns admit,
  wake, release and capacity changes into plans: frees first (shrink, park, unload), then loads and
  grows. Refusals are typed and change nothing; every decision is receipted.
- Engine side (later): `GNOSTRAL_EXPERT_CACHE_BYTES` becomes a live budget the broker can move;
  the Q-041 reserve feeds `set_capacity`; the Q-040 supervisor hosts tenants instead of switching.

## Gate

| Gate | Pass | Fail |
| --- | --- | --- |
| No overbooking | every executed plan step keeps measured whole-card usage under the envelope | any step over the envelope, any OOM |
| Class rules | audio never loses memory to text; background never squeezes interactive | a lower class takes from a higher one |
| Elastic return | after speech leaves, the expert cache regrows with a receipt | silent starvation |
| Quality | language model tokens identical at every cache size (Q-030) | any token change caused by a resize |
| Desktop | Q-041 reserve honoured during every resize; `NO_XID` | reserve consumed, Xid, session loss |
| Receipts | every admission, refusal and resize recorded with bytes before/after | unexplained change |

## Verified offline

- 200 random sequences of admit/release/wake/busy/capacity changes on 2-8 GiB: no plan step ever
  exceeds capacity; shrinks and parks respect class rules; refusals leave state unchanged.
- The assistant scenario on the reference desktop envelope (8 GiB card, 1 GiB desktop, 512 MiB
  reserve, 512 MiB floor): the language model takes all 6 GiB alone, speech-to-text and
  text-to-speech come out of its expert cache, a background draft is refused, and the cache regrows
  when speech leaves.

## Non-claims

- Byte figures in tests are illustrative inputs, not measured model sizes.
- No claim about speed at a reduced cache size; that is a Q-031 measurement per budget.
- No claim that CUDA frees memory exactly when asked; reclaim is observed (Q-041), not assumed.

## Definition of done

1. Broker wired to a live expert-cache budget in the engine, with resize receipts.
2. A measured run: language model plus speech models resident, whole-card peak within the envelope,
   tokens identical across resizes, `NO_XID`.
3. A per-budget speed table (Q-031) so the cost of each shrink is known, not guessed.
