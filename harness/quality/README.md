# Q-030 quality oracle

CPU-only, offline gate for QUALITY-PARITY-01. It compares **captured** token streams
and distributions; it never runs a model and never produces a performance number.

## What is frozen here

| Artifact | Path | Identity |
| --- | --- | --- |
| Prompt set `six-check-20261023` (6 prompts) | `prompts/quality-prompts-20261023.json` | one SHA-256 per prompt, one canonical SHA-256 over the manifest |
| Oracle | `quality_oracle.py` | — |
| Tests | `test_quality_oracle.py` | 30 tests, no network, no model |

The manifest hash is printed by `manifest-hash` and must be copied verbatim into every
capture, so a capture can never be compared against a different prompt set.

**Provenance note.** Q-025 refers to "the retained six-check prompt set" but the
repository contains no verbatim prompt bytes for it — only the *name* of the probe and
its published medians. This manifest is therefore a **new, explicitly frozen set**: it
is not claimed to be byte-identical to the Q-025 probe. The two must be reconciled
(pin the Q-025 bytes, or amend Q-025 to name this manifest) before D1 is decided.

## Commands

```sh
# canonical identity of the frozen set
python3 harness/quality/quality_oracle.py manifest-hash \
  harness/quality/prompts/quality-prompts-20261023.json

# greedy token parity between two lane captures
python3 harness/quality/quality_oracle.py parity \
  --manifest harness/quality/prompts/quality-prompts-20261023.json \
  --reference ref.json --candidate cand.json

# KL(reference || candidate) over a normalised distribution
python3 harness/quality/quality_oracle.py kl --reference p.json --candidate q.json

# KS uniformity gate over probability integral transforms
python3 harness/quality/quality_oracle.py pit --samples pit.json

# tests
python3 -m unittest discover -s harness/quality -p 'test_*.py'
```

Exit codes: `0` pass, `1` gate failed (divergence / KL over threshold / KS rejected),
`2` blocked or malformed (unpinned lane, wrong manifest hash, incomplete capture,
unnormalised distribution).

## Gate semantics

- **Greedy parity** requires `temperature == 0`, both lanes pinned to the same model
  SHA-256 and their own binary SHA-256. Anything else is `BLOCKED`, never `PASS`.
- **Divergence** reports the first differing token index and both token ids; the
  matched prefix is retained so a reviewer can see how far the lanes agreed.
- **Batch invariance** reuses the same comparison for batch-1 vs batch-N captures of
  one lane and relabels the failure explicitly.
- **KL** applies a declared epsilon floor (default `1e-12`) and reports it; a result
  is only meaningful against a pre-declared threshold.
- **KS/PIT** needs at least 8 samples and returns the statistic, the asymptotic
  p-value and the chosen alpha.

## Capture shape

```json
{
  "schema": "gnostral.quality.capture/1",
  "promptset_sha256": "<manifest-hash>",
  "lane": {"engine": "llama.cpp", "binary_sha256": "<64 hex>", "model_sha256": "<64 hex>"},
  "sampling": {"temperature": 0.0, "seed": 0},
  "captures": [
    {"prompt_id": "p1-factual", "prompt_sha256": "<per-prompt hash>", "token_ids": [1, 2, 3]}
  ]
}
```

Every prompt in the manifest must appear exactly once; an incomplete capture is
rejected rather than compared partially.

## Not done here

- No reference capture exists yet: no engine has produced token streams for this
  manifest. The oracle is a gate, not evidence.
- No KL/KS measurement against a reference engine has been taken.
- No long-context parity, no perplexity harness, no sampler-level PIT capture from a
  real engine — those need a GPU session with the frozen protocol.
- This module does not judge *quality*; it detects divergence and miscalibration. A
  `PARITY_PASS` means "same tokens", not "good tokens".
