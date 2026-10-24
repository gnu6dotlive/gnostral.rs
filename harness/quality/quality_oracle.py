"""Q-030 quality oracle — parity, divergence and sampler gates over captured tokens.

This module is the CPU-only half of QUALITY-PARITY-01. It does not run a model,
does not download anything and never invents a measurement: it validates captured
token streams against a frozen prompt-set manifest and computes the gates the
engine program requires before any performance number is recorded.

Design rules, in the repository's evidence discipline:

- **Frozen inputs.** A prompt set is a manifest with one SHA-256 per exact prompt
  byte sequence and one SHA-256 over the canonical manifest. A capture that does
  not name the same manifest and prompt hashes is `BLOCKED`, never compared.
- **Pinned identities.** A lane is only comparable when its engine binary and
  model file are both pinned by SHA-256. Missing pins block the gate.
- **No silent defaults.** Missing fields, NaN, empty token streams and unnormalised
  distributions raise instead of returning a neutral value.
- **Unknowns stay unknown.** `UNKNOWN` is a verdict, not a zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_MANIFEST = "gnostral.quality.promptset/1"
SCHEMA_CAPTURE = "gnostral.quality.capture/1"
SCHEMA_GATES = "gnostral.quality.gates/1"
SCHEMA_LEDGER = "gnostral.quality.ledger/1"
SCHEMA_GATE_RECEIPT = "gnostral.quality.gate-receipt/1"

VERDICT_PASS = "PARITY_PASS"
VERDICT_DIVERGED = "DIVERGED"
VERDICT_BLOCKED = "BLOCKED"

_HEX64 = set("0123456789abcdef")


class OracleError(ValueError):
    """A malformed manifest, capture or distribution. Never downgraded to a verdict."""


# ---------------------------------------------------------------------------
# Hashing and canonical forms
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no insignificant space)."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _require_pin(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
        raise OracleError(f"{field} must be a 64-character lowercase SHA-256, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Frozen prompt manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    sha256: str


@dataclass(frozen=True)
class PromptSet:
    name: str
    frozen: str
    prompts: tuple[Prompt, ...]
    manifest_sha256: str

    def by_id(self) -> dict[str, Prompt]:
        return {prompt.id: prompt for prompt in self.prompts}


def load_manifest(path: str | Path) -> PromptSet:
    """Load and verify a frozen prompt set. Any inconsistency raises."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OracleError("manifest must be a JSON object")
    if raw.get("schema") != SCHEMA_MANIFEST:
        raise OracleError(f"manifest schema must be {SCHEMA_MANIFEST}")
    name = raw.get("name")
    frozen = raw.get("frozen")
    if not isinstance(name, str) or not name:
        raise OracleError("manifest name must be a non-empty string")
    if not isinstance(frozen, str) or not frozen:
        raise OracleError("manifest frozen date must be a non-empty string")

    entries = raw.get("prompts")
    if not isinstance(entries, list) or not entries:
        raise OracleError("manifest prompts must be a non-empty list")

    prompts: list[Prompt] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise OracleError(f"prompt {index} must be an object")
        prompt_id = entry.get("id")
        text = entry.get("text")
        digest = entry.get("sha256")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise OracleError(f"prompt {index} id must be a non-empty string")
        if prompt_id in seen:
            raise OracleError(f"duplicate prompt id {prompt_id!r}")
        if not isinstance(text, str) or not text:
            raise OracleError(f"prompt {prompt_id!r} text must be a non-empty string")
        expected = sha256_text(text)
        if digest != expected:
            raise OracleError(
                f"prompt {prompt_id!r} declares sha256 {digest!r} but its text hashes to {expected}"
            )
        seen.add(prompt_id)
        prompts.append(Prompt(id=prompt_id, text=text, sha256=expected))

    return PromptSet(
        name=name,
        frozen=frozen,
        prompts=tuple(prompts),
        manifest_sha256=canonical_sha256(raw),
    )


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lane:
    engine: str
    binary_sha256: str | None
    model_sha256: str | None

    @property
    def pinned(self) -> bool:
        return bool(self.binary_sha256) and bool(self.model_sha256)


@dataclass(frozen=True)
class Capture:
    prompt_id: str
    prompt_sha256: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LaneCapture:
    lane: Lane
    promptset_sha256: str
    sampling: dict[str, Any]
    captures: dict[str, Capture]

    def tokens(self, prompt_id: str) -> tuple[int, ...]:
        return self.captures[prompt_id].token_ids


def load_capture(path: str | Path, manifest: PromptSet) -> LaneCapture:
    """Load a lane capture and verify it against the frozen manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OracleError("capture must be a JSON object")
    if raw.get("schema") != SCHEMA_CAPTURE:
        raise OracleError(f"capture schema must be {SCHEMA_CAPTURE}")

    declared = raw.get("promptset_sha256")
    if declared != manifest.manifest_sha256:
        raise OracleError(
            "capture names promptset "
            f"{declared!r} but the manifest hashes to {manifest.manifest_sha256!r}"
        )

    lane_raw = raw.get("lane")
    if not isinstance(lane_raw, dict):
        raise OracleError("capture lane must be an object")
    engine = lane_raw.get("engine")
    if not isinstance(engine, str) or not engine:
        raise OracleError("capture lane.engine must be a non-empty string")
    binary = lane_raw.get("binary_sha256")
    model = lane_raw.get("model_sha256")
    lane = Lane(
        engine=engine,
        binary_sha256=None if binary is None else _require_pin(binary, "lane.binary_sha256"),
        model_sha256=None if model is None else _require_pin(model, "lane.model_sha256"),
    )

    sampling = raw.get("sampling")
    if not isinstance(sampling, dict):
        raise OracleError("capture sampling must be an object")

    entries = raw.get("captures")
    if not isinstance(entries, list):
        raise OracleError("capture captures must be a list")

    known = manifest.by_id()
    captured: dict[str, Capture] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise OracleError("capture entry must be an object")
        prompt_id = entry.get("prompt_id")
        if prompt_id not in known:
            raise OracleError(f"capture names unknown prompt {prompt_id!r}")
        if prompt_id in captured:
            raise OracleError(f"duplicate capture for prompt {prompt_id!r}")
        prompt = known[prompt_id]
        if entry.get("prompt_sha256") != prompt.sha256:
            raise OracleError(
                f"capture prompt {prompt_id!r} hash {entry.get('prompt_sha256')!r} "
                f"does not match manifest {prompt.sha256!r}"
            )
        token_ids = entry.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise OracleError(f"capture {prompt_id!r} token_ids must be a non-empty list")
        for token in token_ids:
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                raise OracleError(
                    f"capture {prompt_id!r} has a non-token id {token!r}"
                )
        captured[prompt_id] = Capture(
            prompt_id=prompt_id,
            prompt_sha256=prompt.sha256,
            token_ids=tuple(token_ids),
        )

    missing = sorted(set(known) - set(captured))
    if missing:
        raise OracleError(f"capture is incomplete; missing prompts {missing}")

    return LaneCapture(
        lane=lane,
        promptset_sha256=manifest.manifest_sha256,
        sampling=sampling,
        captures=captured,
    )


# ---------------------------------------------------------------------------
# Greedy token parity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityResult:
    prompt_id: str
    verdict: str
    matched_prefix: int
    divergence_index: int | None
    detail: str


def first_divergence(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def greedy_parity(
    reference: LaneCapture, candidate: LaneCapture, prompt_id: str
) -> ParityResult:
    """Compare one prompt's greedy token stream between two lanes."""
    if prompt_id not in reference.captures or prompt_id not in candidate.captures:
        raise OracleError(f"prompt {prompt_id!r} is absent from one capture")

    if not reference.lane.pinned or not candidate.lane.pinned:
        return ParityResult(
            prompt_id=prompt_id,
            verdict=VERDICT_BLOCKED,
            matched_prefix=0,
            divergence_index=None,
            detail="lane identity is not pinned (binary_sha256 and model_sha256 are required)",
        )

    if reference.lane.model_sha256 != candidate.lane.model_sha256:
        return ParityResult(
            prompt_id=prompt_id,
            verdict=VERDICT_BLOCKED,
            matched_prefix=0,
            divergence_index=None,
            detail="lanes were measured on different model files; parity is not comparable",
        )

    if reference.sampling.get("temperature") != 0 or candidate.sampling.get("temperature") != 0:
        return ParityResult(
            prompt_id=prompt_id,
            verdict=VERDICT_BLOCKED,
            matched_prefix=0,
            divergence_index=None,
            detail="greedy parity requires temperature 0 in both lanes",
        )

    left = reference.tokens(prompt_id)
    right = candidate.tokens(prompt_id)
    divergence = first_divergence(left, right)
    if divergence is None:
        return ParityResult(
            prompt_id=prompt_id,
            verdict=VERDICT_PASS,
            matched_prefix=len(left),
            divergence_index=None,
            detail=f"{len(left)} tokens identical",
        )
    return ParityResult(
        prompt_id=prompt_id,
        verdict=VERDICT_DIVERGED,
        matched_prefix=divergence,
        divergence_index=divergence,
        detail=(
            f"diverged at token {divergence}: "
            f"{left[divergence] if divergence < len(left) else 'END'} vs "
            f"{right[divergence] if divergence < len(right) else 'END'}"
        ),
    )


def parity_suite(
    reference: LaneCapture, candidate: LaneCapture
) -> list[ParityResult]:
    return [
        greedy_parity(reference, candidate, prompt_id)
        for prompt_id in reference.captures
    ]


def batch_invariance(
    single: LaneCapture, batched: LaneCapture, prompt_id: str
) -> ParityResult:
    """Same lane, batch size 1 vs batch size N: token streams must be identical."""
    result = greedy_parity(single, batched, prompt_id)
    if result.verdict == VERDICT_DIVERGED:
        return ParityResult(
            prompt_id=prompt_id,
            verdict=VERDICT_DIVERGED,
            matched_prefix=result.matched_prefix,
            divergence_index=result.divergence_index,
            detail=f"batch invariance failed — {result.detail}",
        )
    return result


# ---------------------------------------------------------------------------
# Distribution gates
# ---------------------------------------------------------------------------


def _as_distribution(values: Sequence[float], name: str) -> list[float]:
    if len(values) < 2:
        raise OracleError(f"{name} needs at least two entries")
    cleaned: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OracleError(f"{name} contains a non-number {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise OracleError(f"{name} contains a non-finite value")
        if number < 0:
            raise OracleError(f"{name} contains a negative probability")
        cleaned.append(number)
    total = sum(cleaned)
    if total <= 0:
        raise OracleError(f"{name} sums to zero")
    if abs(total - 1.0) > 1e-6:
        raise OracleError(f"{name} is not normalised (sums to {total!r})")
    return cleaned


def kl_divergence(reference: Sequence[float], candidate: Sequence[float], epsilon: float = 1e-12) -> float:
    """KL(reference || candidate) in nats, with a declared smoothing floor.

    A zero in `candidate` where `reference` is non-zero is infinite in theory; the
    epsilon floor makes it finite and the caller must record that the floor was
    applied. `epsilon` itself is returned to the caller as part of the receipt.
    """
    if len(reference) != len(candidate):
        raise OracleError("distributions must have the same length")
    if not 0 < epsilon < 1:
        raise OracleError("epsilon must be in (0, 1)")
    p = _as_distribution(reference, "reference distribution")
    q = _as_distribution(candidate, "candidate distribution")

    smoothed = [value + epsilon for value in q]
    total = sum(smoothed)
    smoothed = [value / total for value in smoothed]

    divergence = 0.0
    for p_value, q_value in zip(p, smoothed):
        if p_value > 0.0:
            divergence += p_value * math.log(p_value / q_value)
    return divergence


def pit_ks(samples: Iterable[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and asymptotic p-value for PIT uniformity.

    `samples` must be probability integral transforms in [0, 1]. A well-calibrated
    sampler produces a large p-value; a biased sampler does not.
    """
    values = sorted(float(value) for value in samples)
    count = len(values)
    if count < 8:
        raise OracleError("KS needs at least 8 samples to be meaningful")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise OracleError("PIT samples must lie in [0, 1]")

    d_plus = max((index + 1) / count - value for index, value in enumerate(values))
    d_minus = max(value - index / count for index, value in enumerate(values))
    statistic = max(d_plus, d_minus)

    root = math.sqrt(count)
    lam = (root + 0.12 + 0.11 / root) * statistic
    if lam <= 0:
        return statistic, 1.0
    series = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        series += term
        if abs(term) < 1e-12:
            break
    return statistic, min(1.0, max(0.0, series))


def uniformity_verdict(samples: Sequence[float], alpha: float = 0.01) -> dict[str, Any]:
    statistic, p_value = pit_ks(samples)
    return {
        "verdict": "PASS" if p_value >= alpha else "FAIL",
        "ks_statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "samples": len(samples),
    }



# ---------------------------------------------------------------------------
# Frozen gates, divergence ledger and the full gate receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gates:
    name: str
    sha256: str
    min_prompts: int
    min_greedy_agreement: float
    max_kl_nats: float
    kl_epsilon: float
    ks_alpha: float
    batch_sizes: tuple[int, ...]
    long_context_tiers: tuple[int, ...]
    ledger_categories: tuple[str, ...]


def load_gates(path: str | Path) -> Gates:
    """Load the thresholds frozen before the first run; their hash goes into every receipt."""
    data = Path(path).read_bytes()
    raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_GATES:
        raise OracleError(f"gates schema must be {SCHEMA_GATES}")

    def number(key: str, lo: float, hi: float) -> float:
        value = raw.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not lo <= value <= hi:
            raise OracleError(f"gates.{key} must be a number in [{lo}, {hi}]")
        return float(value)

    def ints(key: str) -> tuple[int, ...]:
        value = raw.get(key)
        if not isinstance(value, list) or not value or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value
        ):
            raise OracleError(f"gates.{key} must be a non-empty list of positive integers")
        return tuple(value)

    categories = raw.get("ledger_categories")
    if not isinstance(categories, list) or not categories or not all(isinstance(c, str) and c for c in categories):
        raise OracleError("gates.ledger_categories must be a non-empty list of strings")
    return Gates(
        name=str(raw.get("name") or ""),
        sha256=sha256_bytes(data),
        min_prompts=int(number("min_prompts", 1, 1_000_000)),
        min_greedy_agreement=number("min_greedy_agreement", 0.0, 1.0),
        max_kl_nats=number("max_kl_nats", 0.0, 10.0),
        kl_epsilon=number("kl_epsilon", 1e-300, 0.5),
        ks_alpha=number("ks_alpha", 1e-12, 0.5),
        batch_sizes=ints("batch_sizes"),
        long_context_tiers=ints("long_context_tiers"),
        ledger_categories=tuple(categories),
    )


@dataclass(frozen=True)
class LedgerEntry:
    prompt_id: str
    divergence_index: int
    category: str
    explanation: str
    evidence_ref: str


def load_ledger(path: str | Path, gates: Gates) -> dict[tuple[str, int], LedgerEntry]:
    """Every divergent position must be explained here, at its exact index."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_LEDGER:
        raise OracleError(f"ledger schema must be {SCHEMA_LEDGER}")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise OracleError("ledger entries must be a list")
    ledger: dict[tuple[str, int], LedgerEntry] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise OracleError(f"ledger entry {index} must be an object")
        prompt_id = entry.get("prompt_id")
        position = entry.get("divergence_index")
        category = entry.get("category")
        explanation = entry.get("explanation")
        evidence = entry.get("evidence_ref")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise OracleError(f"ledger entry {index}: prompt_id required")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise OracleError(f"ledger entry {index}: divergence_index must be a non-negative integer")
        if category not in gates.ledger_categories:
            raise OracleError(f"ledger entry {index}: category {category!r} is not one of {list(gates.ledger_categories)}")
        if not isinstance(explanation, str) or not explanation.strip():
            raise OracleError(f"ledger entry {index}: explanation required")
        if not isinstance(evidence, str) or not evidence.strip():
            raise OracleError(f"ledger entry {index}: evidence_ref required")
        key = (prompt_id, position)
        if key in ledger:
            raise OracleError(f"ledger entry {index}: duplicate {key}")
        ledger[key] = LedgerEntry(prompt_id, position, category, explanation, evidence)
    return ledger


def token_agreement(reference: LaneCapture, results: Sequence[ParityResult]) -> float:
    """Matched-prefix tokens over reference tokens. Tokens after a divergence do not count:
    their context differs, so positional agreement after it would be meaningless."""
    total = sum(len(reference.tokens(r.prompt_id)) for r in results)
    matched = sum(r.matched_prefix for r in results)
    return matched / total if total else 0.0


def _result_json(r: ParityResult) -> dict[str, Any]:
    return {
        "prompt_id": r.prompt_id,
        "verdict": r.verdict,
        "matched_prefix": r.matched_prefix,
        "divergence_index": r.divergence_index,
        "detail": r.detail,
    }


def gate_receipt(
    manifest: PromptSet,
    gates: Gates,
    reference: LaneCapture,
    candidate: LaneCapture,
    ledger: dict[tuple[str, int], LedgerEntry] | None = None,
    batched: dict[int, LaneCapture] | None = None,
    long_context: dict[int, LaneCapture] | None = None,
) -> dict[str, Any]:
    """Full Q-030 gate. Verdict is BLOCKED, FAIL or PASS; every reason is listed."""
    ledger = ledger or {}
    batched = batched or {}
    long_context = long_context or {}
    blocked: list[str] = []
    failed: list[str] = []

    if len(manifest.prompts) < gates.min_prompts:
        blocked.append(
            f"prompt set {manifest.name!r} has {len(manifest.prompts)} prompts; the frozen gate needs {gates.min_prompts}"
        )

    results = parity_suite(reference, candidate)
    if any(r.verdict == VERDICT_BLOCKED for r in results):
        blocked.append("greedy parity is BLOCKED (unpinned lane, model mismatch or non-zero temperature)")
    agreement = token_agreement(reference, results)
    if agreement < gates.min_greedy_agreement:
        failed.append(f"greedy agreement {agreement:.6f} < {gates.min_greedy_agreement}")

    explained, unexplained = [], []
    for r in results:
        if r.verdict != VERDICT_DIVERGED:
            continue
        entry = ledger.get((r.prompt_id, r.divergence_index))
        if entry is None:
            unexplained.append(r.prompt_id)
        else:
            explained.append({"prompt_id": r.prompt_id, "divergence_index": r.divergence_index,
                              "category": entry.category, "evidence_ref": entry.evidence_ref})
    if unexplained:
        failed.append(f"unexplained divergences (hard fail): {unexplained}")

    batch_results: dict[str, Any] = {}
    for size in gates.batch_sizes:
        if size == 1:
            continue
        capture = batched.get(size)
        if capture is None:
            batch_results[str(size)] = "UNKNOWN"
            continue
        rs = [batch_invariance(candidate, capture, pid) for pid in candidate.captures]
        batch_results[str(size)] = [_result_json(r) for r in rs]
        if any(r.verdict == VERDICT_DIVERGED for r in rs):
            failed.append(f"batch invariance failed at batch {size}")
        if any(r.verdict == VERDICT_BLOCKED for r in rs):
            blocked.append(f"batch {size} capture is BLOCKED")

    tier_results: dict[str, Any] = {}
    for tier in gates.long_context_tiers:
        capture = long_context.get(tier)
        if capture is None:
            tier_results[str(tier)] = "UNKNOWN"
            continue
        rs = [greedy_parity(candidate, capture, pid) for pid in candidate.captures]
        tier_results[str(tier)] = [_result_json(r) for r in rs]
        if any(r.verdict == VERDICT_DIVERGED for r in rs):
            failed.append(f"long-context tier {tier} drifted from the short-context baseline")
        if any(r.verdict == VERDICT_BLOCKED for r in rs):
            blocked.append(f"long-context tier {tier} capture is BLOCKED")

    unknown = [f"batch {k}" for k, v in batch_results.items() if v == "UNKNOWN"]
    unknown += [f"tier {k}" for k, v in tier_results.items() if v == "UNKNOWN"]
    verdict = "BLOCKED" if blocked else ("FAIL" if failed else ("PASS" if not unknown else "INCOMPLETE"))
    return {
        "schema": SCHEMA_GATE_RECEIPT,
        "verdict": verdict,
        "blocked": blocked,
        "failed": failed,
        "unknown": unknown,
        "gates": {"name": gates.name, "sha256": gates.sha256},
        "promptset": {"name": manifest.name, "sha256": manifest.manifest_sha256, "prompts": len(manifest.prompts)},
        "reference": {"engine": reference.lane.engine, "binary_sha256": reference.lane.binary_sha256,
                      "model_sha256": reference.lane.model_sha256},
        "candidate": {"engine": candidate.lane.engine, "binary_sha256": candidate.lane.binary_sha256,
                      "model_sha256": candidate.lane.model_sha256},
        "greedy_agreement": agreement,
        "parity": [_result_json(r) for r in results],
        "explained_divergences": explained,
        "batch_invariance": batch_results,
        "long_context": tier_results,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-reproducibility: one lane, two runs, identical bits
# ---------------------------------------------------------------------------


def _float_bits(value: Any, where: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleError(f"{where}: logprob must be a number")
    return struct.pack("<d", float(value))


def _load_logprobs(path: str | Path) -> dict[str, list[list[tuple[int, bytes]]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise OracleError("logprobs document must be a list")
    out: dict[str, list[list[tuple[int, bytes]]]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("prompt_id"), str):
            raise OracleError("logprobs entry must name its prompt_id")
        pid = entry["prompt_id"]
        if pid in out:
            raise OracleError(f"duplicate logprobs for prompt {pid!r}")
        steps = entry.get("steps")
        if not isinstance(steps, list):
            raise OracleError(f"logprobs {pid!r} steps must be a list")
        parsed = []
        for i, step in enumerate(steps):
            if not isinstance(step, list) or not step:
                raise OracleError(f"logprobs {pid!r} step {i} must be a non-empty list")
            row = []
            for cand in step:
                if not (isinstance(cand, list) and len(cand) == 2 and isinstance(cand[0], int)):
                    raise OracleError(f"logprobs {pid!r} step {i} candidates must be [token, logprob]")
                row.append((cand[0], _float_bits(cand[1], f"logprobs {pid!r} step {i}")))
            parsed.append(row)
        out[pid] = parsed
    return out


def self_reproduction(
    manifest: PromptSet,
    first_capture: str | Path,
    second_capture: str | Path,
    first_logprobs: str | Path,
    second_logprobs: str | Path,
) -> dict[str, Any]:
    """Does a lane reproduce itself? Tokens and every recorded logprob must be bit-identical.

    `BLOCKED` when the two runs are not the same lane (engine, binary, model) or a pin is
    missing; a different reported placement is a `FAIL` of kind `placement`.
    """
    first = load_capture(first_capture, manifest)
    second = load_capture(second_capture, manifest)
    raw_first = json.loads(Path(first_capture).read_text(encoding="utf-8"))["lane"]
    raw_second = json.loads(Path(second_capture).read_text(encoding="utf-8"))["lane"]
    result: dict[str, Any] = {"lane": first.lane.engine, "prompts": len(first.captures)}
    reasons = []
    identity = lambda raw: {k: v for k, v in raw.items() if k != "placement"}
    if first.lane != second.lane or identity(raw_first) != identity(raw_second):
        reasons.append("the two captures are not the same lane identity")
    if first.lane.binary_sha256 is None or first.lane.model_sha256 is None:
        reasons.append("binary and model must both be pinned")
    if reasons:
        result.update(verdict=VERDICT_BLOCKED, reasons=reasons, first_difference=None)
        return result

    def fail(prompt_id: str | None, kind: str, index: int | None) -> dict[str, Any]:
        result.update(verdict="FAIL", reasons=[],
                      first_difference={"prompt_id": prompt_id, "kind": kind, "index": index})
        return result

    if raw_first.get("placement") != raw_second.get("placement"):
        return fail(None, "placement", None)
    lp_first, lp_second = _load_logprobs(first_logprobs), _load_logprobs(second_logprobs)
    for prompt_id in sorted(first.captures):
        a, b = first.tokens(prompt_id), second.tokens(prompt_id)
        index = first_divergence(a, b)
        if index is None and len(a) != len(b):
            index = min(len(a), len(b))
        steps_a, steps_b = lp_first.get(prompt_id), lp_second.get(prompt_id)
        if steps_a is None or steps_b is None:
            raise OracleError(f"logprobs missing for prompt {prompt_id!r}")
        lp_index = next((i for i, (x, y) in enumerate(zip(steps_a, steps_b)) if x != y), None)
        if lp_index is None and len(steps_a) != len(steps_b):
            lp_index = min(len(steps_a), len(steps_b))
        if lp_index is not None and (index is None or lp_index <= index):
            return fail(prompt_id, "logprob", lp_index)
        if index is not None:
            return fail(prompt_id, "token", index)
    result.update(verdict="PASS", reasons=[], first_difference=None)
    return result


def _load_pair(args: argparse.Namespace) -> tuple[PromptSet, LaneCapture, LaneCapture]:
    manifest = load_manifest(args.manifest)
    reference = load_capture(args.reference, manifest)
    candidate = load_capture(args.candidate, manifest)
    return manifest, reference, candidate


def _cmd_manifest_hash(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(json.dumps({
        "name": manifest.name,
        "frozen": manifest.frozen,
        "prompts": len(manifest.prompts),
        "manifest_sha256": manifest.manifest_sha256,
    }, indent=2, sort_keys=True))
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    _, reference, candidate = _load_pair(args)
    results = [greedy_parity(reference, candidate, prompt_id) for prompt_id in reference.captures]
    payload = {
        "reference": reference.lane.engine,
        "candidate": candidate.lane.engine,
        "results": [
            {
                "prompt_id": result.prompt_id,
                "verdict": result.verdict,
                "matched_prefix": result.matched_prefix,
                "divergence_index": result.divergence_index,
                "detail": result.detail,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    blocked = any(result.verdict == VERDICT_BLOCKED for result in results)
    diverged = any(result.verdict == VERDICT_DIVERGED for result in results)
    return 2 if blocked else (1 if diverged else 0)


def _cmd_self_repro(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    result = self_reproduction(manifest, args.first, args.second, args.first_logprobs, args.second_logprobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, VERDICT_BLOCKED: 2}[result["verdict"]]


def _cmd_kl(args: argparse.Namespace) -> int:
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    value = kl_divergence(reference, candidate, epsilon=args.epsilon)
    payload = {
        "kl_nats": value,
        "kl_bits": value / math.log(2.0),
        "epsilon": args.epsilon,
        "verdict": "PASS" if value <= args.max_kl else "FAIL",
        "max_kl_nats": args.max_kl,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if value <= args.max_kl else 1


def _cmd_pit(args: argparse.Namespace) -> int:
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    result = uniformity_verdict(samples, alpha=args.alpha)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _sized_captures(pairs: Sequence[str], manifest: PromptSet, flag: str) -> dict[int, LaneCapture]:
    out: dict[int, LaneCapture] = {}
    for pair in pairs:
        size, sep, path = pair.partition("=")
        if not sep or not size.isdigit() or int(size) <= 0:
            raise OracleError(f"{flag} expects SIZE=PATH, got {pair!r}")
        if int(size) in out:
            raise OracleError(f"{flag} {size} given twice")
        out[int(size)] = load_capture(path, manifest)
    return out


def _cmd_gate(args: argparse.Namespace) -> int:
    manifest, reference, candidate = _load_pair(args)
    gates = load_gates(args.gates)
    ledger = load_ledger(args.ledger, gates) if args.ledger else {}
    receipt = gate_receipt(
        manifest,
        gates,
        reference,
        candidate,
        ledger,
        _sized_captures(args.batch, manifest, "--batch"),
        _sized_captures(args.long_context, manifest, "--long-context"),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 1, "BLOCKED": 2}[receipt["verdict"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quality-oracle", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_hash = sub.add_parser("manifest-hash", help="print the frozen manifest hash")
    manifest_hash.add_argument("manifest")
    manifest_hash.set_defaults(func=_cmd_manifest_hash)

    parity = sub.add_parser("parity", help="greedy token parity between two lane captures")
    parity.add_argument("--manifest", required=True)
    parity.add_argument("--reference", required=True)
    parity.add_argument("--candidate", required=True)
    parity.set_defaults(func=_cmd_parity)

    repro = sub.add_parser("self-repro", help="two runs of one lane: identical tokens and logprob bits")
    repro.add_argument("--manifest", required=True)
    repro.add_argument("--first", required=True)
    repro.add_argument("--second", required=True)
    repro.add_argument("--first-logprobs", required=True)
    repro.add_argument("--second-logprobs", required=True)
    repro.set_defaults(func=_cmd_self_repro)

    kl = sub.add_parser("kl", help="KL divergence between two normalised distributions")
    kl.add_argument("--reference", required=True)
    kl.add_argument("--candidate", required=True)
    kl.add_argument("--epsilon", type=float, default=1e-12)
    kl.add_argument("--max-kl", type=float, default=0.001)
    kl.set_defaults(func=_cmd_kl)

    gate = sub.add_parser("gate", help="full Q-030 gate receipt against frozen thresholds")
    gate.add_argument("--manifest", required=True)
    gate.add_argument("--gates", required=True)
    gate.add_argument("--reference", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--ledger")
    gate.add_argument("--batch", action="append", default=[], metavar="SIZE=CAPTURE")
    gate.add_argument("--long-context", action="append", default=[], metavar="TOKENS=CAPTURE")
    gate.set_defaults(func=_cmd_gate)

    pit = sub.add_parser("pit", help="KS uniformity gate over PIT samples")
    pit.add_argument("--samples", required=True)
    pit.add_argument("--alpha", type=float, default=0.01)
    pit.set_defaults(func=_cmd_pit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, json.JSONDecodeError, OracleError) as error:
        print(f"QUALITY_ORACLE_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
