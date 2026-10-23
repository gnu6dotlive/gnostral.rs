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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_MANIFEST = "gnostral.quality.promptset/1"
SCHEMA_CAPTURE = "gnostral.quality.capture/1"

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
# CLI
# ---------------------------------------------------------------------------


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

    kl = sub.add_parser("kl", help="KL divergence between two normalised distributions")
    kl.add_argument("--reference", required=True)
    kl.add_argument("--candidate", required=True)
    kl.add_argument("--epsilon", type=float, default=1e-12)
    kl.add_argument("--max-kl", type=float, default=0.01)
    kl.set_defaults(func=_cmd_kl)

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
