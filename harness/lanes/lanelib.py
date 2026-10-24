"""Shared plumbing for benchmark lane scripts (Q-031 plan v2).

A lane script runs one engine once and prints the lines `gnostral-bench` reads:
`GNOSTRAL_PLACEMENT`, `GNOSTRAL_SELF_REPRO`, `GNOSTRAL_PARITY` (exact lanes),
`GNOSTRAL_METRICS`, plus informational lines (`GNOSTRAL_DEVICE`,
`GNOSTRAL_AGREEMENT`). This module holds what every lane needs so that two lanes
measure the same things the same way:

- identity hashes, cached per process;
- a GPU sampler (whole-card memory every 200 ms, and whether the engine's pid is
  listed as a CUDA compute app) that never invents a value: no `nvidia-smi`, no number;
- the self-reproducibility rule: a lane's first capture in a configuration is its
  reference, every later capture must match it in tokens and logprob bits.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "harness" / "quality" / "quality_oracle.py"
SAMPLE_PERIOD_S = 0.2


@functools.lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def manifest_hash(manifest: Path) -> str:
    out = subprocess.run([sys.executable, "-I", str(ORACLE), "manifest-hash", str(manifest)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["manifest_sha256"]


def smi(*query: str) -> list[str] | None:
    """Rows of an nvidia-smi CSV query, or None when nvidia-smi is absent or fails."""
    try:
        run = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return run.stdout.strip().splitlines() if run.returncode == 0 else None


def card_used_mib() -> float | None:
    rows = smi("--query-gpu=memory.used")
    try:
        return float(rows[0]) if rows else None
    except ValueError:
        return None


def process_used_mib(pid: int) -> tuple[bool | None, float | None]:
    """(pid listed as a compute app, its VRAM); (None, None) when nvidia-smi cannot say."""
    rows = smi("--query-compute-apps=pid,used_memory")
    if rows is None:
        return None, None
    for row in rows:
        fields = [f.strip() for f in row.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return True, float(fields[1])
            except ValueError:
                return True, None
    return False, None


class GpuSampler:
    """Samples the card while a process runs; `result()` after `stop()`."""

    def __init__(self, pid: int, enabled: bool = True) -> None:
        self.pid = pid
        self.enabled = enabled
        self.ambient = card_used_mib() if enabled else None
        self.peak_card: float | None = None
        self.peak_proc: float | None = None
        self.seen: bool | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        card = card_used_mib()
        if card is not None:
            self.peak_card = card if self.peak_card is None else max(self.peak_card, card)
        listed, used = process_used_mib(self.pid)
        if listed is not None:
            self.seen = bool(self.seen) or listed
        if used is not None:
            self.peak_proc = used if self.peak_proc is None else max(self.peak_proc, used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(SAMPLE_PERIOD_S)

    def start(self) -> "GpuSampler":
        if self.enabled:
            self._thread.start()
        return self

    def stop(self) -> None:
        if self.enabled:
            self._stop.set()
            self._thread.join()
            self._sample()

    def result(self) -> dict:
        return {"ambient_mib": self.ambient, "peak_card_mib": self.peak_card,
                "pid_seen": self.seen, "peak_process_mib": self.peak_proc}


def run_sampled(cmd: list[str], env: dict[str, str], sample: bool = True) -> dict:
    """Run a command to completion while sampling the card; no number is invented."""
    with tempfile.TemporaryFile("w+") as out_f, tempfile.TemporaryFile("w+") as err_f:
        proc = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=err_f, text=True)
        sampler = GpuSampler(proc.pid, sample).start()
        proc.wait()
        sampler.stop()
        out_f.seek(0)
        err_f.seek(0)
        return {"returncode": proc.returncode, "stdout": out_f.read(), "stderr": err_f.read(),
                **sampler.result()}


def fmt(v: float | bool | None) -> str:
    return "UNKNOWN" if v is None else str(round(v, 1) if isinstance(v, float) else v).lower()


def device_line(stats: dict) -> str:
    return (f"GNOSTRAL_DEVICE cuda pid_seen={fmt(stats['pid_seen'])} "
            f"peak_process_mib={fmt(stats['peak_process_mib'])} "
            f"peak_card_mib={fmt(stats['peak_card_mib'])} ambient_mib={fmt(stats['ambient_mib'])}")


def logprobs_path(capture: Path) -> Path:
    return capture.with_name(capture.stem + ".logprobs.json")


def self_repro_line(manifest: Path, ref: Path, latest: Path) -> str:
    """The first capture of a lane in a configuration (the warmup) becomes its reference."""
    if not ref.exists():
        ref.write_bytes(latest.read_bytes())
        logprobs_path(ref).write_bytes(logprobs_path(latest).read_bytes())
        return "GNOSTRAL_SELF_REPRO REFERENCE"
    done = subprocess.run(
        [sys.executable, "-I", str(ORACLE), "self-repro", "--manifest", str(manifest),
         "--first", str(ref), "--second", str(latest),
         "--first-logprobs", str(logprobs_path(ref)), "--second-logprobs", str(logprobs_path(latest))],
        capture_output=True, text=True)
    if done.returncode == 0:
        return "GNOSTRAL_SELF_REPRO PASS"
    try:
        result = json.loads(done.stdout)
        diff = result.get("first_difference") or {}
        detail = (f"{result['verdict']} prompt={diff.get('prompt_id')} kind={diff.get('kind')} "
                  f"index={diff.get('index')} {';'.join(result.get('reasons', []))}").strip()
    except (json.JSONDecodeError, KeyError):
        detail = "oracle error: " + (done.stderr.strip().splitlines() or ["no output"])[-1]
    return f"GNOSTRAL_SELF_REPRO FAIL {detail}"


def agreement(manifest: Path, reference: Path, candidate: Path) -> dict:
    """Token agreement with the reference lane, as Q-030 defines it (`token_agreement`: matched-prefix
    tokens over reference tokens). Reported, never a gate here."""
    done = subprocess.run([sys.executable, "-I", str(ORACLE), "parity", "--manifest", str(manifest),
                           "--reference", str(reference), "--candidate", str(candidate)],
                          capture_output=True, text=True)
    try:
        results = json.loads(done.stdout)["results"]
    except (json.JSONDecodeError, KeyError):
        return {"verdict": "UNKNOWN", "detail": (done.stderr.strip().splitlines() or ["no output"])[-1]}
    ref_tokens = {c["prompt_id"]: len(c["token_ids"])
                  for c in json.loads(Path(reference).read_text(encoding="utf-8"))["captures"]}
    total = sum(ref_tokens.values())
    matched = sum(r["matched_prefix"] for r in results)
    return {
        "reference": json.loads(Path(reference).read_text(encoding="utf-8"))["lane"]["engine"],
        "prompts": len(results),
        "prompts_identical": sum(r["verdict"] == "PARITY_PASS" for r in results),
        "token_agreement": matched / total if total else None,
        "blocked": sum(r["verdict"] == "BLOCKED" for r in results),
    }


def wait_until(predicate, timeout_s: float, period_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(period_s)
    return predicate()
