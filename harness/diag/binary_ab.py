#!/usr/bin/env python3
"""Interleaved A/B of two `gnostral_capture` builds on one lane, gates before timing.

For comparing two engine builds (flash-attn against none, a patch against its absence) on the same
model, prompts and placement, in one session:

    python3 harness/diag/binary_ab.py --bin noflash=PATH --bin flash=PATH \
        --model /mnt/models/.../Qwen3-30B-A3B-Q2_K.gguf --layers 48 --experts 128 \
        --device cuda --out evidence/runs/<run>/

Each build captures one warmup (its own reference), then `--rounds` measured captures in
alternating order (A B, B A, A B ...). A run counts only if it exits 0, reports the expected
placement (engine patch 0008), reproduces its build's warmup in tokens and logprob bits, and, on
CUDA, shows its pid as an `nvidia-smi` compute app. Builds whose kernels differ are not expected to
agree with each other (parity class `self`); their token agreement is reported, not gated.

The verdict compares full ranges of decode tok/s: one build is faster only when its slowest run
beats the other's fastest; overlapping ranges are "inconclusive". Any failed gate refuses the
comparison. Separated ranges also happen by chance: if both builds were the same, n runs each would
separate with probability 2 / C(2n, n), 33% at n = 2, 10% at n = 3, 0.8% at n = 5 (two copies of one
binary were "ranked" at n = 2 on the cloud VM). The summary prints that probability next to the
result; use the default 5 rounds for any claim. Everything lands in `--out`: every capture,
`runs.jsonl` and `ab-summary.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness" / "lanes"))
import lanelib  # noqa: E402
from lanelib import logprobs_path, manifest_hash, run_sampled, sha256_file  # noqa: E402

MANIFEST = ROOT / "harness" / "quality" / "prompts" / "quality-prompts-32-20261023.json"
LANES = {
    "per-route": {"GNOSTRAL_MOE_EXECUTOR": "per-route"},
    "grouped": {"GNOSTRAL_MOE_EXECUTOR": "grouped"},
    "gnostral": {"GNOSTRAL_MOE_EXECUTOR": "grouped", "GNOSTRAL_CPU_EXPERT_KERNEL": "gnostral"},
}


def expected_placement(device: str, layers: int, gpu_layers: int | None) -> str:
    last = layers - 1
    if device == "cpu":
        return f"0-{last}:cpu;nm=cpu"
    g = layers if gpu_layers is None else gpu_layers
    if g >= layers:
        return f"0-{last}:cuda:0;nm=cuda:0"
    if g <= 0:
        return f"0-{last}:cpu;nm=cuda:0"
    return f"0-{g - 1}:cuda:0,{g}-{last}:cpu;nm=cuda:0"


def schedule(labels: list[str], rounds: int) -> list[tuple[str, str, int]]:
    """Warmups first, then measured rounds whose order alternates so drift hits both builds."""
    order = [(label, "warmup", 0) for label in labels]
    for r in range(1, rounds + 1):
        for label in (labels if r % 2 else labels[::-1]):
            order.append((label, "measured", r))
    return order


def verdict(runs: list[dict], labels: list[str]) -> dict:
    """Gates first; then a ranking only when the ranges do not overlap."""
    measured = [r for r in runs if r["phase"] == "measured"]
    failures = [f"{r['label']} round {r['round']}: {g}" for r in runs for g in r["gate_failures"]]
    per = {}
    for label in labels:
        v = sorted(r["metrics"]["decode_tok_s"] for r in measured
                   if r["label"] == label and r.get("metrics") and not r["gate_failures"])
        per[label] = {"n": len(v), "decode_tok_s": v,
                      **({"min": v[0], "median": statistics.median(v), "max": v[-1]} if v else {})}
    if failures:
        return {"comparison": "REFUSED", "reasons": failures, "builds": per}
    a, b = (per[label] for label in labels)
    if not a["n"] or not b["n"]:
        return {"comparison": "REFUSED", "reasons": ["no measured run"], "builds": per}
    if b["min"] > a["max"]:
        result = f"{labels[1]} faster"
    elif b["max"] < a["min"]:
        result = f"{labels[1]} slower"
    else:
        result = "inconclusive (ranges overlap)"
    return {"comparison": "COMPARED", "result": result, "builds": per,
            "chance_of_separated_ranges_if_equal": chance_of_separation(a["n"], b["n"])}


def chance_of_separation(na: int, nb: int) -> float:
    """Probability that exchangeable samples of sizes na and nb fall in disjoint ranges, either way."""
    return 2 / math.comb(na + nb, na)


def capture_once(args: argparse.Namespace, label: str, binary: Path, dest: Path, env: dict[str, str],
                 model_sha: str, promptset: str) -> dict:
    cmd = [str(binary), "--model-dir", str(args.model.parent), "--file", args.model.name,
           "--manifest", str(args.manifest), "--promptset-sha256", promptset, "--engine", args.lane,
           "--binary-sha256", sha256_file(binary), "--model-sha256", model_sha,
           "--max-tokens", str(args.max_tokens), "--out", str(dest), "--no-prefix-cache",
           "--topn", "2", "--logprobs-out", str(logprobs_path(dest))]
    cmd += ["--cpu"] if args.device == "cpu" else ["--gpu-layers", str(args.gpu_layers or args.layers)]
    t0 = time.monotonic()
    run = run_sampled(cmd, env, sample=args.device == "cuda")
    run["wall_s"] = round(time.monotonic() - t0, 2)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", action="append", required=True, metavar="LABEL=PATH",
                        help="exactly two builds; the first is the baseline")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True, help="routed experts per layer (for the profile)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lane", choices=sorted(LANES), default="gnostral")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-layers", type=int, help="layers on CUDA (default: all); never the automatic map")
    parser.add_argument("--rounds", type=int, default=5, help="measured runs per build (default 5, minimum 3)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="the whole set always runs: the oracle refuses partial captures")
    parser.add_argument("--cache-bytes", type=int, default=1, help="expert cache budget (default 1: every expert on the CPU)")
    parser.add_argument("--profile", type=Path, help="expert profile (default: an empty ranking written to --out)")
    args = parser.parse_args(argv)

    builds = [b.split("=", 1) for b in args.bin]
    if len(builds) != 2 or any(len(b) != 2 for b in builds) or builds[0][0] == builds[1][0]:
        parser.error("give exactly two --bin LABEL=PATH with different labels")
    if args.rounds < 3:
        parser.error("--rounds must be at least 3")
    labels = [label for label, _ in builds]
    binaries = {label: Path(path).resolve() for label, path in builds}
    args.model = args.model.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; every run gets a new directory")
    args.out.mkdir(parents=True, exist_ok=True)

    model_sha = sha256_file(args.model)
    promptset = manifest_hash(args.manifest)
    profile = args.profile
    if profile is None:
        profile = args.out / "profile-empty.json"
        profile.write_text(json.dumps({"schema": 1, "model_sha256": model_sha, "layers": args.layers,
                                       "experts_per_layer": args.experts, "ranking": []}))
    env = dict(os.environ)
    env.update({"GNOSTRAL_EXPERT_RESIDENCY_CPU": "1", "GNOSTRAL_EXPERT_CACHE_BYTES": str(args.cache_bytes),
                "GNOSTRAL_EXPERT_PROFILE": str(profile.resolve()), **LANES[args.lane]})
    placement = expected_placement(args.device, args.layers, args.gpu_layers)

    runs = []
    log = (args.out / "runs.jsonl").open("w")
    for label, phase, rnd in schedule(labels, args.rounds):
        dest = args.out / f"capture-{label}-{phase}-{rnd}.json"
        run = capture_once(args, label, binaries[label], dest, env, model_sha, promptset)
        lines = [l for l in run["stdout"].splitlines() if l.startswith("GNOSTRAL_")]
        got_placement = next((l.split(" ", 1)[1] for l in lines if l.startswith("GNOSTRAL_PLACEMENT ")), None)
        metrics = next((json.loads(l.split(" ", 1)[1]) for l in lines if l.startswith("GNOSTRAL_METRICS ")), None)
        gates = []
        if run["returncode"] != 0:
            gates.append(f"exit {run['returncode']}")
        if got_placement != placement:
            gates.append(f"placement {got_placement!r} != {placement!r}")
        if args.device == "cuda" and run["pid_seen"] is not True:
            gates.append("CUDA not proven (pid never listed by nvidia-smi)")
        repro = None
        if run["returncode"] == 0:
            repro = lanelib.self_repro_line(args.manifest, args.out / f"self-ref-{label}.json", dest)
            if repro.startswith("GNOSTRAL_SELF_REPRO FAIL"):
                gates.append(repro)
        record = {"label": label, "phase": phase, "round": rnd, "binary_sha256": sha256_file(binaries[label]),
                  "wall_s": run["wall_s"], "returncode": run["returncode"], "placement": got_placement,
                  "self_repro": repro, "metrics": metrics, "gate_failures": gates,
                  "device": lanelib.device_line(run) if args.device == "cuda" else None,
                  "stderr_tail": run["stderr"][-2000:] if run["returncode"] else None}
        runs.append(record)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"{label:>10} {phase:>8} {rnd}: {metrics and round(metrics['decode_tok_s'], 2)} tok/s "
              f"{'OK' if not gates else 'GATE ' + '; '.join(gates)}", flush=True)
    log.close()

    result = verdict(runs, labels)
    agreement = None
    refs = [args.out / f"self-ref-{label}.json" for label in labels]
    if all(r.exists() for r in refs):
        agreement = lanelib.agreement(args.manifest, refs[0], refs[1])
    summary = {"schema": "gnostral.binary-ab/1", "lane": args.lane, "device": args.device,
               "expected_placement": placement, "model_sha256": model_sha, "promptset_sha256": promptset,
               "max_tokens": args.max_tokens, "rounds": args.rounds,
               "cache_bytes": args.cache_bytes, "profile_sha256": sha256_file(profile),
               "builds": {label: {"sha256": sha256_file(binaries[label])} for label in labels},
               "verdict": result, "cross_build_agreement": agreement,
               "note": "parity class self: each build is gated on reproducing its own warmup"}
    (args.out / "ab-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["comparison"] == "COMPARED" else 1


if __name__ == "__main__":
    sys.exit(main())
