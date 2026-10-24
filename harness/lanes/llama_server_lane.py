"""One llama.cpp lane run for `gnostral-bench` (Q-025 three-way, plan v2 `parity_class: self`).

Starts `llama-server`, sends every frozen prompt once, writes a
`gnostral.quality.capture/1` document plus top-2 logprobs, stops the server and
prints the lines the bench reads.

Sampling is strictly greedy: `temperature 0`, `top_k 1` and `top_k` as the only
sampler, fixed seed, prompt cache off. Prompts go through the server's own chat
template (`/apply-template`), as `gnostral_capture` uses the GGUF's template.

Metrics use the definitions of `gnostral_capture`: decode tok/s is generated
tokens over decode time summed over prompts, prefill tok/s the same for prompt
tokens, `ttft_ms` the median prompt-processing time. Server start-up and model
load are not timed.

Placement: llama.cpp logs `offloaded A/B layers to GPU`; the lane reports
`llama.cpp offloaded=A/B` (or `offloaded=none` when the build has no GPU
backend) plus whether `--cpu-moe` was given. Like our engine, llama.cpp can fit
its parameters to free memory (`--fit`), so this placement is pinned by the plan
and any drift fails the lane. Pass `-ngl` explicitly; the lane refuses to run
without it.

Logprobs: `llama-server` records a token's probabilities only once the generated text
is complete UTF-8, so a token that ends inside a multi-byte character has none (source:
`tools/server/server-context.cpp`, `add_token` is skipped while the text is incomplete).
The token list (`return_tokens`) is complete. The lane therefore records the server's
probability entries as reported, in order, each tagged by its token id, and checks that
those ids are an ordered subsequence of the tokens. Self-reproducibility compares the
full token list and those entries bit for bit; it does not pretend they are one per token.

Identity: the plan pins the `llama-server` executable. A shared-library build can
change without the executable changing, so the lane also prints
`GNOSTRAL_LANE_DETAIL build_tree_sha256=...` over the executable and the
`libllama*`/`libggml*` libraries beside it; build with `-DBUILD_SHARED_LIBS=OFF`
to make the executable hash sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lanelib  # noqa: E402

CAPTURE_SCHEMA = "gnostral.quality.capture/1"
OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
HEALTH_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 30.0


class LaneError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise LaneError(f"{path}: HTTP {error.code}: {error.read()[:300]!r}") from error


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            return json.loads(response.read()).get("status") == "ok"
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def placement_from_log(log: str, server_args: list[str]) -> str | None:
    """Canonical placement from the server's own load log; None if the log is ambiguous."""
    found = OFFLOAD.findall(log)
    if len(set(found)) > 1:
        return None
    offloaded = f"{found[0][0]}/{found[0][1]}" if found else "none"
    cpu_moe = "--cpu-moe" in server_args or "-cmoe" in server_args
    return f"llama.cpp offloaded={offloaded} cpu_moe={'on' if cpu_moe else 'off'}"


def build_tree_hash(server_bin: Path) -> str:
    files = [server_bin] + sorted(p for p in server_bin.parent.iterdir()
                                  if p.is_file() and p.name.startswith(("libllama", "libggml")))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + lanelib.sha256_file(f.resolve()).encode() + b"\n")
    return h.hexdigest()


def is_subsequence(needle: list[int], haystack: list[int]) -> bool:
    it = iter(haystack)
    return all(any(h == n for h in it) for n in needle)


def completion_step(probs: dict) -> tuple[int, list[list]]:
    top = probs.get("top_logprobs")
    if not isinstance(top, list) or not top:
        raise LaneError("a generated token has no top_logprobs")
    return probs["id"], [[t["id"], t["logprob"]] for t in top]


def capture_prompts(base: str, prompts: list[dict], args: argparse.Namespace) -> tuple[list, list, dict]:
    captures, logprob_docs = [], []
    totals = {"predicted_n": 0, "predicted_ms": 0.0, "prompt_n": 0, "prompt_ms": 0.0}
    prompt_ms = []
    for prompt in prompts:
        text = post(base, "/apply-template", {"messages": [{"role": "user", "content": prompt["text"]}]})["prompt"]
        done = post(base, "/completion", {
            "prompt": text, "n_predict": args.max_tokens, "temperature": 0, "top_k": 1,
            "samplers": ["top_k"], "seed": args.seed, "cache_prompt": False, "n_probs": 2,
            "return_tokens": True,
        })
        tokens = done.get("tokens")
        probs = done.get("completion_probabilities")
        if not isinstance(tokens, list) or not tokens or not isinstance(probs, list):
            raise LaneError(f"prompt {prompt['id']}: no tokens or probabilities returned")
        steps = [completion_step(p) for p in probs]
        if not is_subsequence([s[0] for s in steps], tokens):
            raise LaneError(f"prompt {prompt['id']}: probability entries are not an ordered subsequence of the tokens")
        captures.append({"prompt_id": prompt["id"], "prompt_sha256": prompt["sha256"], "token_ids": tokens})
        logprob_docs.append({"prompt_id": prompt["id"], "steps": [s[1] for s in steps],
                             "alignment": "llama-server entries; tokens inside an incomplete UTF-8 character have none",
                             "entry_token_ids": [s[0] for s in steps]})
        t = done["timings"]
        for key in totals:
            totals[key] += t[key]
        prompt_ms.append(t["prompt_ms"])
    rate = lambda n, ms: n / (ms / 1e3) if ms > 0 else None
    metrics = {
        "decode_tok_s": rate(totals["predicted_n"], totals["predicted_ms"]),
        "prefill_tok_s": rate(totals["prompt_n"], totals["prompt_ms"]),
        "ttft_ms": statistics.median_low(prompt_ms) if prompt_ms else None,
    }
    return captures, logprob_docs, metrics


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run(args: argparse.Namespace) -> int:
    if args.stop_file and Path(args.stop_file).exists():
        sys.stderr.write("skipped: an earlier lane failed parity\n")
        return 5
    server_args = shlex.split(args.server_args)
    if not any(a in server_args for a in ("-ngl", "--gpu-layers", "--n-gpu-layers")):
        raise LaneError("pass -ngl explicitly: an automatic layer count is not a pinnable placement")
    out = Path(args.out)
    server_bin = Path(args.server_bin).resolve()
    model = Path(args.model).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    latest = out / f"capture-{args.config}-{args.engine}-latest.json"
    log_path = out / f"server-{args.config}-{args.engine}-latest.log"
    base = f"http://127.0.0.1:{free_port() if args.port == 0 else args.port}"
    cmd = [str(server_bin), "-m", str(model), "--port", base.rsplit(":", 1)[1], "--host", "127.0.0.1",
           "-np", "1", "-s", str(args.seed), *server_args]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    sampler = lanelib.GpuSampler(proc.pid, args.device == "cuda").start()
    try:
        if not lanelib.wait_until(lambda: healthy(base) or proc.poll() is not None, HEALTH_TIMEOUT_S):
            raise LaneError("server did not become healthy")
        if proc.poll() is not None:
            raise LaneError(f"server exited with {proc.returncode} before serving; see {log_path.name}")
        captures, logprob_docs, metrics = capture_prompts(base, manifest["prompts"], args)
    finally:
        sampler.stop()
        stop(proc)
    stats = sampler.result()
    placement = placement_from_log(log_path.read_text(errors="replace"), server_args)
    document = {
        "schema": CAPTURE_SCHEMA,
        "promptset_sha256": lanelib.manifest_hash(Path(args.manifest)),
        "lane": {"engine": args.engine, "binary_sha256": lanelib.sha256_file(server_bin),
                 "model_sha256": lanelib.sha256_file(model), "placement": placement,
                 "build_tree_sha256": build_tree_hash(server_bin)},
        "sampling": {"temperature": 0, "seed": args.seed, "max_tokens": args.max_tokens,
                     "top_k": 1, "samplers": ["top_k"]},
        "captures": captures,
    }
    latest.write_text(json.dumps(document, indent=2), encoding="utf-8")
    lanelib.logprobs_path(latest).write_text(json.dumps(logprob_docs), encoding="utf-8")

    print(f"GNOSTRAL_PLACEMENT {placement or 'UNKNOWN'}")
    print(f"GNOSTRAL_LANE_DETAIL build_tree_sha256={document['lane']['build_tree_sha256']}")
    if args.device == "cuda":
        print(lanelib.device_line(stats))
        if stats["pid_seen"] is False:
            sys.stderr.write("CUDA not proven: the server pid never appeared as an nvidia-smi compute app\n")
            return 3
        metrics["vram_used_mib"] = stats["peak_card_mib"]
    print(lanelib.self_repro_line(Path(args.manifest), out / f"self-ref-{args.config}-{args.engine}.json", latest))
    if args.reference:
        print("GNOSTRAL_AGREEMENT " + json.dumps(lanelib.agreement(Path(args.manifest), Path(args.reference), latest)))
    print("GNOSTRAL_METRICS " + json.dumps(metrics))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--server-args", required=True,
                        help='llama-server arguments in one string; use the = form so argparse does not read them '
                             'as options: --server-args="-ngl 99 --cpu-moe -t 8 -c 4096 --fit off"')
    parser.add_argument("--engine", default="llama.cpp")
    parser.add_argument("--config", default="cpu-experts")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference", help="reference capture for the (reported, ungated) agreement")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--stop-file", help="exit 5 without running when this file exists")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except LaneError as error:
        sys.stderr.write(f"llama lane: {error}\n")
        return 4


if __name__ == "__main__":
    sys.exit(main())
