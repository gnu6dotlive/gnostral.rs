"""llama.cpp self-consistency: the decode path versus one prefill of prompt plus generated prefix.

The llama.cpp counterpart of the engine's `gnostral_prefill_check` (patch 0010). For every prompt and
step i < STEPS, one `/completion` with the token ids of the chat-templated prompt plus the first i
captured tokens, `n_predict 1`, greedy, prompt cache off, must predict the captured token i.

Usage: llama_prefill_check.py SERVER_BIN MODEL MANIFEST CAPTURE LOGPROBS STEPS OUT
(LOGPROBS is accepted for symmetry with the lane outputs and not read.)
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

server_bin, model, manifest, capture, logprobs, steps, out = sys.argv[1:8]
steps = int(steps)
port = 18123
base = f"http://127.0.0.1:{port}"
proc = subprocess.Popen([server_bin, "-m", model, "--port", str(port), "--host", "127.0.0.1", "-np", "1",
                         "-ngl", "0", "-t", "4", "-c", "4096", "-s", "0"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
try:
    for _ in range(240):
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    break
        except Exception:
            time.sleep(0.5)
    prompts = json.loads(Path(manifest).read_text())["prompts"]
    gen = {c["prompt_id"]: c["token_ids"] for c in json.loads(Path(capture).read_text())["captures"]}
    results = []
    for p in prompts:
        text = post("/apply-template", {"messages": [{"role": "user", "content": p["text"]}]})["prompt"]
        ids = post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        rows = []
        for i in range(steps):
            done = post("/completion", {"prompt": ids + gen[p["id"]][:i], "n_predict": 1, "temperature": 0, "top_k": 1,
                                        "samplers": ["top_k"], "seed": 0, "cache_prompt": False, "n_probs": 2,
                                        "return_tokens": True})
            probs = done.get("completion_probabilities") or []
            top = [[t["id"], t["logprob"]] for t in probs[0]["top_logprobs"]] if probs else None
            rows.append({"step": i, "decode_token": gen[p["id"]][i], "prefill_token": done["tokens"][0], "prefill_top2": top})
        results.append({"prompt_id": p["id"], "prompt_tokens": len(ids), "rows": rows})
    Path(out).write_text(json.dumps(results, indent=1))
finally:
    proc.terminate(); proc.wait(30)
