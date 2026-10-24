"""A stand-in for `llama-server` used by the lane tests. Not an engine.

Implements `/health`, `/apply-template` and `/completion` with the response shapes of
llama.cpp b11857, including its behaviour of omitting probability entries for tokens
that end inside an incomplete UTF-8 character. Output is a deterministic function of
the prompt, so two runs agree; `FAKE_LLAMA_DRIFT=1` perturbs one logprob by one ULP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

INCOMPLETE = {200, 210}  # token ids the fake treats as UTF-8 lead bytes


def tokens_for(prompt: str, n: int) -> list[int]:
    digest = hashlib.sha256(prompt.encode()).digest()
    return [digest[i % len(digest)] for i in range(n)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the server log to what the lane parses
        pass

    def _send(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"status": "ok"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            text = body["messages"][0]["content"]
            self._send({"prompt": f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"})
            return
        tokens = tokens_for(body["prompt"], body["n_predict"])
        probs = []
        for i, t in enumerate(tokens):
            if t in INCOMPLETE:
                continue
            lp = -0.125 * (i % 4)
            if os.environ.get("FAKE_LLAMA_DRIFT") == "1" and i == 1:
                lp = math.nextafter(lp, -1.0)
            probs.append({"id": t, "logprob": lp,
                          "top_logprobs": [{"id": t, "logprob": lp}, {"id": (t + 1) % 256, "logprob": lp - 1}]})
        self._send({"tokens": tokens, "completion_probabilities": probs,
                    "timings": {"prompt_n": 40, "prompt_ms": 4.0, "predicted_n": len(tokens),
                                "predicted_ms": 2.0 * len(tokens)}})


def main() -> int:
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    ngl = args[args.index("-ngl") + 1] if "-ngl" in args else "auto"
    print(f"load_tensors: offloaded {ngl}/3 layers to GPU", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
