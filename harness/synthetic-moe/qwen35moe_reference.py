"""Float64 reference forward pass of a `qwen35moe` GGUF, to judge two engines that disagree.

Written from the model's official definition, transformers 5.20.2
`models/qwen3_5_moe/modeling_qwen3_5_moe.py`, carried into GGUF space with the transformations of
llama.cpp b11857's converter (`conversion/qwen.py`, `Qwen3NextModel.modify_tensors` and
`_LinearAttentionVReorderBase`):

- norms other than the GDN gated norm are stored as 1 + w; `Qwen3_5MoeRMSNorm` multiplies by 1 + w,
  so the stored value is the multiplier. The gated norm (`Qwen3_5MoeRMSNormGated`) is stored raw.
- `ssm_a` = -exp(A_log), so g = ssm_a * softplus(a + dt_bias) is the HF `g`.
- conv1d is stored squeezed as [channels, kernel].
- V heads (and everything indexed by them: V rows of qkv, z, a, b, dt_bias, A_log, conv V channels,
  out_proj columns) are reordered from HF's grouped order to tiled order when k_heads != v_heads.
  `_reorder_v_heads` reshapes [k_heads, v_per_k, d] and swaps the first two axes, so GGUF head j is
  HF group j % k_heads. HF's `repeat_interleave` pairs value head i with key head i // v_per_k, which
  in GGUF order is key head j % k_heads.

Everything is computed in float64 with no cache: each call runs the whole sequence, the recurrent
form of the gated delta rule (`torch_recurrent_gated_delta_rule`), and returns log-probabilities at
every position. Compare engines by teacher forcing: feed each engine's own tokens, so one early
difference does not change every later position.

Quantised expert tensors are dequantised with gguf-py; for a clean comparison use F32 experts
(`make_qwen35moe_gguf.py --expert-type f32`), which leaves only float rounding between engines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARCH = "qwen35moe"
SPECIALS = {"<|im_start|>": 256, "<|im_end|>": 257}


def load(path: Path, gguf_py: Path):
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # noqa: PLC0415
    from gguf.quants import dequantize  # noqa: PLC0415

    reader = GGUFReader(str(path))

    def field(name):
        f = reader.fields[f"{ARCH}.{name}"]
        values = [f.parts[i] for i in f.data]
        return [v.tolist()[0] for v in values] if len(values) > 1 else values[0].tolist()[0]

    hp = {k: field(k) for k in [
        "embedding_length", "block_count", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "rope.dimension_count", "rope.freq_base", "attention.layer_norm_rms_epsilon",
        "expert_count", "expert_used_count", "full_attention_interval", "ssm.conv_kernel", "ssm.inner_size",
        "ssm.state_size", "ssm.group_count", "ssm.time_step_rank"]}
    tensors = {}
    for t in reader.tensors:
        data = t.data if t.tensor_type == 0 else dequantize(t.data, t.tensor_type)
        tensors[t.name] = np.asarray(data, dtype=np.float64)
    return hp, tensors


def rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def tiled_key_of_value(v_heads: int, k_heads: int) -> np.ndarray:
    """Key head of each GGUF value head: tiled order pairs value head j with key head j % k_heads."""
    return np.arange(v_heads) % k_heads


def gated_delta_rule(q, k, v, g, beta):
    """`torch_recurrent_gated_delta_rule` after its l2norm and query scaling, one sequence.

    q, k: [T, heads, dk]; v: [T, heads, dv]; g, beta: [T, heads]. Returns [T, heads, dv].
    """
    T, heads, dk = k.shape
    state = np.zeros((heads, dk, v.shape[-1]))
    out = np.zeros_like(v)
    for t in range(T):
        state = state * np.exp(g[t])[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", state, k[t])
        delta = (v[t] - kv_mem) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", state, q[t])
    return out


def gdn(h, w, p, hp):
    """`Qwen3_5MoeGatedDeltaNet.forward`, no cache, recurrent rule, GGUF tiled V order."""
    T = h.shape[0]
    nk, nv, dk = hp["ssm.group_count"], hp["ssm.time_step_rank"], hp["ssm.state_size"]
    dv = hp["ssm.inner_size"] // nv
    key_dim = nk * dk
    eps = hp["attention.layer_norm_rms_epsilon"]
    qkv = h @ w[p + "attn_qkv.weight"].T
    kernel = w[p + "ssm_conv1d.weight"]
    K = kernel.shape[1]
    padded = np.vstack([np.zeros((K - 1, qkv.shape[1])), qkv])
    # F.conv1d(padding=K-1)[:T]: out[t] = sum_k w[k] * x[t + k - (K - 1)]
    conv = silu(sum(kernel[:, k] * padded[k:k + T] for k in range(K)))
    q = conv[:, :key_dim].reshape(T, nk, dk)
    k = conv[:, key_dim:2 * key_dim].reshape(T, nk, dk)
    v = conv[:, 2 * key_dim:].reshape(T, nv, dv)
    z = (h @ w[p + "attn_gate.weight"].T).reshape(T, nv, dv)
    beta = sigmoid(h @ w[p + "ssm_beta.weight"].T)
    g = w[p + "ssm_a"] * np.logaddexp(0.0, h @ w[p + "ssm_alpha.weight"].T + w[p + "ssm_dt.bias"])
    key_of_value = tiled_key_of_value(nv, nk)
    q, k = l2norm(q[:, key_of_value]), l2norm(k[:, key_of_value])
    out = gated_delta_rule(q / np.sqrt(dk), k, v, g, beta)
    out = rms(out, eps) * w[p + "ssm_norm.weight"] * silu(z)
    return out.reshape(T, nv * dv) @ w[p + "ssm_out.weight"].T


def attention(h, w, p, hp):
    """`Qwen3_5MoeAttention.forward` with eager attention and text-only (equal) MRoPE positions."""
    T = h.shape[0]
    nh, nkv, hd = hp["attention.head_count"], hp["attention.head_count_kv"], hp["attention.key_length"]
    eps, rd = hp["attention.layer_norm_rms_epsilon"], hp["rope.dimension_count"]
    qg = (h @ w[p + "attn_q.weight"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    q = rms(q, eps) * w[p + "attn_q_norm.weight"]
    k = rms((h @ w[p + "attn_k.weight"].T).reshape(T, nkv, hd), eps) * w[p + "attn_k_norm.weight"]
    v = (h @ w[p + "attn_v.weight"].T).reshape(T, nkv, hd)
    inv_freq = 1.0 / (hp["rope.freq_base"] ** (np.arange(0, rd, 2) / rd))
    freqs = np.arange(T)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]

    def rope(x):
        rot = x[..., :rd]
        half = np.concatenate([-rot[..., rd // 2:], rot[..., :rd // 2]], axis=-1)
        return np.concatenate([rot * cos + half * sin, x[..., rd:]], axis=-1)

    q, k = rope(q), rope(k)
    rep = nh // nkv
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) * hd ** -0.5
    scores = scores + np.triu(np.full((T, T), -np.inf), 1)[None]
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    probs = scores / scores.sum(-1, keepdims=True)
    out = np.einsum("hts,shd->thd", probs, v).reshape(T, nh * hd) * sigmoid(gate)
    return out @ w[p + "attn_output.weight"].T


def moe(h, w, p, hp):
    """`Qwen3_5MoeSparseMoeBlock.forward`: softmax router, top-k renormalised, sigmoid-gated shared expert."""
    logits = h @ w[p + "ffn_gate_inp.weight"].T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = probs / probs.sum(-1, keepdims=True)
    top = np.argsort(-probs, axis=-1, kind="stable")[:, :hp["expert_used_count"]]
    gate_w, up_w, down_w = (w[p + f"ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
    out = np.zeros_like(h)
    for t in range(h.shape[0]):
        weights = probs[t, top[t]] / probs[t, top[t]].sum()
        for e, wt in zip(top[t], weights):
            out[t] += wt * (down_w[e] @ (silu(gate_w[e] @ h[t]) * (up_w[e] @ h[t])))
    shared = (silu(h @ w[p + "ffn_gate_shexp.weight"].T) * (h @ w[p + "ffn_up_shexp.weight"].T)) \
        @ w[p + "ffn_down_shexp.weight"].T
    return out + sigmoid(h @ w[p + "ffn_gate_inp_shexp.weight"])[:, None] * shared


def forward_logprobs(tokens: list[int], w: dict, hp: dict) -> np.ndarray:
    eps = hp["attention.layer_norm_rms_epsilon"]
    x = w["token_embd.weight"][tokens]
    for i in range(hp["block_count"]):
        p = f"blk.{i}."
        h = rms(x, eps) * w[p + "attn_norm.weight"]
        full = (i + 1) % hp["full_attention_interval"] == 0
        x = x + (attention(h, w, p, hp) if full else gdn(h, w, p, hp))
        x = x + moe(rms(x, eps) * w[p + "post_attention_norm.weight"], w, p, hp)
    logits = (rms(x, eps) * w["output_norm.weight"]) @ w["output.weight"].T
    m = logits.max(-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))


def chat_tokens(text: str) -> list[int]:
    """The fixture's chat template under its byte-level vocabulary: specials, then one token per byte."""
    rendered = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    out, rest = [], rendered
    while rest:
        special = next((s for s in SPECIALS if rest.startswith(s)), None)
        if special:
            out.append(SPECIALS[special])
            rest = rest[len(special):]
        else:
            nxt = min([rest.find(s) for s in SPECIALS if rest.find(s) > 0] or [len(rest)])
            out.extend(rest[:nxt].encode("utf-8"))
            rest = rest[nxt:]
    return out


def engine_steps(capture: dict, logprobs: list) -> dict:
    """Per prompt: generated tokens and {step: [[token, logprob], ...]} in either capture flavour."""
    tokens = {c["prompt_id"]: c["token_ids"] for c in capture["captures"]}
    out = {}
    for doc in logprobs:
        pid, steps = doc["prompt_id"], doc["steps"]
        if "entry_token_ids" in doc:  # llama.cpp lane: entries are an ordered subsequence of tokens
            by_step, j = {}, 0
            for i, tok in enumerate(tokens[pid]):
                if j < len(doc["entry_token_ids"]) and doc["entry_token_ids"][j] == tok:
                    by_step[i] = steps[j]
                    j += 1
        else:
            by_step = dict(enumerate(steps))
        out[pid] = (tokens[pid], by_step)
    return out


def judge(model: Path, gguf_py: Path, manifest: Path, capture: Path, logprobs: Path) -> dict:
    hp, w = load(model, gguf_py)
    prompts = {p["id"]: p["text"] for p in json.loads(manifest.read_text(encoding="utf-8"))["prompts"]}
    steps = engine_steps(json.loads(capture.read_text()), json.loads(logprobs.read_text()))
    gaps, argmax_disagree, total, per_prompt = [], 0, 0, {}
    for pid, (generated, by_step) in steps.items():
        prompt = chat_tokens(prompts[pid])
        ref = forward_logprobs(prompt + generated, w, hp)
        disagree = []
        for i, tok in enumerate(generated):
            row = ref[len(prompt) - 1 + i]
            total += 1
            if int(np.argmax(row)) != tok:
                argmax_disagree += 1
                disagree.append(i)
            for t, lp in by_step.get(i, []):
                gaps.append(abs(float(row[t]) - lp))
        per_prompt[pid] = {"prompt_tokens": len(prompt), "steps_where_reference_argmax_differs": disagree}
    gaps.sort()
    q = lambda f: gaps[min(len(gaps) - 1, int(f * len(gaps)))]  # noqa: E731
    return {"steps": total, "reference_argmax_differs": argmax_disagree,
            "logprob_gap": {"n": len(gaps), "median": q(0.5), "p99": q(0.99), "max": gaps[-1]},
            "per_prompt": per_prompt}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf-py", required=True, help="llama.cpp's gguf-py directory (reader and dequantisers)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--logprobs", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = judge(Path(args.model), Path(args.gguf_py), Path(args.manifest), Path(args.capture), Path(args.logprobs))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
