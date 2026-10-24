# Owner reading, 2026-10-24: Muon, Parallax and the 8 GB model survey

Status: **READ, ASSESSED, NOT ACTED ON** except where a finding is named. This note keeps the reasoning
so a later session does not redo it. It makes no capability claim.

## Sources

| Source | Read |
| --- | --- |
| Liu et al., *Muon is Scalable for LLM Training* (Moonlight), arXiv:2502.16982v1 | In full |
| Zuo et al., *Parallax: Parameterized Local Linear Attention for Language Modeling*, arXiv:2605.29157v1 | Body and limitations in full; appendices skimmed |
| Ruan et al., *Muon Learns More Robust and Transferable Features than Adam*, arXiv:2606.09658v1 | Abstract, setup, results and conclusion; proofs not checked |
| Owner's survey *Modeles IA locaux, veille octobre 2026 (8 Go VRAM + 32 Go RAM)* | In full; outside this repository |
| K. Jordan, *Muon* blog post | **Not read**: the session's network policy refused `kellerjordan.github.io`. Moonlight section 2.1 restates the algorithm and its coefficients |

## What the texts support

**Muon.** Momentum, then about 5 Newton-Schulz iterations (coefficients 3.4445, -4.7750, 2.0315) that
orthogonalise the update of each 2-D hidden matrix; embeddings, head and norms stay on AdamW. Moonlight adds
weight decay and scales the update by `0.2 * sqrt(max(A, B))` so AdamW's learning rate and decay carry over.
Their scaling-law fit needs about 52% of AdamW's FLOPs at compute-optimal training. Moonlight itself is a
16B/3B-active MoE (DeepSeek-V3-small layout) trained on 5.7T tokens. Fine-tuning: Muon pretraining plus Muon
SFT was best, but Muon SFT of an AdamW-pretrained base (Qwen2.5-7B) was on par with or below Adam SFT
(their Table 7). Muon-trained matrices have flatter singular spectra, most visibly the router.

**Robust features.** GPT-2 124M and 354M (nanoGPT on FineWeb), ResNet-18 and ViT-S. Muon-pretrained models
have lower perplexity under character-level typos (their FineWeb10B-C), larger probe margins, higher
effective rank of hidden states, and fine-tune better (fine-tuning itself with Adam). The theory covers a
stylised one-layer problem.

**Parallax.** Softmax attention minus a covariance correction `Sigma_KV * rho` with `rho = W_R x`, one extra
query-like projection. The KV cache is unchanged (`rho` is per query, not cached); decode reads the same
K/V stream with about twice the FLOPs, and their Hopper decode kernel matches or beats FlashAttention 2/3.
Pretrained at 0.6B and 1.7B: better perplexity and downstream accuracy than softmax attention under Muon,
largely gone under AdamW (a learned gate closes the correction). Converting a pretrained Transformer by
adding `W_R = 0` and fine-tuning is proposed, not tested.

## Relevance to the engine

1. **The survey is the actionable one** (`docs/FINDINGS.md` F-026). It replaces the program's model,
   Qwen3-30B-A3B, with Qwen3.6-35B-A3B in `UD-IQ4_XS`. This engine tree refuses `IQ4_XS` at load, our CPU
   kernels cover `Q2_K`/`Q3_K` only, and the sizing fix covers the `qwen3moe` loader only. First step:
   `harness/model-inventory/gguf_inventory.py` on the lab's file. If the file shows linear-attention
   tensors (the engine's `qwen3_5_moe` path has linear and full attention layers), the KV and state plan
   (Q-028/Q-038) changes too.
2. **The survey's llama.cpp recipe is the fair competitor.** It keeps attention on the card and only some
   layers' experts in RAM (`-ngl 99 --n-cpu-moe N`, KV in `q8_0`). Our llama.cpp lane runs `--cpu-moe`
   (every expert in RAM). A comparison at equal VRAM needs a second llama.cpp lane with `N` chosen to use the
   card as our expert cache does. Not yet in the bench plan.
3. **The survey's speech picks feed W7 (Q-044).** STT: Parakeet TDT 0.6B v3, Canary 1B v2 (French WER
   figures are the survey's, from secondary sources; our WER oracle measures them before any choice).
   TTS: Kokoro-82M, Supertonic 3, Pocket TTS, small enough for the CPU. Which runtime runs them is open;
   the cohost broker treats them as separate tenants with their own budgets either way.
4. **Muon changes nothing in the engine.** Inference does not depend on the optimizer: a Muon-trained model
   is a GGUF like any other. Whether Muon-trained weights quantise better is **UNKNOWN**: flatter singular
   spectra are not the same as fewer element outliers, and none of the three papers measures quantisation.
   It is testable later with our KL oracle on a Muon/AdamW twin quantised to `Q2_K`; not scheduled.
5. **If the program trains a model** (a draft model for Q-046, a small assistant model): Muon for
   pretraining from scratch with the Moonlight recipe; for fine-tuning an AdamW-pretrained base, Moonlight's
   own Table 7 shows no gain, so Adam is the honest default.
6. **Parallax changes nothing now.** No released checkpoint at a useful size. If a model ships with it, the
   change is one attention op with the same KV cache; the MoE experts and our CPU path are untouched.
7. **Not actionable, worth remembering:** the typo-robustness result concerns the kind of noise an STT
   front end feeds an LLM, but it is measured at 124M-354M and does not say which deployable models are
   Muon-trained.

## Update: the owner's full report (same day)

Source: *Rapport complet, IA locale sur RTX 3070 (8 Go) + 32 Go RAM, octobre 2026*, 21 pages, read in
full; outside this repository. It merges the survey above with ten more categories (3D, formal maths,
OCR, embeddings, music and others). Its LLM recommendation is unchanged: Qwen3.6-35B-A3B with experts in
RAM. Checked against the engine sources, what it changes for the program:

1. **The machine is pinned down:** RTX 3070, 8 GB shared with the display, about 6.5-7 GB usable (the
   report's estimate), 32 GB RAM. Ampere has no FP8/NVFP4 acceleration, so GGUF or FP16/BF16 weights.
2. **The program's engine work covers one architecture** (F-027): residency, kernels and sizing exist
   for `qwen3moe` only. Of the report's candidates, Qwen3-Coder-30B-A3B (19 GB, its agents pick) and
   Qwen3-30B-A3B are `qwen3moe`; Qwen3.6-35B-A3B (expected `qwen35moe`, to be confirmed by the inventory),
   LFM2.5-24B-A2B (`lfm2moe`) and gpt-oss-20b (`gpt-oss`, MXFP4 experts) load as plain mistral.rs models.
   LFM2.5-8B-A1B (about 5 GB) fits the card whole and needs no offload.
3. **Its first recommendation is the cohost broker's workload** (Q-043): one large GPU model at a time,
   OCR (about 2 GB), embedding plus reranker (about 2 GB) and micro-LLMs kept in VRAM, MoE experts in RAM.
   On 6.5-7 GB usable that leaves about 2.5-3 GB for the MoE's attention, KV and expert cache. Those are
   the report's estimates; the broker's admission test is where they get checked, not assumed.
4. **Its voice architecture matches W7:** a VAD, STT, LLM, TTS cascade (Pipecat), because no omni
   "talker" is served well locally. The voice turn machine (Q-044) assumes the same cascade.
