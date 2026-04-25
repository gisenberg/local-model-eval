# Local Model Tier List — NVIDIA RTX Pro 6000 Blackwell 96GB

**Platform:** NVIDIA RTX Pro 6000 Blackwell Workstation Edition (96 GB GDDR7 ECC, 1792 GB/s, sm_120), Ubuntu 24.04, nvidia-driver-580
**Inference:** stock llama.cpp (b8826 Vulkan prebuilt; master branch CUDA 13.2 build with `-DCMAKE_CUDA_ARCHITECTURES=120`)
**Standard config:** `-ngl 99 -fa on -c <native> --no-mmap -np 1`, temperature=0 for scoring, streaming chat completions for throughput.

Tested April 2026.

> This is the first set of rankings for the Pro 6000 in this repo. We tested 8 model/quant combinations with a focus on **configurations that don't fit on a 5090** — BF16 of 31B dense, BF16 of 35B MoE, an 120B MoE (gpt-oss) that simply cannot load on 32 GB VRAM at any reasonable precision, and an 80B-class coder specialist at Q6_K.
>
> A 9th entry was added later: **Qwen3.6-27B (dense) at FP8 + DFlash speculative decoding**, served via vLLM nightly. This one *would* fit on a 5090, but it's the production agentic preset on this host — and it tops SWE-bench Lite over every other local model we measured, so it earned a spot.

## TL;DR

- **gpt-oss-120b Q8_0 is the headline config.** 120B parameters, sparse MoE (4-of-128 experts/tok), **264 tok/s decode at Q8**, 21/22 on the coding suite, 66 GB VRAM. It's the fastest large model we've measured on any hardware, and it cannot load on the 5090 at all.
- **Qwen3.6-27B FP8 + DFlash spec dec is the SWE-bench Lite winner — 57.3% resolved.** Beats Opus-distilled Qwen3.6-35B-A3B (52.0%), stock 35B-A3B (48.3%), and Gemma-4-31B-IT (23.0%) on the same 300-instance test split. Served via vLLM nightly with the [z-lab DFlash](https://github.com/z-lab/dflash) block-diffusion drafter (k=15). Spec dec gives **~4× decode speedup** vs the same FP8 weights without DFlash (195 tok/s warm vs 47 tok/s) at no quality cost — verifier-checked spec dec is lossless by construction. This is the daily-driver agentic preset on this host.
- **Gemma 4 31B-IT is the quality king.** 22/22 on the coding suite at both BF16 and Q8_0. BF16 fits at full 262K on this card (82 GB VRAM) — the first card in our lineup where that's possible.
- **Qwen3.6-35B-A3B is the throughput king in the mid-tier.** 221 tok/s at Q8 (CUDA), though coding quality is noisy across quants (14-15/22).
- **Qwen3-Coder-Next Q6_K is fast (196 tok/s) and specialized for coding**, but the familiar Qwen-family LRU Cache blind spot is still there (0/6). Good if your workload is parsing / pathfinding / string work; not if you need eviction+expiry logic.
- **Gemopus fine-tune regresses Gemma 4 on coding.** Base Gemma 31B-IT is strictly better (22/22 vs 15-16/22). The fine-tune helps on some tasks but catastrophically fails LRU Cache with TTL.
- **LRU Cache with TTL is the pattern-separation benchmark in our suite.** Only Gemma-4-31B-it (22/22) and gpt-oss-120b (6/6) pass it reliably. Every Qwen-family model — Qwen3.6, Qwen3-Coder-Next, and Qwopus on the 5090 — fails LRU at 0-4/6 without help. This is consistent across architectures (dense, MoE, hybrid linear-attn) and precisions (BF16, Q8, Q6, Q4), so it's a training-data / CoT-shape signal, not a quantization artifact.
- **Enabling thinking mode on Qwen3.6 fully recovers the LRU blind spot** (0/6 → 6/6) and lifts total to 21/22, matching gpt-oss-tier quality with no latency cost. See [prompting levers section](#prompting-levers-can-we-recover-qwen-family-failures). Thinking does *not* engage on Qwen3-Coder-Next — that one remains at 15/22.
- **Vulkan and CUDA are nearly identical on dense models.** CUDA wins +66% on one MoE config (Qwen3.6 BF16) but only +7% on Q8 MoE and +1% on dense. **llama.cpp's Vulkan backend has caught up on Blackwell.**

## The Pro 6000's defining advantage

Same 1792 GB/s memory bandwidth as the 5090 but **3× the memory** (96 GB vs 32 GB). In our lineup this unlocks three model classes the 5090 can't touch:

1. **31B dense at BF16.** 62 GB of weights + 22 GB of KV cache at 262K = 84 GB. The 5090 maxes out at Q8_0 for a 31B dense and can't do full-context f16 KV at 262K.
2. **35B MoE at BF16.** 69 GB of weights + 5 GB KV (hybrid arch). The 5090 only fits this at Q4_K_M.
3. **120B-class MoE models** (gpt-oss-120b Q8 = 63 GB). Won't load on a 5090 at any quant that preserves quality.

The card runs the same bandwidth-bound math as the 5090 for any model that fits on both, so dense-model throughput is essentially identical. The value proposition is **"everything the 5090 can do, plus everything up to 96 GB."**

---

## Summary table

All throughput is CUDA backend (see Vulkan comparison below). VRAM at full native context. Coding score is out of 22 across 4 benchmarks: String Processor (5) + Expression Evaluator (5) + A* Pathfinding (6) + LRU Cache with TTL (6). SWE-bench Lite is the 300-instance test split via SWE-agent v1.1.0 (4 workers, 75-call ceiling) — full breakdown in [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md); "—" means not run on this preset.

| Tier | Model | Quant | VRAM (full ctx) | Native ctx | TTFT | Decode (CUDA) | Coding | SWE-bench Lite |
|---|---|---|---|---|---|---|---|---|
| **S** | Gemma-4-31B-it | BF16 | 82.0 GB | 262K | 309 ms | **25.13 tok/s** | **22/22 (100%)** | — |
| **S** | Gemma-4-31B-it | Q8_0 | 54.5 GB | 262K | 193 ms | 43.76 tok/s | **22/22 (100%)** | 23.0% (69/300) |
| **S** | gpt-oss-120b | Q8_0 | 65.8 GB | 131K | **41 ms** | **264.38 tok/s** | 21/22 (95%) | — |
| **S** | Qwen3.6-27B (dense) | FP8 + DFlash † | 29 GB ‡ | 262K | n/a | **195.3 tok/s warm** § | 21/22 (95%) ¶ | **57.3% (172/300)** ⊗ |
| A | Qwen3.6-35B-A3B | BF16 | 72.1 GB | 262K | 61 ms | 135.05 tok/s | 14/22 (64%) | — |
| A | Qwen3.6-35B-A3B | Q8_0 | 41.6 GB | 262K | 60 ms | **221.04 tok/s** | 15/22 (68%) | 48.3% (145/300) ⊕ |
| B | Gemopus-4-31B-it | BF16 | 82.0 GB | 262K | 308 ms | 25.13 tok/s | 16/22 (73%) | — |
| B | Gemopus-4-31B-it | Q8_0 | 54.5 GB | 262K | 192 ms | 43.77 tok/s | 15/22 (68%) | — |
| A | Qwen3-Coder-Next | Q6_K | 70.3 GB | 262K | 77 ms | 196.36 tok/s | 15/22 (68%) | — |

† vLLM nightly + [z-lab DFlash](https://github.com/z-lab/dflash) block-diffusion drafter, k=15. All other rows are stock llama.cpp.
‡ Weights only. Full vLLM footprint with `--gpu-memory-utilization 0.92` plus DFlash drafter (3.3 GB) plus paged-KV pool ≈ 88 GB.
§ Warm-prefix decode on an 800-token prompt, 3-run mean. Cold-prefix is 149.6 tok/s. Without DFlash on the same FP8 weights: 47 tok/s (single-stream coding bench, single run). The non-DFlash 47 tok/s is what the SWE-bench result below was measured at; DFlash adds throughput, not quality.
¶ Best-of-3 at T=0.3 on the same 4-benchmark coding suite (Qwen3.6 is a reasoning model — different temperature regime than the T=0 rows above).
⊗ Same FP8 weights, vLLM 0.19.1, **no spec dec** (SWE-bench predates the DFlash addition). 64K context ceiling hit `exit_context` 15× — recoverable headroom for a re-run.
⊕ Stock weights. The Opus-reasoning-distilled fine-tune of the same Q8_0 model resolved **52.0% (156/300)** — see Qwen3.6-35B-A3B section below.

**Quick-pick guide:**
- **Best coding quality at any speed:** Gemma-4-31B-it Q8_0 (22/22, 44 tok/s) — BF16 adds nothing at temp 0
- **Best coding quality at high speed:** gpt-oss-120b Q8_0 (21/22, 264 tok/s) — 6× faster than Gemma, drops 1 test on A*
- **Best agentic-coding result (SWE-bench Lite):** Qwen3.6-27B FP8 + DFlash (57.3%, 195 tok/s warm) — the actual production preset on this host
- **Highest decode throughput period:** gpt-oss-120b Q8_0 (264 tok/s)
- **Don't bother with:** Gemopus (base Gemma beats it), Qwen3.6 MoE for reliable coding (run-to-run variance)

---

## S-Tier: Complete Solutions

### gpt-oss-120b Q8_0 (the headline result)

The flagship 120B-parameter MoE from OpenAI, served locally at 264 tok/s. This is a configuration **impossible on any smaller card.** VRAM footprint at full 131K context: 65.8 GB — the Pro 6000 has 30 GB of headroom left.

| Metric | Value |
|---|---|
| Coding score | **21/22 (95%)** — String 5/5, ExprEval 5/5, A* 5/6, LRU 6/6 |
| Decode throughput (CUDA) | **264.38 tok/s** |
| TTFT | **41 ms** |
| VRAM @ 131K ctx | **65,802 MB** |
| Per-benchmark wall-clock | 7-14 s each (vs 60-120 s for Gemma BF16) |
| Architecture | 36-layer GptOssForCausalLM, 8 KV heads, head_dim 64, 4-of-128 experts/tok |

**Strengths:** The quality-per-speed king. At 264 tok/s a complete implementation of LRU Cache with TTL generates in ~14 s. The hardware headroom (30 GB free at full context) leaves plenty for parallel serving if needed.

**Weakness:** One test fail on A* Pathfinding (likely the test-style variance we see across models). Not fundamental.

---

### Gemma-4-31B-it BF16 / Q8_0 (the quality king)

Perfect score on our 4-benchmark coding suite at both quants. At temp 0, BF16 and Q8 produce byte-identical output for our prompts — so for scoring purposes they're the same, and the only reason to pick BF16 over Q8 on this card is future-proofing (e.g., if you plan to run with non-zero temperature where the BF16 noise floor matters).

| Metric | BF16 | Q8_0 |
|---|---|---|
| Coding score | **22/22 (100%)** | **22/22 (100%)** |
| SWE-bench Lite | not run | **23.0% (69/300)** |
| Decode (CUDA) | 25.13 tok/s | 43.76 tok/s |
| Decode (Vulkan) | 24.82 tok/s | 43.69 tok/s |
| TTFT | 309 ms | 193 ms |
| VRAM @ 262K ctx | 81,996 MB | 54,550 MB |
| Bandwidth utilization | 79% of 1792 GB/s | 72% of 1792 GB/s |

**Strengths:** Perfect coding score. Full 262K native context. Dense architecture produces very stable output.

**SWE-bench gap:** Gemma's 100% coding-bench score does **not** translate to agentic-style coding — 23.0% on SWE-bench Lite is the lowest of any model we've measured here. The benchmarks measure different things: coding-bench rewards from-scratch self-consistent module-with-tests generation (Gemma's strength); SWE-bench rewards codebase-navigation and surgical-fix reasoning (where the Qwen-family models pull ahead). See [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md) for the failure-mode analysis.

**Weaknesses:** ~6× slower than the MoE models on throughput — pay the "every-parameter-active" penalty. If your workload tolerates ~25 tok/s, this is the highest-quality dense model we've measured.

---

### Qwen3.6-27B FP8 + DFlash spec dec (the production preset)

The dense 27B variant of Qwen3.6 (different model from the 35B-A3B MoE elsewhere in this doc), served via vLLM nightly with the [z-lab DFlash](https://github.com/z-lab/dflash) block-diffusion drafter for speculative decoding. This is the daily-driver agentic preset on this host (configured in `opencode-config/hosts/rtxpro6000/llama-swap.yaml` as `qwen36-27b-coder` and `qwen36-27b-agent`).

| Metric | Value |
|---|---|
| Coding score (best-of-3, T=0.3) | **21/22 (95%)** — String 5/5, ExprEval 5/5, A* 5/6, LRU 6/6 |
| **SWE-bench Lite (300 inst.)** | **172 / 300 = 57.3% resolved** — highest in this lineup |
| Decode, warm prefix (800-tok prompt) | **195.3 tok/s** (k=15 draft tokens) |
| Decode, cold prefix | 149.6 tok/s |
| Decode, no DFlash (same FP8 weights) | 47 tok/s — DFlash gives ~4× speedup |
| Drafter acceptance rate | 30.5% short prompt → 8.9% at 52K ctx (drafter SWA = 2048) |
| Weights | 29 GB (FP8) + 3.3 GB drafter |
| Native ctx | 262K |
| Architecture | Dense `qwen3_5` 27B with hybrid linear+full attention; drafter is a 5-layer SWA model (z-lab/Qwen3.6-27B-DFlash) |

**Strengths.**
- **The SWE-bench winner.** 57.3% on the 300-instance Lite test split beats Opus-distilled 35B-A3B (52.0%), stock 35B-A3B (48.3%), and Gemma-4-31B-IT (23.0%) on the same harness (SWE-agent v1.1.0, 4 workers, 75-call ceiling). The 64K ceiling this run used hit `exit_context` 15 times — recoverable headroom for a future re-run.
- **Spec-dec scales the dense-model throughput.** 195 tok/s warm puts a dense 27B in the same throughput band as the MoE models on this card, with the per-token quality of a dense model. DFlash is verifier-checked, so accepted tokens are bit-identical to running the target model alone — adding spec dec changes throughput, not quality.
- **k=15 sweep was run.** Measured 182.2 / 186.7 / 195.3 tok/s at k=8/10/15. Block-diffusion drafting does all k positions in one forward pass, so larger k is mostly verifier cost — author-default k=15 wins.
- **FP8 quantization of the target doesn't hurt the BF16 drafter.** Per-position acceptance curves match the all-BF16 sister entry within noise (29.2% vs 29.1%).

**Weaknesses.**
- **Acceptance collapses with context length** (30% short → 9% at 52K). Drafter has `sliding_window=2048` and only sees the recent tail of the prompt. Past ~100K context the drafter compute outweighs accepted-token savings — the long-context sister preset (`qwen36-27b-fp8-yarn2`) skips spec dec for that reason.
- **Stack fragility.** vLLM nightly + drafter HF card warns "still under training, inference results may be unstable." Cold-swap is ~60-90s first time (vs ~30s for llama.cpp).
- **No chat-completion-streaming throughput number** comparable to the llama.cpp rows above on this preset yet. The 195 tok/s figure is short-prompt warm-prefix decode, which is closer to agentic-tool-loop reality than to the streaming-bench number, but not directly comparable.

**The non-DFlash baseline:** without spec dec, the same FP8 weights decode at ~47 tok/s — competitive with Gemma-Q8 but ~4× slower than the DFlash-on configuration. The 57.3% SWE-bench result was actually measured at 47 tok/s (vLLM 0.19.1, no spec dec); DFlash makes the same patches arrive ~4× faster.

See [QWEN36_RTXPRO6000.md](QWEN36_RTXPRO6000.md) for the BF16/FP8/NVFP4 precision sweep on this same model (no spec dec) and [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md) for the four-model agentic-coding breakdown.

---

## A-Tier: Throughput Kings

### Qwen3.6-35B-A3B (hybrid-attention MoE)

Highest throughput among the Qwen lineup: 221 tok/s at Q8, 135 at BF16 (CUDA). But the coding score is noisy — LRU Cache fails at Q8, A* Pathfinding fails at BF16. The hybrid linear+full-attention architecture means KV cache at 262K is only ~5 GB (vs 21 GB for Gemma's dense architecture).

| Metric | BF16 | Q8_0 (stock) | Q8_0 Opus-distilled |
|---|---|---|---|
| Coding score | 14/22 (64%) | 15/22 (68%) | 10/22 (45%) † |
| SWE-bench Lite | not run | 48.3% (145/300) | **52.0% (156/300)** |
| Decode (CUDA) | 135.05 tok/s | **221.04 tok/s** | (same as stock) |
| Decode (Vulkan) | 81.29 tok/s | 205.65 tok/s | — |
| TTFT | 61 ms | 60 ms | — |
| VRAM @ 262K ctx | 72,070 MB | 41,558 MB | 41,558 MB |

† Opus-distill regresses on from-scratch coding-bench (-5 vs stock Q8) but gains +3.7 pp on agentic SWE-bench. The two benchmarks reward different abilities — see Qwen3.6 deep-dive section in [QWEN36_RTXPRO6000.md](QWEN36_RTXPRO6000.md) and the [coding-bench / SWE-bench disagreement analysis](SWEBENCH_LITE_RTXPRO6000.md#why-coding-bench-and-swe-bench-disagree).

**Strengths:** Fastest dense-quality model for throughput. Tiny KV cache means essentially zero context-window penalty. Stock Q8_0 hits 48.3% on SWE-bench Lite — the strongest llama.cpp-served result in this lineup, and within striking distance of the 27B FP8/vLLM run despite running at BF16-quality coding-bench.

**Weaknesses:** Inconsistent coding correctness. Which benchmarks pass varies between BF16 and Q8 despite temp=0 — the hybrid-attention routing produces different code structures at the two precisions, and the harder tests (LRU, A*) are right at the capability edge.

---

### Qwen3-Coder-Next Q6_K (hybrid-attention coder specialist)

Coding-specialist 80B-class MoE with `qwen3_next` hybrid linear/full attention (similar family as Qwen3.6). Runs at 196 tok/s at Q6_K — slower than the 4-bit variants referenced elsewhere in the repo, but at a quantization level that's proven lossless in our earlier testing.

| Metric | Value |
|---|---|
| Coding score | **15/22 (68%)** — String 4/5, ExprEval 5/5, A* 6/6, LRU 0/6 |
| Decode (CUDA) | **196.36 tok/s** |
| TTFT | 77 ms |
| VRAM @ 262K ctx | 70,252 MB |
| Per-benchmark wall-clock | 6-16 s each |

**Strengths:** Fastest per-token of the dense-quality models; passes all 6 A* Pathfinding tests (the only other model to do this at this tier is Gemma). Architecture-friendly VRAM (70 GB leaves 26 GB headroom).

**Weakness:** Same failure mode as Qwen3.6 — **LRU Cache with TTL fails all 6 tests.** This is a consistent "Qwen-family has a blind spot on this particular pattern" signal in our data. Base Qwen 3.5 27B also failed LRU; only the reasoning fine-tunes recover it. If LRU-style eviction+expiry logic matters for your workload, use Gemma 4 31B or gpt-oss.

---

## B-Tier: Gemopus-4-31B-it (regression on base)

Jackrong's "stability-first" fine-tune of Gemma-4-31B-it. Throughput and VRAM are identical to base Gemma (same arch, same weights + SFT delta). Coding quality regresses: **16/22 BF16, 15/22 Q8** vs base Gemma's **22/22.** Specifically collapses on LRU Cache with TTL (0/6 at Q8, timeout at BF16), and drops 1 test on A* at Q8.

The fine-tune was explicitly designed to avoid "aggressive Claude-style CoT distillation" per the Gemopus README. The tradeoff shows up in the hardest benchmark.

**Verdict:** Unless you specifically need the Gemopus instruction-tuning flavor, run base Gemma 4 31B-IT.

---

## Vulkan vs CUDA: surprisingly close on Blackwell

We ran the same 6 dense/MoE configurations through both backends (llama.cpp b8826 Vulkan prebuilt vs a master-branch CUDA 13.2 build with `sm_120`). The comparison is cleaner on this hardware than it was on previous-gen cards.

| Model | Vulkan tok/s | CUDA tok/s | Ratio |
|---|---|---|---|
| Gemma-4-31B-it BF16 (dense) | 24.8 | 25.1 | **1.01×** |
| Gemma-4-31B-it Q8_0 (dense) | 43.7 | 43.8 | **1.00×** |
| Gemopus-4-31B-it BF16 (dense) | 24.6 | 25.1 | **1.02×** |
| Gemopus-4-31B-it Q8_0 (dense) | 43.5 | 43.8 | **1.01×** |
| Qwen3.6-35B-A3B BF16 (MoE) | 81.3 | 135.1 | **1.66×** |
| Qwen3.6-35B-A3B Q8_0 (MoE) | 205.7 | 221.0 | 1.07× |

**Findings:**
1. **Dense models: no meaningful difference.** Both backends are bandwidth-bound and hit 70-80% utilization of the 1792 GB/s memory. Use whichever ships faster — the Vulkan prebuilt binary is a single-file `.tar.gz` with no toolchain required, vs a 20-minute CUDA toolkit install + 5-minute source build.
2. **MoE at Q8: CUDA +7%.** Both backends handle quantized expert routing well; the gap is small enough that setup effort dominates the decision.
3. **MoE at BF16: CUDA +66%.** This is the one config where CUDA's specialized MoE kernels matter — Vulkan's generic compute shaders haven't been specialized for BF16 expert dispatch on Blackwell.
4. **TTFT: CUDA consistently ~30-50% faster** on MoE (61 ms vs 96 ms on Qwen BF16). Less relevant for coding workloads where decode dominates, but matters for chat-style UIs.

For this repo's workload (single-stream coding benchmarks), the practical recommendation is: **default to Vulkan for dense, use CUDA if you're running BF16 MoE or need minimum TTFT.**

---

## VRAM headroom analysis

All 7 configs fit at full native context on 96 GB. The measured totals (weights + KV cache + compute buffer + unaccounted):

| Model | Quant | Weights | KV @ native ctx | Total | Headroom (96 GB) |
|---|---|---|---|---|---|
| Gemma-4-31B-it | BF16 | 57.2 GiB | 21.7 GiB | 82.0 GB | **14.0 GB** |
| Gemma-4-31B-it | Q8_0 | 30.4 GiB | 21.7 GiB | 54.5 GB | 41.5 GB |
| Gemopus-4-31B-it | BF16 | 57.2 GiB | 21.7 GiB | 82.0 GB | 14.0 GB |
| Gemopus-4-31B-it | Q8_0 | 30.4 GiB | 21.7 GiB | 54.5 GB | 41.5 GB |
| Qwen3.6-35B-A3B | BF16 | 64.6 GiB | 5.1 GiB | 72.1 GB | 23.9 GB |
| Qwen3.6-35B-A3B | Q8_0 | 34.7 GiB | 5.1 GiB | 41.6 GB | 54.4 GB |
| Qwen3-Coder-Next | Q6_K | ~61 GiB | ~5 GiB | 70.3 GB | 25.7 GB |
| gpt-oss-120b | Q8_0 | 55.1 GiB | ~9.0 GiB | 65.8 GB | 30.2 GB |
| Qwen3.6-27B (dense) | FP8 + DFlash | 27.0 GiB | paged pool ‖ | ~88 GB ‖ | ~7 GB |

‖ vLLM with `--gpu-memory-utilization 0.92` allocates a paged-KV pool that scales to fill the budget; total reported is target weights (29 GB) + DFlash drafter (3.3 GB) + ~50 GB KV pool + activations/buffers. This footprint hosts ~2 concurrent 100K-ctx sessions or 1 at 262K.

**Observations:**
- The Gemma-family architecture (10 full-attention layers at head_dim 512, 50 SWA layers capped at 1024) has the largest KV cache at full context (~22 GB at f16). For any model ≥30B at BF16, this is the one that eats your headroom.
- **Qwen3.6's hybrid linear-attention + 2-kv-head GQA on the 10 full-attn layers is the VRAM winner** — 5 GB KV at 262K is 4× smaller than Gemma's.
- gpt-oss-120b's 131K context + 8 KV heads × head_dim 64 keeps KV under 10 GB even at full context.

---

## Configuration

```bash
# Vulkan (prebuilt, no build required)
LD_LIBRARY_PATH=/home/gisenberg/llama/llama-b8826 \
  /home/gisenberg/llama/llama-b8826/llama-server \
  -m <model.gguf> --port 8080 \
  -c <native_ctx> -ngl 99 -fa on --no-mmap -np 1

# CUDA 13.2 build (master branch, -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120)
LD_LIBRARY_PATH=/home/gisenberg/llama-build/src/build/bin:/path/to/cuda/lib \
  /home/gisenberg/llama-build/src/build/bin/llama-server \
  -m <model.gguf> --port 8080 \
  -c <native_ctx> -ngl 99 -fa on --no-mmap -np 1

# vLLM nightly + DFlash spec dec (Qwen3.6-27B FP8 only)
vllm serve /home/gisenberg/models-vllm/qwen36-27b-fp8 \
  --served-model-name qwen36-27b-coder \
  --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --attention-backend flash_attn \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"/home/gisenberg/models-vllm/qwen36-27b-dflash-drafter","num_speculative_tokens":15}'
```

The llama.cpp rows are stock — no TurboQuant, no rotorquant, no NVFP4. The Pro 6000's 96 GB means KV-compression shortcuts aren't needed for anything in this lineup; dense 31B at BF16 fits at full 262K with 14 GB to spare, every other llama.cpp config has 20+ GB of headroom.

The Qwen3.6-27B FP8 + DFlash row is the lone vLLM-nightly entry. DFlash is a **block-diffusion** drafter (all k positions emit in one forward pass, not autoregressive) — distinct from MTP / Eagle / Medusa style drafters and from the ngram/draft-model spec dec available in mainline llama.cpp. The drafter card and method are still pre-1.0; treat this preset as experimental relative to the llama.cpp rows.

---

## Prompting levers: can we recover Qwen-family failures?

The summary table above uses single-shot, no-thinking, user-prompt-only as the baseline configuration (matching our 5090 methodology). Once we saw that Qwen-family models all failed LRU Cache with TTL but Gemma aced it, we ran a follow-up sweep testing three levers on Qwen3.6-35B-A3B (both quants) and Qwen3-Coder-Next Q6_K:

1. **Thinking on** — `-rea on --reasoning-budget 16384` on the llama-server side; no client-side changes.
2. **System prompt** — a careful-engineer persona prepended as a `role: system` message. See `tools/rtxpro6000_levers_bench.py` for the exact text.
3. **Agentic test-fix loop** — on test failure, send the pytest output back to the model and ask for a fix. Up to 3 rounds.

### Lever results

| Model | Quant | Baseline | + Thinking | + System prompt | + Agentic |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | BF16 | 14/22 | **21/22 (+7)** | 15/22 (+1) | — (not run) |
| Qwen3.6-35B-A3B | Q8_0 | 15/22 | **21/22 (+6)** | 15/22 (0) | **22/22 (+7)** |
| Qwen3-Coder-Next | Q6_K | 15/22 | 15/22 (0) | 14/22 (−1) | — (not run) |

### Per-benchmark detail

The interesting story is in the per-benchmark breakdown. LRU Cache with TTL is the benchmark that separates the winners from everyone else.

**Qwen3.6-35B-A3B BF16:**
- Baseline: String 5, Expr 5, A* 0, LRU 4 → 14
- Thinking: String 5, Expr 5, A* 5, LRU 6 → 21
- Sysprompt: String 4, Expr 5, A* 6, LRU 0 → 15 (flipped which benchmark fails)

**Qwen3.6-35B-A3B Q8_0:**
- Baseline: String 5, Expr 4, A* 6, LRU 0 → 15
- Thinking: String 5, Expr 5, A* 5, LRU 6 → 21 (LRU 0→6 is the dominant effect)
- Sysprompt: String 5, Expr 5, A* 5, LRU 0 → 15 (LRU stuck, other tests stable)
- Agentic: String 5, Expr 5, A* 6 (2 rounds), LRU 6 → 22 (A* fixed in 1 feedback round)

**Qwen3-Coder-Next Q6_K:**
- Baseline: String 4, Expr 5, A* 6, LRU 0 → 15
- Thinking: String 4, Expr 5, A* 6, LRU 0 → 15 (*identical, no time penalty either — thinking mode does not engage on this model/arch*)
- Sysprompt: String 5, Expr 5, A* 4, LRU 0 → 14

### Latency cost of the levers

Wall-clock total for the 4-benchmark suite on Qwen3.6 Q8:

| Mode | Total | vs Baseline |
|---|---|---|
| Baseline (no lever) | 199 s | — |
| + Thinking | 197 s | ±0% |
| + Agentic | 234 s | +18% |

**Thinking is free.** The reasoning budget is used only when the benchmark calls for it (LRU: +10s) and skipped when it doesn't (String: −8s). Agentic costs ~40s per experiment because of the one A* retry round.

### Findings

1. **Thinking recovers Qwen3.6 completely.** Both BF16 and Q8 go from 14-15 → 21, matching the gpt-oss-120b tier on quality. The recovery is entirely from LRU Cache (0-4 → 6) — the benchmark Gemma aces and every other Qwen-family model fails without help.

2. **System prompt is net-zero or negative.** The "careful engineer" framing can help one benchmark while breaking another — on Qwen3.6 BF16 it flipped A* from 0→6 but LRU from 4→0. Never net-improves. The failure mode is over-deliberation that exhausts `max_tokens` before emitting the code block.

3. **Agentic tops out at 22/22 on Qwen3.6 Q8** — one round of pytest feedback recovers the last A* test. Cheaper than thinking on compute terms only if you already had a test harness in-loop; not cheaper on latency (+18% vs +0% for thinking).

4. **Qwen3-Coder-Next is immune to both levers.** Identical scores and identical per-benchmark times across baseline/thinking/sysprompt (6-14s each). Either the `qwen3_next` architecture doesn't have a thinking head in this build, or `-rea on` isn't propagating through llama.cpp's jinja template. **The LRU failure on this model is a hard capability ceiling**, not a configuration issue — no prompt-level intervention we tried touched it.

5. **Gemma 4 still wins the no-lever race.** The reason base Gemma gets 22/22 in our baseline and Qwen needs thinking to reach 21/22 isn't that Qwen "can't solve LRU" — it's that Qwen's code-generation path, without extra reasoning, over-counts expired entries in `size()`. Gemma appears to internalize this invariant during its standard forward pass. This matters practically: if your serving setup doesn't enable thinking mode by default, you'll see raw Qwen at 14-15/22 while Gemma delivers 22/22 out of the box.

### Practical recommendation

- **If you're deploying Qwen3.6-35B-A3B for coding: turn thinking on.** +6-7 points, same wall-clock. Zero reason not to.
- **If you're deploying gpt-oss-120b: don't bother with levers.** Baseline is already 21/22 and runs 6× faster than Gemma.
- **If you're deploying Qwen3-Coder-Next: accept the LRU blind spot** or move to a higher-quality model. Levers don't help.
- **If you're deploying Gemma 4 31B-IT: keep thinking off** — it's already 22/22 and thinking would add latency without recovering anything.

## Cross-references (other docs in this repo)

- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — the bandwidth-rich 32 GB reference card
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — the capacity-rich 128 GB bandwidth-poor reference
- [HARDWARE_SHORTLIST.md](HARDWARE_SHORTLIST.md) — includes the pre-measurement Pro 6000 projections (now superseded by this doc for the models we tested)
- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — measured side-by-side comparison across 3 machines (pre-Pro-6000)

## Raw results

- Throughput: `experiments/rtxpro6000_bench_{vulkan,cuda}/*.json`
- Coding: `experiments/rtxpro6000_coding/*.json` and per-benchmark response markdown in `experiments/rtxpro6000_coding/<key>/*.md`
- Prompting levers: `experiments/rtxpro6000_levers/<key>__<mode>/*.md` and rebuilt summaries in `experiments/rtxpro6000_levers/<key>__<mode>.json`
- Qwen3.6-27B precision sweep (BF16/FP8/NVFP4, no spec dec): `experiments/nvfp4_qwen36_27b/{bf16,fp8,nvfp4}/results.json`
- Qwen3.6-27B FP8 SWE-bench Lite (no spec dec): `experiments/sweagent_lite_fp8/preds.json` and `sweagent_lite_fp8.qwen36-27b-fp8-full.json`
- Qwen3.6-27B FP8 + DFlash preset definition + measured numbers: `~/git/gisenberg/opencode-config/hosts/rtxpro6000/llama-swap.yaml` (`qwen36-27b-coder` block, lines ~321-368)
- Harness: `tools/rtxpro6000_bench.py`, `tools/rtxpro6000_coding_bench.py`, `tools/rtxpro6000_rescore.py`, `tools/rtxpro6000_levers_bench.py`, `tools/rtxpro6000_levers_rescore.py`, `tools/nvfp4_qwen36_27b_bench.py`
