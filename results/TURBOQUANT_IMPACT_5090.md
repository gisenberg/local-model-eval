# TurboQuant Impact: What It Unlocks on RTX 5090 (32 GB)

For the M4 Max story (where TurboQuant unlocks one model and hurts speed everywhere else), see [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md).

## Models That Require TurboQuant for Practical Use

Some models simply can't function at useful context lengths without KV cache compression. TurboQuant transforms them from "technically loads but unusable" to "full context, full speed."

| Model | Weights | KV/Token (f16) | Max Ctx (f16 KV) | Max Ctx (turbo4 KV) | Gained | Score |
|---|---|---|---|---|---|---|
| **gemma-4-31b-it Q4_K_M** | 18 GB | ~870 KB | **~16K** | **~58K** | **+42K** | 17/17 (S) |
| **gemma-4-26b-a4b Q6_K** | 22 GB | ~120 KB | **~85K** | **~230K** | **+145K** | 17/17 (S) |
| **gemma-4-26b-a4b Q4_K_M** | 16 GB | ~120 KB | **~136K** | **262K (full)** | **+126K** | 16/17 (A) |
| qwopus-3.5-27b-v3 Q6_K | 22 GB | ~64 KB | **~160K** | **262K (full)** | **+102K** | 16/17 (A) |
| qwen3.5-27b-opus Q4_K_M | 16 GB | ~64 KB | **~256K** | **262K (full)** | ~6K | 17/17 (S) |
| qwen3.5-35b-a3b Q4_K_M | 20 GB | ~20 KB | **~262K** | **262K (full)** | 0 | 11/17 (C) |

### How to Read This Table

**Max Ctx (f16 KV)** = how much context fits in 32 GB VRAM with standard f16 KV cache (what LM Studio uses). Calculated as: `(32 GB - weights - 1.5 GB overhead) / KV_per_token`.

**Max Ctx (turbo4 KV)** = same calculation but with turbo4's 3.8x compression on KV cache.

**Gained** = additional context tokens unlocked by TurboQuant.

## The Three Categories

### Must-Have: Models That Don't Work Without TurboQuant

**Gemma 4 31B-IT** is the extreme case. Its dense architecture burns **870 KB per token** on KV cache — 7x more than the MoE variant. With f16 KV, after loading 18 GB of weights, only ~12 GB remains for KV cache. That's **~16K tokens of context** — barely enough for a single coding prompt + response, with no room for multi-turn conversation or long documents.

With turbo4, that same 12 GB budget stretches to **~58K tokens** — enough for real work. And it scored **17/17 (100%)** on our coding benchmarks.

Without TurboQuant, this model is a paperweight on 32 GB. With it, it's S-tier.

### Game-Changer: Models That Go From Limited to Full Context

**Gemma 4 26B-A4B** (both quants) has a sliding window + global attention architecture with ~120 KB/token KV cache. The Q6_K variant with f16 KV maxes out at ~85K context — and our earlier tests showed it drops from 150 to 55 tok/s at 256K due to RAM spill.

With turbo4 KV:
- Q6_K reaches ~230K context entirely in VRAM, no spill, no throughput cliff
- Q4_K_M reaches the full 262K native context window
- Both score 16-17/17 on coding benchmarks

**Qwopus 27B Q6_K** goes from ~160K to full 262K. Not as dramatic, but the difference between "most of the context window" and "all of it" matters for long-document tasks.

### Minimal Impact: Models That Already Fit

**Qwen 3.5 35B-A3B** has a DeltaNet hybrid architecture with only 10 attention layers and tiny KV cache (~20 KB/token). Even with f16 KV, it fits 262K context in 32 GB VRAM. TurboQuant saves negligible VRAM for this model. (It also scores poorly on llama-server regardless of KV config, so the benefit is near zero.)

## Throughput Impact

TurboQuant isn't free — the dequantization overhead costs ~8-10% throughput at 32K context:

| Model | f16 KV (tok/s) | turbo4 KV (tok/s) | Delta |
|---|---|---|---|
| Gemma 26B Q6_K | 152* | 142 | -7% |
| Gemma 26B Q4_K_M | 164* | 156 | -5% |
| Gemma 31B Q4_K_M | N/A** | 53 | N/A |
| Qwopus 27B Q6_K | 52* | 52 | ~0% |

*llama-server with f16 KV at 32K context
**Gemma 31B can't run at useful context with f16 KV, so no baseline exists

At short context (32K), this is a modest cost. At long context (128K+), TurboQuant is actually *faster* than f16 because f16 would spill to system RAM while turbo4 stays in VRAM.

## The Bottom Line

| Without TurboQuant | With TurboQuant |
|---|---|
| Gemma 31B: 16K context, unusable | Gemma 31B: 58K context, S-tier (17/17) |
| Gemma 26B Q6_K: 85K context, cliff at 256K | Gemma 26B Q6_K: 230K context, S-tier (17/17) |
| Gemma 26B Q4_K_M: 136K, cliff at 256K | Gemma 26B Q4_K_M: 262K full, A-tier (16/17) |
| 3 models viable for long context | 5+ models viable for long context |

TurboQuant's value scales with KV cache cost. Models with cheap KV (DeltaNet hybrids like Qwen 35B) don't need it. Models with expensive KV (dense transformers like Gemma 31B, or global-attention models like Gemma 26B) are transformed by it.

## Model-Size Effect on Compression Quality

TurboQuant's quality cost also scales with model size — but inversely. Smaller models pay a larger quality penalty for the same KV compression ratio.

| Model | turbo4 vs f16 KV | Delta |
|---|---|---|
| Gemma 4 31B-IT (31B dense) | 17/17 → 17/17 | **No degradation** |
| Gemma 4 26B-A4B (4B active MoE) | 17/17 → 17/17 | **No degradation** |
| Qwen3-8B (8B dense) | 12/17 → 8/17 | **-4 tests (-24%)** |
| Gemma 4 E4B (2.3B active) | 5/22 → 5/22 | No effect (model already at floor) |

**The pattern:** Models with sufficient capacity (~20B+ active params or large total) absorb the quantization noise without quality loss. 8B models lose meaningful quality. Below ~5B, the model is already too small for our benchmark suite to differentiate.

**Practical implication:** Use turbo4 freely on 26B+ models. On 8B models, prefer **TriAttention** (token eviction) instead — it preserves more quality at this scale by keeping a smaller number of tokens at full precision rather than degrading all tokens.

## Comparison to NVFP4-turbo (NVIDIA Blackwell)

NVIDIA's NVFP4 format uses Blackwell tensor cores via CUTLASS kernels. We tested LilaRest's NVFP4-turbo Gemma 31B against our llama.cpp pipeline:

| | llama.cpp Q4_K_M + turbo4 | vLLM NVFP4-turbo |
|---|---|---|
| Score (3 benchmarks) | **17/17 (100%)** | 16/17 (94%) |
| Throughput (single-stream) | **53 tok/s** | 41 tok/s |
| VRAM | **22.3 GB** | 29.6 GB |
| Max context | **~58K** | ~16K |
| Setup difficulty | Medium | High (CUDA 13 toolkit, cu130 wheel, JIT compile) |

For our single-stream coding workload, **llama.cpp + turbo4 wins on every metric we measure**. NVFP4-turbo's strength is concurrent batched throughput (1,244 tok/s with multiple concurrent users in published benchmarks) — relevant for production serving but not for our benchmarks.
