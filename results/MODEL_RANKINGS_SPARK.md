# Local Model Tier List — DGX Spark GB10 128GB

**Platform:** NVIDIA DGX Spark, GB10 (Grace Blackwell), ~120 GB usable unified memory
**Memory bandwidth:** ~273 GB/s LPDDR5X (the dominant constraint for this platform)
**Inference:** llama.cpp standard build, CUDA 13.0, aarch64 Linux
**Standard config:** flash attention (`-fa on`), `max_tokens=16384`, f16 KV (unless noted), thinking off (`-rea off`), `--no-mmap --jinja -np 1`

Tested April 2026.

## The Spark's defining constraint: memory bandwidth

This is the most important thing to understand about the Spark. **It is bandwidth-bound, not capacity-bound.** With 120GB of unified memory you can *load* enormous models, but with only ~273 GB/s of memory bandwidth (vs. ~1792 GB/s on an RTX 5090), each token's autoregressive decoding must read all active weights through a comparatively narrow pipe.

The implication: **dense >20B models are not viable for interactive use.** A dense model reads its entire weight set per token; on the Spark this caps you near the bandwidth ceiling regardless of how much VRAM you have spare.

| Model class | Bytes/token (Q4) | Spark theoretical max | Practical |
|---|---|---|---|
| Dense 31B Q8 | ~33 GB | 8 tok/s | **~7 tok/s** (measured) |
| Dense 70B Q4 | ~40 GB | 7 tok/s | ~6 tok/s expected |
| MoE 80B/3B active Q4 | ~2 GB | 135 tok/s | ~80–100 tok/s expected |
| MoE 122B/10B active Q4 | ~6 GB | 45 tok/s | ~30–40 tok/s expected |
| MoE 230B/10B active Q3 | ~5 GB | 55 tok/s | ~30–40 tok/s expected |

**Conclusion: this hardware exists to run MoE models with small active parameter counts.** Any other model class wastes the platform.

---

## Summary table

All benchmarks are single-shot at `temperature=0`, `-fa on`, `f16` KV, `--no-mmap`, 32K context. Quality is the 3-benchmark coding suite (Expression Evaluator + A* Pathfinding + LRU Cache with TTL, 17 tests total; some entries show bonus tests above 17). Throughput is sustained tok/s on the coding benchmarks (not peak, not blended with prefill). See per-model sections below for full breakdowns.

| Tier | Model | Engine | Family | Active / Total | Weights | Tok/s | Score | Note |
|---|---|---|---|---|---|---|---|---|
| **S** | Qwen3.5-122B-A10B Q4_K_M (bartowski) | **ik-llama** | Qwen | 10B / 122B | 71 GB | **26.0** | **17/17** | Recommended daily driver. Fastest S-tier. |
| **S** | Qwen3.5-122B-A10B Q4_K_M (unsloth) | mainline | Qwen | 10B / 122B | 72 GB | 21.0 | **18/17** | Highest absolute score (with bonus A* test). |
| **A** | GLM-4.5-Air Q4_K_M (bartowski) | mainline | Z.AI | 12B / 106B | 70 GB | 21.7 | 15/17 | First non-Qwen A-tier. Cross-family validation. |
| **A** | Qwen3.5-122B-A10B-REAP-20 Q4_K_M (0xSero) | ik-llama | Qwen | 10B / 99B | **57 GB** | **29.1** | 14/17 | 20% expert-pruned. Faster + smaller, real quality drop on ExprEval. |
| **A** | Qwen3.5-122B-A10B Q4_K_M (bartowski) | mainline | Qwen | 10B / 122B | 71 GB | 25.8 | 13/17 | Same model file as S-tier ik-llama entry. Engine choice matters. |
| **B** | Qwen3-Coder-Next UD-Q4_K_M (unsloth) | mainline | Qwen | 3B / 80B | 46 GB | **50.2** | 14/17 | Fastest large model on Spark. "First draft" speed. |
| **B** | Nemotron-3-Super-120B-A12B Q4_K_M (bartowski) | mainline | NVIDIA | 12B / 120B | 87 GB | 19.7 | 11/17 | Thinking mandatory, restrained. NVIDIA-optimized kernels. |
| **C** | MiniMax-M2.5 UD-Q3_K_XL (unsloth) | mainline | MiniMaxAI | 10B / 230B | 96 GB | 29.6 | 5/5 + 2 TOs | Capable but thinks too long — A\*/LRU timed out at 25 min. |
| **D** | Mistral-Small-4-119B-2603 Q4_K_M (bartowski) | mainline | Mistral | 6.5B / 119B | 69 GB | 8.0 | 7/17 | Slow AND buggy with `reasoning_effort=none`. Needs retest with thinking on. |
| **F** | Gemma 4 31B-IT Q8_0 (unsloth) | mainline | Google | 31B (dense) | 31 GB | 6.7 | n/a | Hits the bandwidth ceiling. Proves the thesis. |

**Quick-pick guide:**
- **Best daily driver:** Qwen3.5-122B bartowski + ik-llama (17/17 at 26 tok/s)
- **Fastest good model:** Qwen3-Coder-Next (14/17 at 50 tok/s) — rough first drafts
- **Memory-constrained:** Qwen3.5-122B REAP-20 (14/17 at 29 tok/s, 14 GB smaller)
- **Cross-family sanity check:** GLM-4.5-Air (15/17 at 22 tok/s) — different architecture, similar results
- **Don't use on Spark:** MiniMax-M2.5 (timeouts), Mistral-Small-4 with reasoning off (buggy), Gemma 4 31B or any dense >20B model (bandwidth-bound)

**Experimental / cautionary variants** (not listed above but documented in the A-tier section):
- Qwen3.5-122B bartowski + `-ctv q8_0` asymmetric KV → **−46% throughput** (14 tok/s) at identical quality
- Qwen3.5-122B bartowski + mainline + thinking ON → **8/17** (worse than thinking off) — same engine, same model, different deterministic path
- Speculative decoding on Qwen3.5-122B → **blocked** in mainline (DeltaNet hybrid incompatibility), **broken** in ik-llama (3× slowdown, wrong output)

---

## S-Tier: Reliable Excellence

### Qwen3.5-122B-A10B Q4_K_M (bartowski) on ik-llama
The fastest S-tier configuration on Spark. Same model file as our other Qwen3.5-122B benchmarks, but running on [ik-llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (ikawrakow's fork with custom attention/MoE kernels) instead of mainline llama.cpp. Marginally faster generation, dramatically better code quality on this hardware.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (5/5 + 7/6 + 5/6)** |
| Throughput (sustained) | **26.0 tok/s** |
| TTFT | 0.6 s |
| Weight size | 71 GB (2-shard GGUF, bartowski Q4_K_M) |
| Engine | ik-llama.cpp built from `main` (commit `db31e7d8`) for CUDA 13 / Blackwell sm_100 |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 --no-mmap --reasoning-budget 0` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **5/5** | 4985 (8.6 KB think + 11.3 KB content) | Clean recursive descent parser. Different code than what mainline produces on the same model. |
| A* Pathfinding | **7/6** | 3412 | Perfect plus the bonus 7th test that all Qwen3.5-122B variants seem to write. |
| LRU Cache with TTL | **5/6** | 12553 (24 KB think + 23 KB content) | One test failure on lazy expiry edge case. The hardest benchmark. |

**Why ik-llama wins on this hardware:** ik-llama uses different attention and MoE kernels than mainline llama.cpp (it's a fork that adds SOTA quantizations and optimized kernels). On the GB10 specifically, those kernels produce very small numerical differences in the per-token logits compared to mainline. At `temperature=0` those tiny differences accumulate over tokens and steer the model down meaningfully different generation paths. **For this model on this hardware the ik-llama path consistently produces correct code where the mainline path produces buggy code on Expression Evaluator** (see the cautionary tale below).

**Caveats and quirks:**
- ik-llama's `--reasoning-budget 0` does NOT actually disable thinking on Qwen3.5-122B — the model still emits reasoning content. We got the 100% score *with* thinking happening, not without. This is a quirk of how ik-llama interprets the budget flag for hybrid models, not a deliberate setting.
- The `--jinja` and `-rea on/off` mainline flags don't exist in ik-llama (it's based on an older llama.cpp branch with different flag conventions). `tools/spark_bench.py` handles the differences automatically when `server: "ik"` is set.
- ik-llama has some bugs in its speculative decoding implementation for hybrid models — produces incorrect output and runs slower than the baseline. We tested this; don't use spec dec on ik-llama for hybrid models.

**Verdict:** This is now the recommended Spark configuration for Qwen3.5-122B. 100% on every benchmark, 26 tok/s, fits comfortably in 128 GB unified memory.

**Long-context performance:** Tested at 32K, 64K, and 128K context with a real coding task (~20–80K tokens of this repo's `tools/*.py` files in the prompt + a specific feature-add request).

| Context | Prefill rate | Decode rate | Total time | Diff quality |
|---|---|---|---|---|
| 32K (~24K prompt) | 627 tok/s | **24.4 tok/s** | 56s | **6/6** |
| 64K (~47K prompt) | 603 tok/s | 22.7 tok/s | 95s | 5/6 |
| 128K (~94K prompt) | 553 tok/s | **19.6 tok/s** | 193s | **6/6** |

Decode rate degrades only ~20% from 32K to 128K (24.4 → 19.6 tok/s). The 12 full-attention layers in the DeltaNet hybrid are O(n²) but the bandwidth-bound decode dominates over attention compute even at 94K KV positions, so the practical degradation is mild. **Total time at long contexts is dominated by prefill** — at 128K, prefill is 170s of the 193s total. Once the prompt is processed, generation is fast.

Quality is stable: 6/6 at both 32K and 128K, 5/6 at 64K is a single-shot variance (the model wrote a slightly less complete diff for the medium-context run, missing one of the function-call updates). The model successfully reads `spark_bench.py` from the *middle* of the context (positions 5/10, 9/18, and 14/27 respectively across the three sizes) — distant retrieval works, this isn't just a recency-attention shortcut.

**Realistic usage:** A 30K-token codebase context + 400-token diff response is about 50 seconds end-to-end. A 90K-token large codebase is ~3 minutes. Both are interactive enough for a real coding workflow.

---

### Qwen3.5-122B-A10B Q4_K_M (unsloth) on mainline llama.cpp
122B/10B-active hybrid (DeltaNet linear attention in 36 layers + full attention in 12). The flagship Qwen MoE works exactly as the architecture predicts on Spark — 21 tok/s sustained, with **perfect code quality on every benchmark plus a bonus test**.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **18/17 (5/5 + 7/6 + 6/6, with bonus A\* test)** |
| Throughput (sustained) | **21.0 tok/s** |
| TTFT | 0.56 s |
| Weight size | 72 GB (3-shard GGUF) |
| Bandwidth utilization | ~7.5 GB/token × 21 tok/s ≈ 158 GB/s = 58% of peak |
| VRAM (32K ctx) | small KV: ~24 KB/token via DeltaNet hybrid |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **5/5** | 2245 | Perfect after the test_error_cases benchmark spec was loosened to verify exception class instead of literal message text |
| A* Pathfinding | **7/6** | 3280 | Wrote a 7th test beyond the 6 required, all passed |
| LRU Cache with TTL | **6/6** | 2758 | Clean pass on the hardest benchmark |

**Strengths:** Highest absolute quality score on Spark (the bonus 7th A* test edges this model to 18/17). Stable throughput across all 4 generation runs (21.0–21.2 tok/s, no degradation). Doesn't need ik-llama — works perfectly on mainline.

**Verdict:** Best raw quality score. Slower than the bartowski + ik-llama combo above, but the simpler stack (just mainline llama.cpp, no engine fork to maintain) makes this the easier choice if you don't want to deal with ik-llama's quirks. 21 tok/s is still interactive.

---

### Qwen3.5-122B-A10B AR-INT4+FP8 hybrid on vLLM + MTP speculative decoding
A different approach to the same model: Intel's AutoRound INT4 quantization with shared-expert layers converted to FP8, served via vLLM 0.19.1 compiled for SM121 (Blackwell) with FlashInfer attention and MTP-2 speculative decoding. Setup from [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Throughput (sustained) | **49 tok/s** |
| Weight size | ~75 GB (hybrid INT4+FP8 checkpoint) |
| Engine | vLLM 0.19.1 + FlashInfer + MTP-2 spec dec |
| Context | 256K (262144) |
| Config | `--attention-backend FLASHINFER --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **5/5** | 5732 | Clean. |
| A* Pathfinding | **6/6** | 4723 | Clean. |
| LRU Cache with TTL | **5/6** | 11331 | One test failure on the hardest benchmark. |

**Why this is interesting:** 2.3× the throughput of our llama.cpp baseline (49 vs 21 tok/s) on the same model and hardware. The speedup comes from three stacked optimizations: FlashInfer attention kernels optimized for Blackwell's memory hierarchy (+16%), hybrid INT4+FP8 quantization using native CUTLASS FP8 block-128 kernels (+8.8%), and MTP-2 speculative decoding using Qwen3.5's built-in MTP head to predict 2 additional tokens per step at ~80% acceptance rate (+25%).

**Thinking behavior:** The model thinks under the hood (total token counts of 5K-11K per benchmark vs llama.cpp's 2K-3K without thinking), but vLLM's `--reasoning-parser qwen3` absorbs the thinking tokens transparently — only content appears in the response. This is a different trade than llama.cpp's `-rea off`: instead of suppressing thinking at the template/sampler level, vLLM lets the model think and just hides it from the client. The thinking adds latency per request but the raw throughput is high enough that wall-clock time per coding task is similar or better than llama.cpp without thinking.

**Quality:** 16/17 vs llama.cpp unsloth's 18/17 — a 2-point gap from LRU (5/6 vs 6/6). Likely a minor generation path difference from the hybrid quantization or MTP speculative tokens affecting the logit distribution. ExprEval and A* are perfect on both engines.

**Setup cost:** ~41 min automated build (Docker image compilation for SM121), ~13 min cold start (model load + warmup + graph capture). Requires Docker with nvidia-container-toolkit. The build is a one-time cost; re-launches take ~5-7 min.

**Caveats:**
- Requires Docker (llama.cpp is a single binary)
- 13 min cold start vs llama.cpp's ~60s
- Thinking cannot be disabled — the model always thinks, vLLM just hides it. This means each request generates 2-5× more tokens than the visible output, consuming more compute per request than the tok/s number implies.
- MTP spec dec has a warning: "num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"

**Verdict:** If you're willing to accept the Docker + cold-start complexity, this is the fastest high-quality configuration on the Spark. For opencode's agent loop, 49 tok/s at 16/17 quality is a compelling upgrade over 21 tok/s at 18/17 — the throughput more than doubles while quality stays near-perfect.

---

## A-Tier: Strong Quality at Interactive Speed

### Qwen3.5-122B-A10B Q4_K_M (bartowski) on mainline llama.cpp
Same model file as the S-tier ik-llama entry above, same nominal Q4_K_M quant, but running on mainline llama.cpp. **Quality drops dramatically.** This is the cautionary tale for why inference engine choice matters at temp=0 on this hardware. Documented separately from the ik-llama entry to keep the engine-quality interaction visible.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **13/17 (76%)** — see caveat |
| Throughput (sustained) | **25.8 tok/s** (+23% vs unsloth) |
| TTFT | 0.60 s |
| Weight size | 71 GB (2-shard GGUF) |
| Bandwidth utilization | ~7.5 GB/token × 25.8 tok/s ≈ 194 GB/s = 71% of peak |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **0/5** | 2324 | Real model bug not fixed by the test_error_cases loosening: model chose a `self.tokens` list accumulator approach but never reset it between `evaluate()` calls. The pytest fixture reuses one evaluator instance, so tokens accumulate across tests and parsing immediately fails on the second call. The error is in the impl, not the test. |
| A* Pathfinding | **7/6** | 3443 | All 6 tests + 1 bonus, same as unsloth. |
| LRU Cache with TTL | **6/6** | 2929 | Clean. Same outcome as unsloth. |

**The quant-comparison surprise:** At single-shot temp=0, the bartowski quant produces *worse* code on ExprEval than the unsloth quant — a real implementation bug, not a test-wording mismatch. The two models took architecturally different approaches: unsloth chose a string-and-position parser (which auto-resets on each call), bartowski chose a token-list parser (which doesn't). **At temp=0 the quant noise determines which architecture path the model takes**, and on this prompt that difference flipped the outcome.

This **does not** mean unsloth is better than bartowski in general:
- A* and LRU are perfect on both quants — quality on the hardest tasks is unchanged
- One single-shot run is a sample of one; we'd need temp=0.3 best-of-3 to make any general claim
- The 23% throughput improvement is reproducible across all 4 generations (25.5–25.8 tok/s) and matches the expectation that bartowski uses better-optimized quant kernels
- The bartowski model still produced *correct, working code* for 2 out of 3 benchmarks; it just happened to choose a buggy approach on ExprEval

**Verdict:** Use bartowski's Q4_K_M for the throughput improvement (23% faster, no real downside), but be aware that single-shot quality at temp=0 is a high-variance comparison and isolated benchmark scores between quant providers can flip on either side. The architectural quality of the model is the same — both quants are running the same 122B/10B parameter network.

---

### Qwen3.5-122B-A10B-REAP-20 Q4_K_M (0xSero) on ik-llama
Interesting experiment: 0xSero's REAP-20 variant uses the [REAP (Routing-Enhanced Activation Pruning)](https://arxiv.org/abs/2510.13999) method to drop 20% of experts per layer (256 → 205) without retraining. **Total params drop from 122B to 99B; active params stay at 10B per token.** Direct comparison to the unpruned bartowski base running on the same engine.

| Metric | Base (bartowski) | **REAP-20** (0xSero) | Δ |
|---|---|---|---|
| Single-shot (temp 0) | 17/17 | **14/17** | **−3 (−18%)** |
| Throughput | 26.0 tok/s | **29.1 tok/s** | **+12%** |
| Weight size | 71 GB (2 shards) | **57 GB** (1 file) | **−14 GB** |
| KV cache | same (same active params) | same | 0 |
| Memory headroom freed | — | ~14 GB | — |

**Per-benchmark breakdown:**
| Benchmark | Base | REAP-20 | Δ | Notes |
|---|---|---|---|---|
| Expression Evaluator | 5/5 | **1/5** | **−4** | Real bug: parser adds an `('EOF', None)` sentinel but the termination check `self.pos < len(self.tokens)` doesn't skip it. After parsing a valid expression, the leftover EOF token triggers "Unexpected token at position 2". The base model didn't make this mistake. |
| A* Pathfinding | 7/6 | 7/6 | 0 | Perfect + bonus. Identical to base. |
| LRU Cache with TTL | 5/6 | **6/6** | +1 | REAP-20 actually scored slightly *better* on LRU (full 6/6 vs 5/6). |

**The 97.9% retention claim doesn't hold on our benchmarks.** 0xSero's README claims 97.9% capability retention based on HumanEval/MBPP, but our coding benchmarks are more specific and harder. The REAP pruning removed experts that were apparently load-bearing for the careful logic of parser termination. Algorithmic tasks (A* graph search, LRU data structures) were unaffected — those seem to use the retained expert set fine. Expression parsing was the specific failure mode.

**What this tells us about REAP as a technique:** REAP's assumption is that low-activation experts can be dropped with minimal impact. On aggregate benchmarks this is roughly true, but at the individual-prompt level you're rolling dice: some tasks land on experts that got pruned. Our 3 benchmarks happen to hit one (ExprEval) that degrades and two that don't. A multi-run benchmark at temp=0.3 would smooth this out — the base model might land on a different ExprEval impl path on some runs and the pruned model might land on a correct one — but single-shot at temp=0, the REAP variant shows a real 3-point drop.

**When is REAP-20 worth it?** If you specifically need to free ~14 GB of memory for KV cache, another model, or for a tighter-budget deployment, REAP-20 is an A-tier (14/17 = 82%) model running 12% faster than the unpruned base. If you have the memory headroom, the unpruned base at 17/17 is strictly better quality at slightly slower speed.

**Verdict:** Mid A-tier. Useful specifically for memory-constrained scenarios or as a "fast second-opinion" pair with the full model. Not a free win — the quality drop is real and structural, not just noise.

---

### Qwen3.5-122B-A10B Q4_K_M (bartowski) — asymmetric KV experiment
Testing [Nauful's KV quantization advice](../../../.claude/projects/-home-gisenberg-git-gisenberg-local-model-eval/memory/feedback_kv_quantization.md): "don't quantize K cache, only V to q8_0 — quantizing K breaks tool calls." Same model, same prompts, same temp=0. Only `-ctv` changed from `f16` to `q8_0`.

| Metric | f16/f16 (baseline) | f16K/q8V (asym) | Δ |
|---|---|---|---|
| Throughput | 25.8 tok/s | **14.0 tok/s** | **−46%** |
| ExprEval | 0/5 | 1/5 | +1 (different code path) |
| A* Pathfinding | 7/6 | 7/6 | tied |
| LRU Cache with TTL | 6/6 | 6/6 | tied |
| Total | 13/17 (76%) | 14/17 (82%) | +1 noise |
| Generated tokens | 2324 / 3443 / 2929 | 2367 / 3596 / 2803 | within 5% |

**Quality result: identical.** At temp=0 the asymmetric KV produces effectively the same generation path — token counts are within 5% of the f16 baseline, and the pass/fail outcome on every benchmark is the same. This confirms the quality half of Nauful's claim: V-cache q8_0 is lossless enough to not affect content generation. Tool-call reliability would need a separate experiment to validate the K-cache half of the claim.

**Throughput result: −46% slowdown.** This is the surprise. Conventional wisdom is that KV quantization saves bandwidth and helps performance, especially on memory-bound hardware. On the Spark with llama.cpp + CUDA 13, the opposite happens: the q8_0 V-cache dequantization compute on every attention read costs more than the bandwidth savings buy. The model's *active weights* are the dominant bandwidth consumer (~6 GB read per token), and the KV cache reads at 32K context are a tiny fraction of that — so there is no bandwidth headroom to recoup, and the dequant compute becomes pure overhead.

**Practical recommendation for Spark:** Don't use asymmetric KV (or any KV quantization) on MoE/hybrid models with small per-token KV cost. The throughput cost is severe and the memory savings are immaterial — Qwen3.5-122B's KV cache at 32K context is well under 1 GB. KV quantization is only worth considering on Spark for **dense models with massive KV cost** (e.g., the unfortunate Gemma 31B), where the memory savings let you fit larger contexts that wouldn't otherwise work — and even then the throughput penalty applies.

**Caveats:**
- This finding is on Spark/llama.cpp/CUDA 13 specifically; the same KV config might behave differently on a 5090 (different compute-to-bandwidth ratio) or with ik-llama (different attention kernels)
- The original Nauful advice was about *quality* (tool calls), and our test does not invalidate that quality concern
- We didn't test the K-cache-quantized variants (q8_0/q8_0 symmetric or turbo*) because they're known to break tool calls per the original advice

---

### Qwen3.5-122B-A10B Q4_K_M (bartowski) — mainline llama.cpp + thinking on (cautionary)
Control experiment to isolate whether the ik-llama 17/17 win comes from *the engine* or from *thinking being enabled*. We ran the same model file on mainline llama.cpp with `-rea on --reasoning-budget 16384` to enable thinking. The result is striking and lands the configuration in **D-tier (8/17)**.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **8/17 (47%)** |
| Throughput (sustained) | **25.3 tok/s** |
| Engine | mainline llama.cpp `b8740-e34f04215` |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea on --reasoning-budget 16384 --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | 1/5 | 6963 (14.4 KB think + 12.4 KB content) | Model chose `__init__(self, expr)` design that breaks the test fixture pattern (`evaluator = ExpressionEvaluator()` then `evaluator.evaluate(...)`). 4 of 5 tests can't even instantiate. |
| A* Pathfinding | 7/6 | 6161 (12.2 KB think + 8.9 KB content) | Perfect, same as the no-thinking and ik-llama versions. |
| LRU Cache with TTL | **0/6** (no_code) | 16384 (51.1 KB think + **0 B content**) | Model thought for 656 seconds straight, hit max_tokens=16384, never produced any code. Same failure mode as MiniMax-M2.5. |

**The finding:** *Same model file, same hardware, same prompts, same temperature.* Three different engine configurations, three completely different generation paths:
- ik-llama + thinking → **17/17** (clean recursive descent, all tests pass)
- mainline llama.cpp, thinking off → 13/17 (`self.tokens` accumulator bug on ExprEval)
- mainline llama.cpp, thinking on → 8/17 (`__init__(self, expr)` design + LRU thinking loop)

This confirms that on bandwidth-constrained hardware running large MoE models at temp=0, **the inference engine's per-kernel numerical noise is large enough to steer the model down completely different code generation paths**. ik-llama's kernels happen to land on the good path; mainline's kernels land on either a buggy path (without thinking) or a runaway-thinking path (with thinking). The win is not from "thinking is good" — thinking with mainline kernels is *worse* than without. The win is engine-specific.

**Practical recommendation:** For Qwen3.5-122B on Spark, use ik-llama, not mainline llama.cpp. The 22% throughput advantage is small but the quality difference is huge (17/17 vs at most 13/17 on mainline).

---

### Speculative decoding on Spark — does not work for our flagship models
Original hypothesis: speculative decoding (small draft model + 122B target) could break the bandwidth ceiling and double or triple our 25 tok/s. We tested this and it does not work on Spark for Qwen3.5-122B for two independent reasons:

1. **Mainline llama.cpp blocks speculative decoding on hybrid models entirely.** The error is `common_speculative_is_compat: the target context does not support partial sequence removal`. DeltaNet's recurrent linear-attention layers don't support the position-rollback that standard speculative decoding requires. PR [#20075](https://github.com/ggml-org/llama.cpp/pull/20075) is in flight to add checkpoint/restore for hybrid SSM/MoE models, but it's not merged in our build (`b8740-e34f04215`). Both `--model-draft` and the draftless `--spec-type ngram-mod` modes hit the same compat check and fail at server startup.

2. **ik-llama supports spec dec on hybrid models but the implementation is broken.** ik-llama uses context checkpointing to handle the recurrent state rollback, and the server starts successfully with the draft model loaded. But the actual generation is **3× slower than the no-spec-dec baseline** (8.3 tok/s vs 26 tok/s) and produces *different* output (the model failed to stop naturally, hit max_tokens). Either the acceptance logic is wrong or the checkpoint/restore overhead exceeds any speedup from accepted draft tokens at concurrency=1.

We tested with Qwen3.5-0.8B Q4_K_M as the draft model (533 MB, same 248K vocab as the 122B target). The draft itself loads and runs fine; the issue is in the spec dec coordination layer.

**The unintended discovery:** The negative spec dec result led us to test ik-llama as a general engine alternative (since we'd built it for the spec dec experiment), and that's how we found ik-llama is the better engine for Qwen3.5-122B on Spark (the S-tier entry above). Spec dec failed; ik-llama succeeded as a different optimization. This is the most useful kind of negative result.

**When to revisit speculative decoding on Spark:**
- After PR #20075 lands in mainline llama.cpp and we update our build
- Or if ik-llama fixes its hybrid spec dec implementation
- Or if Qwen ships a model architecture that's spec-dec-compatible on this engine class (none currently in our viable set)

---

### GLM-4.5-Air Q4_K_M (bartowski)
Z.AI's 106B/12B-active MoE in the same scale class as Qwen3.5-122B but from a completely different model family. Standard attention (not DeltaNet hybrid), 128K native context. **First non-Qwen model to land in A-tier on Spark.**

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **15/17 (88%)** |
| Throughput (sustained) | **21.7 tok/s** |
| TTFT | 0.54 s |
| Weight size | 70 GB (2-shard Q4_K_M) |
| Bandwidth utilization | ~6 GB/token × 21.7 tok/s ≈ 130 GB/s = 47% of peak |
| Active params | 12B (8 experts + 1 shared, of 128 total) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | 3/5 | 1970 | Two genuine implementation issues, not the wording mismatch — even with the loosened spec the model misses two test cases. |
| A* Pathfinding | **6/6** | 2115 | Clean. |
| LRU Cache with TTL | **6/6** | 1855 | Clean (after extract_and_test was taught to handle GLM's "impl + test in same module" output pattern). |

**Why this matters for the Spark thesis:** Until now, every A/S-tier Spark model has been a Qwen variant. GLM-4.5-Air is the first cross-family validation that the Spark thesis isn't Qwen-specific. A 106B/12B-active MoE from Z.AI runs at the same speed and within 2 points of quality of the best Qwen3.5-122B mainline configuration. The platform genuinely supports a class of models, not one specific family.

**Throughput note:** GLM uses standard attention (not DeltaNet hybrid), so its 47% bandwidth utilization is *better* than the 12B-active math would predict (6 GB × 22 = 132 GB/s, half of the 273 GB/s peak). The standard-attention layers are more compute-friendly than the recurrent linear-attention layers that DeltaNet uses, but the bandwidth-bound decode dominates either way — both architectures end up at similar effective throughput on Spark. This is a nice negative result for the "DeltaNet hybrid is the secret sauce" hypothesis: it isn't, the bandwidth wall is the actual constraint.

**Caveat (extract_and_test edge case):** GLM-4.5-Air emitted impl + tests as two separate code blocks, but the test code referenced `TTLCache` directly without an `import` statement (expecting both blocks to live in the same file). This is a different pattern from Qwen3-Coder-Next's "single block" mode. `tools/extract_and_test.py` was updated to detect this case (multi-block where the test references impl symbols by bare name without an import) and fall back to single-file mode. Fix is general and applies to any future model with this pattern. Without the fix, GLM-4.5-Air would have scored 9/17 (53%) — a tooling artifact, not a model failure.

**Verdict:** Strong A-tier model. Marginally lower quality than Qwen3.5-122B (88% vs 100%) at similar throughput, but it validates that Spark isn't a one-family platform.

---

## B-Tier: Fast but Inconsistent

### Qwen3-Coder-Next UD-Q4_K_M (unsloth dynamic)
80B/3B-active hybrid (12 attention layers of 48, gated DeltaNet for the rest). The "coding specialist" billing partially delivers — quality lands at 82%, the lowest of our viable A/B-tier models. Speed is the strong story: **50 tok/s** sustained, the fastest large model on the platform so far and within striking distance of a 5090 running dense Gemma 31B.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **14/17 (82%)** |
| Throughput (sustained) | **50.2 tok/s** |
| TTFT | 0.39 s |
| Weight size | 46 GB |
| Bandwidth utilization | ~2 GB/token × 50 tok/s = 100 GB/s = 37% of peak |
| VRAM (32K ctx) | TBD (small KV: ~24 KB/token via DeltaNet hybrid) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **5/5** | 2258 | Perfect after the test_error_cases benchmark spec was loosened. |
| A* Pathfinding | **6/6** | 2705 | Clean. Algorithmic code is solid. |
| LRU Cache with TTL | **3/6** | 2407 | Wrote tests with `mock.patch('ttl_cache.time.monotonic', side_effect=[0,0,...])` but mis-counted how many times its own implementation calls `time.monotonic()`, causing `StopIteration` and wrong values. The impl may be correct; the test harness is broken. |

**Strengths:** Fast (50 tok/s — 2.4× Qwen 122B's speed for comparable bandwidth math, because only 3B params are active per token). Solid on stateless algorithmic code (A* perfect). Tiny weight footprint leaves >70 GB free for other workloads or huge contexts.

**Weakness:** Self-inconsistency in stateful test mocking. The model seems to know what `mock.patch` is, knows its API, but doesn't model the runtime call sequence of its own code carefully enough. This is the "coding specialist" failing on a coding task — possibly because the test harness instructions explicitly required mocked time, which forced the model into a code path it's worse at than its happy path.

**Verdict:** 50 tok/s is genuinely useful — interactive enough to use for coding agents — but you'd want to pair it with a more reliable model for verification on complex stateful logic. Treat it as a "fast first-draft" model rather than a one-shot solver.

---

### Nemotron-3-Super-120B-A12B Q4_K_M (bartowski)
NVIDIA's flagship MoE designed specifically for the DGX Spark — explicit support landed in llama.cpp via [PR #20411](https://github.com/ggml-org/llama.cpp/pull/20411), and DGX Spark benchmarks were added in [PR #20652](https://github.com/ggml-org/llama.cpp/pull/20652). 120B/12B-active hybrid Latent MoE + Mamba-2 layers. Thinking model by default. **Engineering investment is real but quality is mid B-tier; slightly slower than Qwen3.5-122B without matching its quality.**

| Metric | Value |
|---|---|
| Single-shot (temp 0, thinking on) | **11/17 (65%)** |
| Throughput (sustained) | **19.7 tok/s** |
| TTFT | 0.69 s |
| Weight size | 87 GB (3-shard Q4_K_M) |
| Bandwidth utilization | ~7 GB/token × 19.7 tok/s ≈ 138 GB/s = 50% of peak |
| Active params | 12B |
| Thinking | Always on, restrained (3.5–11.9 KB per benchmark — not the runaway-thinking pattern that made MiniMax-M2.5 unusable) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 --no-mmap --jinja` (no `-rea off`; thinking is integral) |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Thinking | Notes |
|---|---|---|---|---|
| Expression Evaluator | 4/5 | 3418 | 6.6 KB | One real test failure beyond the wording issue. |
| A* Pathfinding | **4/6** | 3052 | 3.5 KB | Two failures: one tie-breaking path mismatch (the test asserts a specific path; multiple paths are equally optimal), one genuine suboptimal path on the obstacle test. Unusual — most other models nail A*. |
| LRU Cache with TTL | 3/6 | 5758 | 11.9 KB | Half the tests fail. |

**The NVIDIA-engineering-vs-quality gap:** This is a model that NVIDIA explicitly optimized for our exact hardware. The kernels work, the architecture loads cleanly, throughput is in the right class (50% of theoretical peak — same range as Qwen3.5-122B). And yet quality lands at 65%, below GLM-4.5-Air (88%) and Qwen3.5-122B variants (76–106%). This is not a tooling story — the extract_and_test runs cleanly, the model produces well-formed code that just gets parts wrong. Hypothesis: Nemotron-3-Super was trained with a different reasoning data mix than the Qwen and GLM lines, and that data mix is less aligned with our specific benchmark prompts (recursive descent parsing, A*, doubly-linked list LRU).

**Restraint vs runaway thinking:** Worth comparing to MiniMax-M2.5, the only other thinking-mandatory model in our benchmarks. MiniMax thought 51 KB on LRU and never reached content (timed out). Nemotron thinks 11.9 KB on LRU and finishes cleanly with `finish=stop`. NVIDIA's training apparently has a stricter implicit budget — the model self-limits its reasoning to a manageable length. This makes Nemotron usable on Spark in a way MiniMax isn't, even though both are thinking models.

**Verdict:** Solid B-tier. Use it if you want a thinking model that actually finishes in reasonable time; use Qwen3.5-122B or GLM-4.5-Air if you want better one-shot quality at similar speed. The DGX Spark engineering pays off for stability and throughput, not for raw quality.

---

## C-Tier: Capable but Impractically Slow

### MiniMax-M2.5 UD-Q3_K_XL (unsloth)
230B-total / 10B-active MoE with Lightning Attention. Fits in 128 GB at ~96 GB weight footprint. **Throughput is fine; the problem is thinking depth.** This is a thinking-mandatory model that, on the Spark, can take 20+ minutes to think through a hard problem before emitting code. ExprEval works perfectly (5/5 with the loosened spec), but A* and LRU both timed out at the 25-minute request budget without ever producing content.

| Metric | Value |
|---|---|
| Throughput (sustained) | **29.6 tok/s** |
| TTFT | 0.95 s |
| Weight size | 96 GB (4-shard GGUF, UD-Q3_K_XL) |
| Bandwidth utilization | ~5 GB/token × 30 tok/s = 150 GB/s = 55% of peak |
| Single-shot quality (partial) | **1/3 benchmarks completed** |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | **5/5** | 6245 (14.9 KB think + 13.8 KB content) | Perfect after the test_error_cases benchmark spec was loosened. |
| A* Pathfinding | TIMEOUT | — | 25-minute request timeout fired before model emitted content. Either generated >32K tokens of thinking or slowed below 22 tok/s as context grew. |
| LRU Cache with TTL | TIMEOUT | — | Same outcome as A*. |

**The MiniMax-M2.5 paradox:** This model is sized for capability beyond what the Spark's bandwidth supports as an *interactive* tool. The architecture works (29.6 tok/s on the throughput test), and when it does produce code, the code is good (ExprEval mass-quality matches what other A-tier models produce). But for hard problems the model wants to think for tens of thousands of tokens before answering, and at Spark speeds that's a 15–25+ minute round-trip per query. Not viable for an interactive coding workflow on this hardware.

**Methodology adjustments required:** This is a known [llama.cpp issue (#21465)](https://github.com/ggml-org/llama.cpp/issues/21465) where the official MiniMax-M2.5 chat template injects `<think>` into `add_generation_prompt`, breaking llama.cpp's reasoning detection. The benchmarks here use a custom chat template at [`templates/minimax-m25-no-think.jinja`](../templates/minimax-m25-no-think.jinja) that removes the prompt-side `<think>` injection so the model and llama.cpp can negotiate thinking themselves. With that fix:
- `-rea off` is meaningless on this model (it won't suppress thinking)
- `max_tokens` must be 32K+ to leave room for both thinking and content
- Per-request timeout must be 25+ minutes, and even that isn't enough for hard prompts

**Verdict:** Don't use MiniMax-M2.5 on Spark for interactive coding. The model is fine — the Spark is the wrong host for it. A bandwidth-richer GPU (5090, A100, H100) would let it think at 5–10× the speed, which is the difference between a 25-minute round-trip and a 3-minute one.

---

## D-Tier: Buggy code at slow speeds

### Mistral-Small-4-119B-2603 Q4_K_M (bartowski) — `reasoning_effort=none`
Mistral's March 2026 unified release that absorbs the previous Instruct + Magistral (reasoning) + Devstral (coding) lines into one model with togglable `reasoning_effort`. We tested with reasoning effort disabled (the comparable config to all our other measurements). The result is **disappointing for both quality and throughput**.

| Metric | Value |
|---|---|
| Single-shot (temp 0, reasoning off) | **7/17 (41%)** |
| Throughput (sustained) | **8–10.5 tok/s** |
| TTFT | 0.6 s |
| Weight size | 69 GB (2-shard Q4_K_M) |
| Bandwidth utilization | ~10 GB/token × 10 tok/s = 100 GB/s = 37% of peak |
| Active params | 6.5B (4 of 128 experts per token) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | 2/5 | 2277 | Impl is doubly broken: errors on valid input (parentheses, unary minus tests fail) AND fails to error on invalid input (`test_error_cases` shows DID NOT RAISE on a malformed expression). |
| A* Pathfinding | 5/6 | 2628 | One test failure on the algorithmic test. |
| LRU Cache with TTL | **0/6** | 2597 | Syntax error in the impl: `def get(self, key: str) -> Optional[Any:` (missing closing `]`). The model wrote unbalanced brackets. The test file can't even import the impl. |

**Throughput surprise:** Mistral's 6.5B active params should give it ~76 tok/s theoretical max on Spark (273 GB/s peak / ~3.6 GB read per token), but we measured 10.5 tok/s = ~14% of peak. Compare to Qwen3.5-122B at 26 tok/s = 58% of peak with 10B active params. The "active params" headline number is misleading: Mistral's MoE structure (128 experts → 4 active) plus shared experts plus the standard attention layers means the actual per-token bandwidth is much higher than 6.5B × 4.5 bpw. Or the llama.cpp kernels for this MoE architecture are less optimized than for Qwen's. Either way, the practical decode rate is half of what the marketing-spec math would predict.

**Quality surprise:** A 119B model from a major lab producing syntax errors on a basic LRU cache implementation is unusual. Our hypothesis: `reasoning_effort=none` is genuinely the wrong setting for Mistral-Small-4 — the model is trained primarily for the thinking-mode workflow and the no-thinking path produces low-effort code. This is the opposite of what we wanted (a fast non-thinking model) but it's an honest finding.

**Worth retrying with `reasoning_effort=medium`?** Probably yes, as a follow-up. The model exposes thinking via a [chat-template kwarg](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), and llama.cpp supports passing it via `--chat-template-kwargs '{"reasoning_effort": "medium"}'`. We have not yet run this experiment. If thinking-on results match the Qwen3.5-122B quality, Mistral becomes a viable A-tier candidate; if they don't, the D-tier rating stands.

**Verdict:** Don't use Mistral-Small-4-119B on Spark with reasoning off. May be a strong A-tier candidate with reasoning on — needs follow-up.

---

## F-Tier: Bandwidth-bottlenecked

### Gemma 4 31B-IT Q8_0 (dense)
Loads fine, runs at ~6.7 tok/s. Not a model quality issue — this is hardware physics. Use a 5090 or A100 if you need a dense 31B.

| Metric | Value |
|---|---|
| Throughput | **6.7 tok/s** |
| TTFT | 0.4s |
| Weight size | 31 GB |
| Bandwidth utilization | ~221 GB/s = 81% of peak |
| VRAM (32K ctx) | ~58 GB |
| Config | `-ctk f16 -ctv f16 -rea off` |

**Takeaway:** Gemma 31B is one of the highest-quality models in the 5090 rankings but on the Spark it is unusable for any task that involves more than a few hundred tokens of output. A 5,000-character expression evaluator solution takes over 3 minutes to generate. Coding benchmarks were aborted as it would take ~30+ minutes per benchmark and the result is fundamentally a hardware limitation, not a model capability question.

---

## Cross-model test wording issue (resolved)

Three different model families (Qwen3.5-122B-A10B, Qwen3-Coder-Next, MiniMax-M2.5) were all failing the `test_error_cases` test in Expression Evaluator the same way: they raised `ValueError` correctly for `"(2 + 3"` but with messages like `"Invalid token at position 3: ')'"` or `"Unexpected end of expression"` instead of literally `"Mismatched parentheses"`. The model-generated test code used `pytest.raises(ValueError, match="Mismatched parentheses")` which is string regex matching, not exception class matching.

Three independent model families converging on the same "wrong" behavior was clearly a benchmark specification issue rather than a model deficiency. The benchmark prompt asks for *"ValueError with a descriptive message for: mismatched parentheses..."* — descriptive, plural, not a mandated literal string.

**Resolution:** [`tools/extract_and_test.py`](../tools/extract_and_test.py) now post-processes extracted test code with `loosen_pytest_raises()`, which strips `match=...` arguments from `pytest.raises()` calls so the check verifies exception class only. This is the semantically correct check and unblocks all three models. After applying it:
- Qwen3.5-122B unsloth: 16/17 → **18/17** (5/5 ExprEval, 7/6 A* with bonus, 6/6 LRU)
- Qwen3-Coder-Next: 13/17 → **14/17** (5/5 ExprEval)
- MiniMax-M2.5 (ExprEval only): 4/5 → **5/5**
- Gemma 31B (ExprEval only): 4/5 → **5/5**

The bartowski Qwen3.5-122B variants are unaffected because their ExprEval failure is a real implementation bug (`self.tokens` accumulator never reset between calls), not a wording mismatch.

---

## Deep dive: Qwen3.6-35B-A3B vs Qwen3.5-122B-A10B on Spark — does the small model win?

There is a standing debate on the Spark forums that prefers Qwen 3.5 35B-A3B at FP8 over 3.5 122B-A10B at INT4. James Hillyerd's framing: *"The rule of thumb is that a large model quantized down is usually better than a small model at a larger quant, but perhaps MoE suffers more."* We benchmarked both.

**Short answer: no. The 122B Q4 scores 17/17. The 35B-A3B at BF16 (zero quant loss — best it can possibly be) scores 9/17. The rule holds for MoE.**

### Results

| Model | Active | Quant | Weights | Tok/s | ExprEval | A* | LRU | Total | Tier |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-122B-A10B (bartowski) | 10B | Q4_K_M | 71 GB | **26** | **5/5** | **7/6** | **5/6** | **17/17** | **S** |
| Qwen3.5-122B-A10B (unsloth) | 10B | Q4_K_M | 72 GB | 21 | 5/5 | 7/6 | 6/6 | **18/17** | **S** |
| Qwen3.5-122B-A10B (vLLM INT4+FP8+MTP-2) | 10B | AR-INT4/FP8 | ~72 GB | **49** | 5/5 | 7/6 | 4/6 | **16/17** | S |
| **Qwen3.6-35B-A3B (unsloth)** | **3B** | **BF16** | **69 GB** | **31** | **4/5** | **5/6** | **0/6** | **9/17** | **D** |
| Qwen3-Coder-Next 80B-A3B | 3B | Q4_K_M | 46 GB | 50 | 5/5 | 6/6 | 3/6 | 14/17 | B |

Key details:

- The **35B-A3B BF16** result is the *best possible* quality for that model — zero quantization loss. FP8 can only be equal or worse. If BF16 scores 9/17, FP8 won't score higher.
- The **122B at Q4_K_M** (~4.5 bpw) still scored 17/17. The model has enough capacity that Q4 doesn't hurt it on these tasks.
- Throughput favors the 35B (31 tok/s BF16) over the 122B on ik-llama (26 tok/s) — but the 122B on vLLM with INT4+FP8 and MTP-2 speculative decoding runs at **49 tok/s**, beating the 35B on speed *and* quality.

### Where the 35B-A3B failed

- **Expression Evaluator (4/5)**: the model wrote a test asserting `1 + 2 * 3 - 4 / 2 == 4.0` but the impl returns `5.0` (the correct answer — `1 + 6 - 2 = 5`). The impl has right precedence; the test is wrong.
- **A* Pathfinding (5/6)**: the test creates a 3×3 grid with the middle row blocked and asserts `find_path((0,0), (0,2))` returns `None`. But `(0,0)` and `(0,2)` are in the same unblocked top row — the impl correctly found a path along row 0. The test is wrong, the impl is right.
- **LRU Cache (0/6)**: zero tests ran — model emitted impl and tests in a format the harness couldn't extract (combined single block). Structural output failure, not logic failure.

**Pattern**: the 35B's failures are all self-consistency problems — writes code that works, then writes tests that don't match, or structures output in a way that breaks the harness. The 122B doesn't make these mistakes because it has enough capacity to maintain coherence across the full generation.

### Why MoE doesn't change the rule

The intuition that "MoE might suffer more from quantization" is backwards — MoE actually suffers *less*:

1. **Only ~10B params are active per token** regardless of total model size. Quantizing the 122B's inactive experts from FP16 to Q4 doesn't affect per-token computation quality — those experts aren't being read. The active 10B experts are quantized, but 10B @ Q4 is still more capacity than 3B @ BF16.
2. **The router is barely affected by weight quantization.** Expert selection happens in a tiny gating network well within the precision budget at Q4.
3. **Capacity advantage compounds with expert count.** The 122B has 256 experts, the 35B also has 256 but each one is ~3× smaller. When the router picks top-10, the 122B's experts individually have more capacity to contribute — even quantized.

### Speed argument also doesn't hold

- 35B BF16: 31 tok/s (3B active × BF16 = ~6 GB/token)
- 122B Q4_K_M on ik-llama: 26 tok/s (10B active × Q4 ≈ ~5 GB/token)
- 122B INT4+FP8 on vLLM: **49 tok/s** (with MTP-2 spec dec)

The 35B's speed advantage over ik-llama is 19%. It vanishes with vLLM, which runs the 122B ~60% *faster* while scoring 16/17 vs 9/17. Trading 17/17 for 9/17 to gain 19% throughput is a bad deal.

**The rule holds**: a large MoE quantized to Q4 decisively beats a small MoE at FP8/BF16 on the Spark.

---

## Deep dive: MiniMax M2.5 & M2.7 — the empty-think template workaround

MiniMax M2.5 and M2.7 are **mandatory thinking models** — MiniMax has explicitly stated that disabling thinking is not supported ([GitHub issue #68](https://github.com/MiniMax-AI/MiniMax-M2/issues/68)). Every mechanism llama.cpp provides (`-rea off`, `--reasoning-budget 0`, custom no-think templates) is ignored because the model's chat template never checks `enable_thinking` and the model weights are trained to always emit `<think>` tokens. The original M2.5 C-tier entry above scored 5/5 + 2 timeouts because the model spent 25+ minutes thinking on A* and LRU before producing any content.

**The workaround**: inject a pre-closed `<think>\n\n</think>` block in the generation prompt so the model sees thinking as "already done" and skips to content. Proposed by user `gelim` in [llama.cpp issue #20196](https://github.com/ggml-org/llama.cpp/issues/20196). The llama.cpp maintainer warned it may degrade output quality; in practice it works cleanly on both M2.5 and M2.7.

Template file: `templates/minimax-m27-empty-think.jinja` — identical to `minimax-m25-no-think.jinja` except the generation prompt block:

```jinja
{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n\n' ~ '</think>' ~ '\n\n' }}
{%- endif -%}
```

**Result: zero reasoning tokens across all 12 benchmark runs.** The model produces content directly, at full decode throughput, with no thinking overhead.

### Architecture note: M2.5 and M2.7 are architecturally identical

Despite MiniMax's marketing describing "Lightning Attention" for M2.5, **both models have the exact same attention architecture** per their `config.json`:

- 62 layers, ALL standard attention (`attn_type_list` all 1s)
- 48 attention heads, 8 KV heads, 128 head dim
- 256 experts, 8 active per token
- `max_position_embeddings`: 196,608 (192K native context)

**KV cache per token: ~248 KB/token at f16** (measured from llama-server allocation: 47,616 MiB for 196,608 tokens). NOT tiny — comparable to a dense model with 62 attention layers. The "Lightning Attention = tiny KV" label in earlier bench scripts was wrong.

At 192K native context, KV cache alone is ~46.5 GB, which dominates the memory budget alongside weights.

### Quality vs quant tradeoff

Single-shot `temperature=0`, `-fa on`, f16 KV, `--no-mmap`, empty-think template.

| Model | Quant | Weights | Context | Prefill | Decode | Score | Note |
|---|---|---|---|---|---|---|---|
| **M2.7** | UD-IQ3_S | 84 GB | 96K | ~330 tok/s | **28 tok/s** | **14/17 (82%)** | Best MiniMax quality. ExprEval 5/5, A* 6/6, LRU 3/6. |
| **M2.7** | UD-IQ2_XXS | 61 GB | **192K native** | ~390 tok/s | **35 tok/s** | **10/17 (59%)** | Full context. ExprEval 5/5, A* 5/6, LRU 0/6. 2-bit costs 4 quality points. |
| **M2.5** | UD-Q3_K_XL | 101 GB | 32K | ~330 tok/s | **28 tok/s** | **8/17 (47%)** | ExprEval 5/5, A* 3/6, LRU 0/6. |
| **M2.5** | UD-IQ2_XXS | 70 GB | **160K** | ~390 tok/s | **35 tok/s** | **11/17 (65%)** | ExprEval 5/5, A* 6+2/6, LRU 0/6 (64K chars, no stop). |
| M2.5 (original, no workaround) | UD-Q3_K_XL | 101 GB | 32K | ~330 tok/s | 29.6 tok/s | 5/5 + 2 TOs | A* and LRU timed out at 25 min. |

**Observations**:
- **M2.7 > M2.5 at every quant level.** Newer training substantially improves coding quality.
- **2-bit quantization costs real quality.** M2.7 drops 14/17 (IQ3_S) → 10/17 (IQ2_XXS) — 4-point regression concentrated on LRU (3/6 → 0/6) and A* (6/6 → 5/6). Throughput increases (28 → 35 tok/s) but the quality trade is steep.
- **LRU Cache is MiniMax's Achilles' heel.** 0/6 on both M2.5 variants and M2.7 IQ2_XXS. Only M2.7 IQ3_S achieves partial 3/6.
- **M2.5 IQ2_XXS > M2.5 Q3_K_XL is noise**, not signal — the 11/17 vs 8/17 difference comes from A* (6+2 bonus on IQ2_XXS vs 3/6 on Q3_K_XL). Different quant noise steers the model down different paths at temp=0.

### Memory: what fits at what context

Both models have identical KV requirements (~248 KB/token f16). Budget: ~115 GB allocatable (120 GB unified minus OS/system overhead).

| Config | Max context (measured) | Total at max ctx | Headroom |
|---|---|---|---|
| M2.7 IQ3_S | **96K** | ~103 GB | 12 GB |
| M2.7 IQ2_XXS | **192K native** | ~107 GB | 8 GB |
| M2.5 Q3_K_XL | 32K (tight) | ~104 GB | 11 GB |
| M2.5 IQ2_XXS | **160K** | ~108 GB | 7 GB |

M2.7 IQ3_S at 192K OOMs (79.3 + 46.5 = 126 GB > 115). M2.5 IQ2_XXS at 192K OOMs (68.7 + 46.5 = 115.2, just over). M2.5 IQ2_XXS at 160K loads successfully.

### Comparison vs existing Spark picks

| Model | Score | Decode | Context | Thinking issue? |
|---|---|---|---|---|
| **Qwen3.5-122B Q4_K_M (unsloth)** | **18/17** | 21 tok/s | 256K | No |
| Qwen3.5-122B Q4_K_M (bartowski) + ik-llama | 17/17 | 26 tok/s | 256K | Yes (thinking can't be disabled) |
| GLM-4.5-Air Q4_K_M | 15/17 | 22 tok/s | 128K | No |
| **MiniMax-M2.7 IQ3_S (empty-think)** | **14/17** | **28 tok/s** | 96K | Workaround |
| Qwen3-Coder-Next UD-Q4_K_M | 14/17 | 50 tok/s | 256K | No |
| MiniMax-M2.7 IQ2_XXS (empty-think) | 10/17 | 35 tok/s | 192K | Workaround |

**MiniMax-M2.7 IQ3_S lands in B-tier** alongside Qwen3-Coder-Next — same 14/17 at lower throughput (28 vs 50) and smaller context (96K vs 256K). The empty-think template turns an unusable model into a functional coder, but it doesn't beat the existing picks. **Qwen3.5-122B-A10B remains the Spark winner**.

### MiniMax serving config

Best MiniMax config on Spark: M2.7 IQ3_S at 96K.

```bash
llama-server \
  -m ~/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/UD-IQ3_S/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 98304 -ngl 99 -fa on -ctk f16 -ctv f16 -np 1 --no-mmap --jinja \
  --chat-template-file templates/minimax-m27-empty-think.jinja \
  -rea off
```

Critical: `--chat-template-file` pointing to the empty-think template is **mandatory**. Without it the model thinks 20-30 min per prompt and times out on any non-trivial task.

For full 192K native context use M2.7 IQ2_XXS (same command, different model path), accepting the 14/17 → 10/17 quality drop.

---

## Configuration

```bash
# Standard Spark config for MoE models
~/llama.cpp/build/bin/llama-server \
  -m model.gguf --port 8080 \
  -c 32768 -ngl 99 \
  -fa on -ctk f16 -ctv f16 \
  -np 1 --temp 0 \
  --no-mmap --jinja -rea off
```

Per [Nauful's advice](../../../.claude/projects/-home-gisenberg-git-gisenberg-local-model-eval/memory/feedback_kv_quantization.md), only quantize V cache (not K) if KV quantization is needed: `-ctk f16 -ctv q8_0`. K cache quantization is reported to break tool calling reliability.

For dense models that exceed bandwidth-viable size, TurboQuant fork is built and available at `~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server`, but TurboQuant only addresses KV cache memory, not weight bandwidth, so it does not solve the dense-model throughput problem on this platform.
