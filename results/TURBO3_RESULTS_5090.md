# TurboQuant KV Cache Benchmark Results — RTX 5090

Tested April 2026 on RTX 5090 32 GB. TurboQuant fork of llama.cpp (`feature/turboquant-kv-cache` branch), CUDA 13.2, flash attention on, temperature 0, `max_tokens=16384`.

For the M4 Max story (where TurboQuant has the *opposite* effect), see [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md).

## TL;DR

**turbo4 KV + thinking off (`-rea off`) is the optimal configuration.** Three models scored 100% (17/17) — including Gemma 31B which previously maxed out at 10/17. turbo4 provides 3.8x KV compression with only +0.23% PPL impact, and disabling thinking eliminates the reasoning loop failures that plagued turbo3.

turboquant_plus documents that **Q4_K_M weights + symmetric turbo is the risky combination** — Q6_K weights consistently outperform Q4_K_M under KV quantization because higher weight precision gives the attention mechanism headroom to absorb KV noise.

## Best Results: turbo4 + Thinking Off (Recommended)

| Tier | Model | Quant | KV Config | Thinking | VRAM (32K) | Tok/s | TTFT | Expr Eval (5) | A* Path (6) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** | gemma-4-26b-a4b | Q6_K | turbo4/turbo4 | off | 25,636 MB | 142.2 | 2.32s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Perfect. Fastest S-tier at 121-124 tok/s gen. |
| **S** | gemma-4-31b-it | Q4_K_M | turbo4/turbo4 | off | 22,293 MB | 53.3 | 2.20s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Transformed from D-tier. Dense arch works with turbo4. |
| **A** | gemma-4-26b-a4b | Q4_K_M | turbo4/turbo4 | off | 20,046 MB | 156.1 | 2.31s | 5/5 | 6/6 | 5/6 | **16/17 (94%)** | Lowest VRAM. One LRU lazy-cleanup edge case. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | turbo4/turbo4 | off | 24,549 MB | 51.7 | 2.38s | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Reliable. Embeds reasoning in content even with -rea off. |
| **C** | qwen3.5-35b-a3b | Q4_K_M | turbo4/turbo4 | off | 23,910 MB | 188.1 | 2.31s | 4/5 | 7/7 | 0/6 | **11/17 (65%)** | Fast but LRU consistently fails. Q4_K_M sensitivity. |

## Previous Best: turbo4 + Thinking On (with reasoning budget)

| Tier | Model | Quant | KV Config | VRAM (32K) | Tok/s | TTFT | Expr Eval (5) | A* Path (6) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** | gemma-4-26b-a4b | Q6_K | turbo4/turbo4 | 25,162 MB | 146.5 | 2.42s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Perfect score. turbo4 preserves thinking quality. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | turbo3/turbo3 | 24,035 MB | 52.9 | 2.23s | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Heavy thinker but reliable. Handles turbo3 well. |
| **B** | gemma-4-26b-a4b | Q4_K_M | turbo4/turbo4 | 19,555 MB | 155.6 | 2.29s | 0/5 | 7/7 | 6/6 | **13/17 (76%)** | Fast, lowest VRAM. ExprEval had import errors. |
| **C** | qwen3.5-35b-a3b | Q4_K_M | q8_0/q8_0 | 23,715 MB | 196.7 | 2.38s | 0/5 | 6/7 | 0/6 | **6/17 (35%)** | Fastest throughput but inconsistent code. |
| **D** | gemma-4-31b-it | Q4_K_M | turbo3/turbo3 | 21,499 MB | 54.0 | 2.34s | 4/5 | 0/6 | 0/6 | **4/17 (24%)** | Dense arch needed turbo3. Struggles with thinking on. |

Reasoning budget flags: `--reasoning-budget 12288` for Gemma 26B models, `--reasoning-budget 16384` for Gemma 31B, unrestricted for Qwen and Qwopus.

## Thinking Off vs Thinking On (turbo4 KV)

| Model | turbo4 + thinking ON | turbo4 + thinking OFF | Delta |
|---|---|---|---|
| Gemma 26B Q6_K | 17/17 (100%) | **17/17 (100%)** | Same — both perfect |
| Gemma 31B Q4_K_M | 4/17 (24%) | **17/17 (100%)** | **+13 tests** |
| Gemma 26B Q4_K_M | 13/17 (76%) | **16/17 (94%)** | +3 tests |
| Qwopus 27B Q6_K | 16/17 (94%) | **16/17 (94%)** | Same |
| Qwen 35B Q4_K_M | 6/17 (35%) | **11/17 (65%)** | +5 tests |

Thinking off is equal or better for every model. The Gemma 31B result is transformative — from D-tier to S-tier.

## turboquant_plus Findings

The [turboquant_plus](https://github.com/TheTom/turboquant_plus) research suite documents validated configurations and quality tradeoffs:

### Q4_K_M Weight Sensitivity (Critical)

Q4_K_M weights + symmetric turbo KV is the documented risky combination. Smaller models (7B) can see catastrophic PPL (3556x). Larger models tolerate it better but show reduced quality vs Q6_K/Q8_0 weights. Our results confirm this:

| Weight Quant | turbo4 + no-think | Notes |
|---|---|---|
| Q6_K | 17/17 (100%) | Consistently perfect |
| Q4_K_M | 11-16/17 (65-94%) | Model-dependent, always worse than Q6_K |

### Recommended Configs (from turboquant_plus)

| Scenario | Config | Notes |
|---|---|---|
| Safe default | `-ctk q8_0 -ctv turbo4 -fa on` | Asymmetric — **GPU offload broken**, CPU only |
| Quality-critical (code) | `-ctk turbo4 -ctv turbo4 -fa on` | Symmetric, validated for code generation |
| Max compression | `-ctk turbo3 -ctv turbo3 -fa on` | +1.06% PPL, avoid on Q4_K_M weights |
| Reasoning models | `-ctk turbo4 -ctv turbo4 -rea off` | **Our recommended config** |

### Asymmetric K/V Limitation

turboquant_plus recommends asymmetric K/V (e.g., `-ctk q8_0 -ctv turbo3`) as the safest config since K precision controls quality. However, [ggml-org/llama.cpp#20866](https://github.com/ggml-org/llama.cpp/issues/20866) documents that **asymmetric K/V types cannot be GPU-offloaded** — they force CPU processing, dropping throughput to ~30 tok/s. Until fixed, symmetric turbo4/turbo4 is the practical optimum.

## Why Per-Model KV Configs Matter

Not all models respond equally to KV cache quantization. The key factor is **architecture**:

| Model | Architecture | KV/Token (f16) | turbo3 Saves | Recommendation |
|---|---|---|---|---|
| qwen3.5-35b-a3b | DeltaNet hybrid (10 attn layers) | ~20 KB | ~15 KB | **q8_0** — savings negligible, turbo causes thinking loops |
| qwopus-3.5-27b-v3 | DeltaNet hybrid (16 attn layers) | ~64 KB | ~48 KB | **turbo3** — moderate savings, handles quantization well |
| gemma-4-26b-a4b | Sliding window + global (15+15) | ~120 KB | ~90 KB | **turbo4** — significant savings, turbo3 too aggressive |
| gemma-4-31b-it | Dense (50 sliding + 10 global) | ~870 KB | ~650 KB | **turbo3** — must compress, dense arch is VRAM-hungry |

### Key Insight: K Precision Controls Quality

Research from the TurboQuant community confirms that **key (K) precision is the dominant quality factor** because keys control attention routing via softmax. Value (V) compression is comparatively cheap quality-wise. This is why turbo4 (+0.23% PPL) is dramatically safer than turbo3 (+1.06% PPL) for thinking models — the extra bit per key preserves the attention patterns that control "stop thinking, start coding" transitions.

## Comparison: Optimal TurboQuant vs LM Studio (f16 KV)

| Model | LM Studio (f16) | TurboQuant (optimal) | Quality Delta | Speed Delta |
|---|---|---|---|---|
| gemma-4-26b-a4b Q6_K | 16/17 (94%) | **17/17 (100%)** | **+1 test** | 146 vs 162 tok/s (-10%) |
| qwen3.5-35b-a3b Q4_K_M | 19/19 (100%) | 6/17 (35%) | -13 tests | 197 vs 93 tok/s (+112%) |
| gemma-4-26b-a4b Q4_K_M | -- (partial) | 13/17 (76%) | N/A | 156 vs 169 tok/s (-8%) |

Gemma Q6_K with turbo4 **exceeds** its own LM Studio score — the reasoning budget cap and turbo4 KV produced better-quality output than unconstrained f16 KV through LM Studio. The throughput cost is only 10%.

Qwen 35B remains problematic even with q8_0 KV. The model's behavior differs significantly between LM Studio and llama-server due to different FP accumulation order (parallel slots, batch scheduling). This is a determinism issue, not a KV quant issue.

## Full Context Window Analysis

With these optimal KV configs, can each model reach its full 262K native context on the RTX 5090?

| Model | Weights | KV Config | KV at 262K | Total Est. | Fits 32 GB? | Max Usable Ctx |
|---|---|---|---|---|---|---|
| qwen3.5-35b-a3b Q4_K_M | ~21 GB | q8_0/q8_0 | ~5.2 GB | **~26 GB** | **Yes** | **262K (full)** |
| qwopus-3.5-27b-v3 Q6_K | ~23 GB | turbo3/turbo3 | ~4.2 GB | **~27 GB** | **Yes** | **262K (full)** |
| gemma-4-26b-a4b Q4_K_M | ~17 GB | turbo4/turbo4 | ~9.4 GB | **~26 GB** | **Yes** | **262K (full)** |
| gemma-4-26b-a4b Q6_K | ~23 GB | turbo4/turbo4 | ~9.4 GB | **~32 GB** | **Barely** | **~230K** |
| gemma-4-31b-it Q4_K_M | ~20 GB | turbo3/turbo3 | ~57 GB | **~77 GB** | **No** | **~58K** |

For reference, Gemma Q6_K with f16 KV at 256K required ~41 GB and dropped from 150 to 55 tok/s due to RAM spill. With turbo4 KV, it fits to ~230K entirely in VRAM with no spill.

## What We Learned About Thinking Budget Exhaustion

The most significant finding across all runs: **turbo3 KV causes some models to enter infinite thinking loops**, burning their entire token budget on chain-of-thought reasoning and producing no code.

| Run | Config | Models Affected | Cause |
|---|---|---|---|
| turbo3, 32K ctx, 16K max | turbo3/turbo3 everywhere | Qwen 35B (0/17), Gemma Q6_K (partial) | KV quant noise disrupts "stop thinking" signal |
| turbo3, 32K ctx, 16K max, parallel=4 | turbo3/turbo3, parallel=4 | Same models + Qwopus degraded | FP accumulation order change compounds issue |
| turbo3, 96K ctx, 64K max | turbo3/turbo3, parallel=1 | Gemma Q6_K (65K tokens of thinking!) | More budget = more wasted thinking, not better code |
| **optimal settings** | **per-model KV + reasoning budget** | **None** (all produced code) | **Solved via turbo4 + budget cap** |

**The fix was not more tokens — it was better settings.** turbo4 instead of turbo3 for Gemma, q8_0 for Qwen, and reasoning budget caps as a safety net.

## Experimental Runs (Detailed)

### Run 1: turbo3 everywhere, 32K context, 16K max_tokens, parallel=auto

| Model | KV | Tok/s | ExprEval | A* | LRU | Total |
|---|---|---|---|---|---|---|
| Qwen 35B Q4_K_M | turbo3 | 172 | 0/5 | 0/6* | 0/6* | 0/17 |
| Gemma 26B Q6_K | turbo3 | 150 | 0/5* | 0/6 | 6/6 | 6/17 |
| Gemma 26B Q4_K_M | turbo3 | 164 | 4/5 | 3/6 | 0/6* | 7/17 |
| Gemma 31B Q4_K_M | turbo3 | 55 | 4/5 | 0/6 | 6/6 | 10/17 |
| Qwopus 27B Q6_K | turbo3 | 52 | 4/5 | 6/6 | 6/6 | 16/17 |

*Thinking budget exhaustion (finish_reason=length, content_chars=0)

### Run 2: turbo3 everywhere, 32K context, 16K max_tokens, parallel=4

| Model | KV | Tok/s | ExprEval | A* | LRU | Total |
|---|---|---|---|---|---|---|
| Qwen 35B Q4_K_M | turbo3 | 189 | 0/5 | 0/6* | 0/6* | 0/17 |
| Gemma 26B Q6_K | turbo3 | 150 | 0/5* | 7/7 | 0/6* | 7/17 |
| Gemma 26B Q4_K_M | turbo3 | 161 | 0/5* | 6/6 | 0/6* | 6/17 |
| Gemma 31B Q4_K_M | turbo3 | 54 | 4/5 | 0/6 | 0/6* | 4/17 |
| Qwopus 27B Q6_K | turbo3 | 52 | 0/5* | 0/6* | 6/6 | 6/17 |

parallel=4 improved A* scores for Gemma but caused MORE thinking exhaustion elsewhere.

### Run 3: turbo3, 96K context, 64K max_tokens, parallel=1

| Model | KV | Ctx | VRAM | Tok/s | ExprEval | A* | LRU | Total |
|---|---|---|---|---|---|---|---|---|
| Qwen 35B | turbo3 | 96K | 31,466 MB | 22-125 | 4/5 | 1/6 | 0/6* | 5/17 |
| Gemma 26B Q6_K | turbo3 | 96K | 31,813 MB | 80-110 | 0/5* | 6/6 | 0/6* | 6/17 |
| Gemma 26B Q4_K_M | turbo3 | 96K | 22,691 MB | 19-31 | 4/5 | 0/6 | ERR | 4/11 |
| Gemma 31B | turbo3 | 48K | 31,620 MB | 40-47 | 3/5 | 6/6 | 4/6 | 13/17 |
| Qwopus 27B | turbo3 | 96K | 31,574 MB | 14-50 | ERR | 6/6 | 6/6 | 12/12 |

96K context caused severe VRAM pressure and throughput degradation. Gemma Q6_K generated 65,536 tokens of pure thinking on ExprEval (10 min of thinking, zero code).

### Run 4: Optimal per-model settings (RECOMMENDED)

See "Optimal Settings" table above. This is the configuration to use going forward.

## Expanded Benchmark Suite (temp 0.3, 3 runs, 6 benchmarks)

### New Benchmarks

In addition to the original 3 (Expression Evaluator, A* Pathfinding, LRU Cache), we added:

- **String Processor** (Easy, 5 tests): reverse_words, count_vowels, is_palindrome, caesar_cipher, most_common_word. Establishes a quality floor.
- **Expression Evaluator Implementation-Only** (Medium, 5 tests): Same prompt but model writes only the implementation; tests are provided by the harness. Isolates code quality from test-writing.
- **Debug BST Fix** (Medium, 4 tests): Given a BST with 3 bugs, fix all bugs. Tests code reading and targeted repair.

### Temperature 0.3 Methodology

Temperature 0 is deterministic but implementation-dependent — FP accumulation order in the inference engine changes the output. This caused Qwen 35B to score 19/19 on LM Studio but 6/17 on llama-server with the same model. Temperature 0.3 with 3 runs per benchmark:

- Introduces enough sampling diversity to reduce infrastructure coupling
- Reveals model consistency (average score) vs peak capability (best-of-3)
- Separates genuine capability gaps from sampling luck

### Results: Consistency Is the Differentiator

| Model | Quant | Best-of-3 | Average | Gap | Most Consistent Benchmark |
|---|---|---|---|---|---|
| gemma-4-31b-it | Q4_K_M | 31/31 (100%) | 30.7/31 (99%) | 0.3 | All benchmarks near-perfect |
| gemma-4-26b-a4b | Q6_K | 30/31 (97%) | 30.0/31 (97%) | 0 | ExprEval: 4/5 all 3 runs |
| harmonic-27b | Q8_0 | 31/31 (100%) | 28.7/31 (93%) | 2.3 | A* 6/6 every run, Debug 4/4 every run |
| harmonic-27b | Q4_K_M | 31/31 (100%) | 25.7/31 (83%) | 5.3 | String Processor: 5/5 all 3 runs |
| qwen3.5-27b-opus | Q4_K_M | 31/31 (100%) | 25.3/31 (82%) | 5.7 | A* always 6-8, Debug 4/4 every run |
| qwopus-3.5-27b-v3 | Q6_K | 30/31 (97%) | 24.0/31 (77%) | 6 | LRU: 5/6, 2/6, 0/6 — most variable |
| qwen3.5-35b-a3b | Q4_K_M | 25/31 (81%) | 24.3/31 (78%) | 0.7 | LRU: 0/6 all 3 runs (genuine gap) |

**Gemma models are the most consistent.** Both 26B and 31B have best-of-3 nearly equal to their average. Qwen-family models (Qwopus, Opus-distilled, Harmonic) show higher variance — their best-of-3 scores are generous.

### Universal Passes

Every model, every run, 100%:
- **String Processor** (easy): Confirms basic coding ability
- **Debug BST Fix**: All models reliably identify and fix bugs

### Harmonic 27B (New)

Fine-tune of Qwen 3.5 27B focused on structured reasoning (self-correction, verification, multi-path exploration). Trained on only 799 curated rows.

- Q8_0 (27 GB): 15/17 single-shot, 31/31 best-of-3, A* 6/6 every run
- Q4_K_M (16 GB): 15/17 single-shot, 31/31 best-of-3, same quality at half the VRAM
- Consistent on A* and Debug. Variable on ExprEval and LRU at temp 0.3.

## Settings Reference

### Build

```bash
cd T:/git/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

### Server Startup (per model)

```bash
# Gemma 26B Q6_K (S-tier)
llama-server -m gemma-4-26B-A4B-it-Q6_K.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo4 -ctv turbo4 --reasoning-budget 12288

# Qwopus 27B Q6_K (A-tier)
llama-server -m Qwopus3.5-27B-v3-Q6_K.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo3 -ctv turbo3

# Qwen 35B Q4_K_M (fast, less reliable)
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk q8_0 -ctv q8_0
```

### Key Flags

| Flag | Purpose | Notes |
|---|---|---|
| `-ctk TYPE` | KV cache K quantization | turbo2/turbo3/turbo4/q8_0/q4_0/f16 |
| `-ctv TYPE` | KV cache V quantization | Same options as K |
| `--reasoning-budget N` | Cap thinking tokens | -1=unlimited, 0=no thinking, N=max thinking tokens |
| `-fa on` | Flash attention | **Required** for turbo KV — without it, turbo is slower than f16 |
| `-np N` | Parallel slots | Divides context per slot. Use 1 for max context per request |

### Asymmetric K/V Note

Asymmetric configs (e.g., `-ctk q8_0 -ctv turbo3`) would be ideal but **cannot be GPU-offloaded** in the current llama.cpp build — they force CPU processing and drop throughput to ~30 tok/s. This is tracked in [ggml-org/llama.cpp#20866](https://github.com/ggml-org/llama.cpp/issues/20866). Until fixed, use symmetric pairs only.

### TQ3_4S Weight Quantization

The Qwopus TQ3_4S GGUF model failed to load — it uses GGML tensor type 46, which is not defined in the `feature/turboquant-kv-cache` branch (max supported: type 45 = TQ4_1S). A newer branch or fork version is needed.

## Cross-Engine Comparisons

### NVFP4-turbo (vLLM) vs Q4_K_M + turbo4 (llama.cpp) — Gemma 31B

We tested LilaRest's `gemma-4-31B-it-NVFP4-turbo` against our standard llama.cpp pipeline. NVFP4 is NVIDIA's 4-bit floating-point format that uses Blackwell tensor cores via CUTLASS kernels.

**Setup difficulty (NVFP4 path):**
- Required vLLM 0.19.0+cu130 wheel (not standard pip install)
- Required CUDA 13.0 toolkit installed in WSL for flashinfer JIT compilation of SM 12.0 NVFP4 kernels
- Required `LD_LIBRARY_PATH` setup for libcudart.so.13
- Required `transformers>=5.5.0` despite vLLM pinning `transformers<5` (dependency conflict, ignored)
- Required `--quantization modelopt` flag to activate CUTLASS backend

**Results (3-benchmark suite, temp 0.3, 3 runs):**

| Config | Score | Tok/s | VRAM | Max Ctx | Notes |
|---|---|---|---|---|---|
| llama.cpp Q4_K_M + turbo4 | **17/17 (100%)** | **53** | **22.3 GB** | ~58K | Our standard pipeline |
| vLLM NVFP4-turbo + FP8 KV | 16/17 (94%) | 41 | 29.6 GB | ~16K | -1 ExprEval test |

**Verdict:** llama.cpp wins on quality, throughput, AND VRAM efficiency for our single-stream coding workload. NVFP4-turbo's published advantage (1,244 tok/s batched, 6.22 concurrent req/s) requires multi-user serving — not what we measure. The FP4 weight quantization noise lost one ExprEval test compared to Q4_K_M's mixed-precision approach.

### TriAttention (vLLM) vs TurboQuant (llama.cpp) — Qwen3-8B

Three-way comparison on the same model to isolate compression strategies:

| Approach | Engine | Compression | Best-of-3 | Avg |
|---|---|---|---|---|
| **TriAttention** | vLLM | Token eviction (4096 budget) | **14/17 (82%)** | 8.3/17 |
| f16 baseline | llama.cpp | None | 12/17 (71%) | 9.1/17 |
| turbo4 | llama.cpp | KV quantization (3.8x) | 8/17 (47%) | 5.7/17 |

**Key findings:**
1. **TriAttention wins on 8B models.** Token eviction (keeping 4096 of 16K+ tokens at full precision) preserves more quality than quantizing all tokens to 4 bits.
2. **turbo4 hurts 8B models more than 27B+ models.** On Gemma 26B/31B, turbo4 scores 17/17. On 8B, it drops to 8/17. Smaller models have less redundancy to absorb quantization noise.
3. **Compression strategy depends on model size.** TriAttention for small models, TurboQuant for large.

### Gemma E4B Capability Floor

Tested Gemma 4 E4B (2.3B active / 5.1B total params with PLE) to find our benchmark suite's lower bound:

| Config | ExprEval | A* | LRU | StrProc | Total |
|---|---|---|---|---|---|
| f16 KV | 0/5 | 0/6 | 0/6 | **5/5** | **5/22 (23%)** |
| turbo4 KV | 0/5 | 0/6 | 0/6 | **5/5** | **5/22 (23%)** |

E4B passes String Processor reliably (often generates 9-13 bonus tests) but completely fails Expression Evaluator, A* Pathfinding, and LRU Cache. **turbo4 is essentially free at this scale** — same 5/22 on both KV configs, only saves 418 MB of KV cache.

**Capability floor:** Need ~8B+ active parameters for medium tasks, ~27B+ for hard data structures.
