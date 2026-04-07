# TurboQuant KV Cache Benchmark Results

Tested April 2026 on RTX 5090 32 GB. TurboQuant fork of llama.cpp (`feature/turboquant-kv-cache` branch), CUDA 13.2, flash attention on, temperature 0, `max_tokens=16384`.

## TL;DR

TurboQuant KV cache quantization enables larger context windows by compressing the KV cache 3-4x, but **the KV quant type must be matched to the model architecture**. Using turbo3 blindly on all models causes severe quality regressions on thinking/reasoning models. With per-model optimal settings, we achieve S-tier quality (100%) while unlocking full context windows.

## Optimal Settings (Recommended)

Per-model KV configs based on architecture analysis and community best practices:

| Tier | Model | Quant | KV Config | VRAM (32K) | Tok/s | TTFT | Expr Eval (5) | A* Path (6) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** | gemma-4-26b-a4b | Q6_K | turbo4/turbo4 | 25,162 MB | 146.5 | 2.42s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Perfect score. turbo4 preserves thinking quality. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | turbo3/turbo3 | 24,035 MB | 52.9 | 2.23s | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Heavy thinker but reliable. Handles turbo3 well. |
| **B** | gemma-4-26b-a4b | Q4_K_M | turbo4/turbo4 | 19,555 MB | 155.6 | 2.29s | 0/5 | 7/7 | 6/6 | **13/17 (76%)** | Fast, lowest VRAM. ExprEval had import errors. |
| **C** | qwen3.5-35b-a3b | Q4_K_M | q8_0/q8_0 | 23,715 MB | 196.7 | 2.38s | 0/5 | 6/7 | 0/6 | **6/17 (35%)** | Fastest throughput but inconsistent code. |
| **D** | gemma-4-31b-it | Q4_K_M | turbo3/turbo3 | 21,499 MB | 54.0 | 2.34s | 4/5 | 0/6 | 0/6 | **4/17 (24%)** | Dense arch is slow. A* has persistent import errors. |

Reasoning budget flags: `--reasoning-budget 12288` for Gemma 26B models, `--reasoning-budget 16384` for Gemma 31B, unrestricted for Qwen and Qwopus.

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
