# Model Rankings

Tested April 2026 on RTX 5090 32GB. Models tested via both LM Studio (f16 KV) and TurboQuant llama-server (turbo3/turbo4 KV). All models fully GPU-offloaded with flash attention.

## Tier List

| Tier | Model | Quant | Backend | KV Config | Thinking | VRAM (32K) | Tok/s | TTFT | Expr Eval (5) | A* Path (6+) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** | gemma-4-26b-a4b | Q6_K | TurboQuant | turbo4/turbo4 | off | 25,636 MB | 142.2 | 2.32s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Best overall. Fast (12-15s/bench). |
| **S** | gemma-4-26b-a4b | Q6_K | TurboQuant | turbo4/turbo4 | on | 25,162 MB | 146.5 | 2.42s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Thinking on also perfect with reasoning budget. |
| **S** | gemma-4-31b-it | Q4_K_M | TurboQuant | turbo4/turbo4 | off | 22,293 MB | 53.3 | 2.20s | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Dense arch. Was D-tier with thinking on. |
| **S** | qwen3.5-35b-a3b | Q4_K_M | LM Studio | f16/f16 | on | -- | 92.9 | 2.39s | 5/5 | 8/8 | 6/6 | **19/19 (100%)** | LM Studio only. Regresses on llama-server. |
| **A** | gemma-4-26b-a4b | Q4_K_M | TurboQuant | turbo4/turbo4 | off | 20,046 MB | 156.1 | 2.31s | 5/5 | 6/6 | 5/6 | **16/17 (94%)** | Lowest VRAM (20 GB). One LRU edge case. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | TurboQuant | turbo4/turbo4 | off | 24,549 MB | 51.7 | 2.38s | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Embeds reasoning in content even with -rea off. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | TurboQuant | turbo3/turbo3 | on | 24,035 MB | 52.9 | 2.23s | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Heavy thinker. Reliable on turbo3. |
| **A** | gemma-4-26b-a4b | Q6_K | LM Studio | f16/f16 | on | -- | 161.9 | 2.43s | 5/5 | 6/6 | 5/6 | **16/17 (94%)** | Best speed/quality on LM Studio. |
| **B** | qwen3.5-9b | Q8_0 | LM Studio | f16/f16 | on | -- | 113.2 | 2.20s | 5/5 | 6/7 | 4/6 | **15/18 (83%)** | Good for light tasks. LM Studio only. |
| **C** | qwen3.5-35b-a3b | Q4_K_M | TurboQuant | turbo4/turbo4 | off | 23,910 MB | 188.1 | 2.31s | 4/5 | 7/7 | 0/6 | **11/17 (65%)** | Fastest. LRU consistently fails (Q4_K_M sensitivity). |
| **D** | nemotron-3-nano | Q4_K_M | LM Studio | f16/f16 | on | -- | 78.4 | 4.22s | 0/5 | 3/7 | 0/6 | **3/18 (17%)** | Broken test files. |
| **F** | nemotron-3-nano-4b | Q8_0 | LM Studio | f16/f16 | on | -- | 209.8 | 2.65s | 0/5 | -- | -- | -- | Non-functional code. |

## What Changed with TurboQuant

### Gemma Q6_K went from A-tier to S-tier

The original LM Studio evaluation scored Gemma 26B Q6_K at 16/17 — one LRU cache test failed due to a subtle lazy-cleanup bug in `size()`. With TurboQuant's llama-server using turbo4 KV + `--reasoning-budget 12288`, the same model scored **17/17 (100%)**. The reasoning budget cap forced the model to transition from thinking to code output more decisively, which paradoxically produced better code.

### Qwen 35B regressed on llama-server

Qwen 35B was the undisputed champion on LM Studio (19/19, 100%). On llama-server — regardless of KV config — it scored 6/17 at best. This is NOT a KV quantization issue: even with q8_0 KV (essentially lossless), the output differs due to FP accumulation order differences between LM Studio's backend and raw llama-server (different parallel slot defaults, batch scheduling, etc.). The model is highly sensitive to these infrastructure-level differences at temperature=0.

### New models entered the rankings

- **Qwopus 3.5 27B-v3 Q6_K** (A-tier, 94%): A fine-tune of Qwen 3.5 27B trained on Opus-distilled reasoning data. Dense but hybrid DeltaNet architecture (16/64 attention layers). Slower than Gemma but highly reliable — only model to score 6/6 on both A* and LRU on turbo3 KV.
- **Gemma 4 31B-IT Q4_K_M** (C-tier, 24%): Dense 30.7B model, NOT MoE. 7x more expensive KV cache per token than the 26B-A4B variant. Needs turbo3 to fit even 32K context, struggles with A* pathfinding.

## Full Context Window Analysis

With optimal TurboQuant KV configs on the RTX 5090 (32 GB):

| Model | KV Config | KV/Token | Max Usable Ctx | Full 262K? |
|---|---|---|---|---|
| qwen3.5-35b-a3b Q4_K_M | q8_0/q8_0 | ~10 KB | **262K** | Yes |
| qwopus-3.5-27b-v3 Q6_K | turbo3/turbo3 | ~16 KB | **262K** | Yes |
| gemma-4-26b-a4b Q4_K_M | turbo4/turbo4 | ~36 KB | **262K** | Yes |
| gemma-4-26b-a4b Q6_K | turbo4/turbo4 | ~36 KB | **~230K** | Nearly |
| gemma-4-31b-it Q4_K_M | turbo3/turbo3 | ~217 KB | **~58K** | No |

Previously, Gemma 26B Q6_K at 256K context with f16 KV dropped from 150 to 55 tok/s due to RAM spill. With turbo4 KV, it fits to ~230K entirely in VRAM.

## Benchmark Details

### Expression Evaluator (Medium)
Recursive descent parser for `+`, `-`, `*`, `/` with operator precedence, parentheses, unary minus, floats, and error handling. 5 pytest tests.

- **S/A tier**: All produced correct implementations. Differences were in test assertions.
- **D/F tier**: Nemotron 30B had a test import bug (`from module import ValueError`). Nemotron 4B had a broken regex (`(?P<plus>+)` — unescaped quantifier) plus undefined variables.

### A* Pathfinding (Medium-Hard)
Weighted 2D grid pathfinding with walls, Manhattan distance heuristic, heapq open set. 6 pytest tests.

- **Qwen 35B (LM Studio)**: 8/8 (generated bonus tests, all passed)
- **Gemma Q6_K (TurboQuant turbo4)**: 6/6
- **Qwopus Q6_K (turbo3)**: 6/6
- **Gemma 31B (turbo3)**: 0/6 (persistent import errors in test file)

### LRU Cache with TTL (Hard)
O(1) LRU cache with doubly-linked list + hash map, time-based expiry via mocked `time.monotonic()`, lazy cleanup. 6 pytest tests.

- **Gemma Q6_K (TurboQuant turbo4)**: 6/6 — improved from 5/6 on LM Studio
- **Qwopus Q6_K (turbo3)**: 6/6
- **Qwen 35B (LM Studio)**: 6/6

## TurboQuant Configuration Guide

### Per-Model KV Config Selection

| Architecture | KV Savings with turbo3 | Recommendation |
|---|---|---|
| DeltaNet hybrid, few attn layers (Qwen 35B) | Minimal (~15 KB/token) | **q8_0/q8_0** — don't compress |
| DeltaNet hybrid, moderate attn (Qwopus 27B) | Moderate (~48 KB/token) | **turbo3/turbo3** — good tradeoff |
| Sliding window + global (Gemma 26B) | Significant (~90 KB/token) | **turbo4/turbo4** — safer for thinking |
| Dense transformer (Gemma 31B) | Essential (~650 KB/token) | **turbo3/turbo3** — must compress |

### Key Flags

- `-ctk TYPE -ctv TYPE`: KV cache quantization (turbo2/turbo3/turbo4/q8_0/q4_0/f16)
- `--reasoning-budget N`: Cap thinking tokens (-1=unlimited, N=max thinking tokens)
- `-fa on`: Flash attention — **required** for turbo KV types
- `-np 1`: Single parallel slot for max context per request

See [TURBO3_RESULTS.md](TURBO3_RESULTS.md) for full experimental data across 4 benchmark runs.

## Quantization Impact

Direct comparison of Gemma 4 26B at two quant levels, same prompt:

| Quant | Backend | KV Config | Tok/s | Expr Eval | Total |
|---|---|---|---|---|---|
| Q4_K_M | LM Studio | f16/f16 | 168.8 | 4/5 | -- |
| Q6_K | LM Studio | f16/f16 | 161.9 | 5/5 | 16/17 |
| Q4_K_M | TurboQuant | turbo4/turbo4 | 155.6 | 0/5 | 13/17 |
| Q6_K | TurboQuant | turbo4/turbo4 | 146.5 | 5/5 | **17/17** |

Q6_K consistently outperforms Q4_K_M on code quality. The 2 extra bits per weight are worth the 7% throughput cost.
