# Model Rankings

Tested April 2026 on RTX 5090 32 GB. Models tested via LM Studio (f16 KV) and TurboQuant llama-server (turbo4 KV, `-rea off`). All models fully GPU-offloaded with flash attention, temperature 0 for single-shot and 0.3 for multi-run.

## Tier List (Single-Shot, temp 0, turbo4 KV, thinking off)

| Tier | Model | Quant | VRAM (32K) | Tok/s | Expr Eval (5) | A* Path (6) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **S** | gemma-4-26b-a4b | Q6_K | 25,636 MB | 142 | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Best overall. Fastest S-tier. |
| **S** | gemma-4-31b-it | Q4_K_M | 22,293 MB | 53 | 5/5 | 6/6 | 6/6 | **17/17 (100%)** | Dense arch. Requires TurboQuant to fit. |
| **S** | qwen3.5-27b-opus-distilled | Q4_K_M | 19,565 MB | 64 | 5/5 | 7/7 | 5/6 | **17/17 (100%)** | Lowest VRAM. Bonus A* tests. |
| **A** | gemma-4-26b-a4b | Q4_K_M | 20,046 MB | 156 | 5/5 | 6/6 | 5/6 | **16/17 (94%)** | Fastest model. One LRU edge case. |
| **A** | qwopus-3.5-27b-v3 | Q6_K | 24,549 MB | 52 | 4/5 | 6/6 | 6/6 | **16/17 (94%)** | Embeds reasoning in content. |
| **B+** | harmonic-27b | Q8_0 | 30,835 MB | 45 | 4/5 | 6/6 | 5/6 | **15/17 (88%)** | VRAM-heavy (31 GB). |
| **B+** | harmonic-27b | Q4_K_M | 19,995 MB | 66 | 4/5 | 6/6 | 5/6 | **15/17 (88%)** | Same score, half the VRAM. |
| **C** | qwen3.5-35b-a3b | Q4_K_M | 23,910 MB | 188 | 4/5 | 7/7 | 0/6 | **11/17 (65%)** | Fastest raw throughput. LRU always fails. |
| **C** | qwen3.5-27b | Q6_K | 24,606 MB | 51 | 4/5 | 6/6 | 0/6 | **10/17 (59%)** | Base model. LRU 0/6 — fine-tuning fixes this. |

## Expanded Suite (temp 0.3, best-of-3 / average, 6 benchmarks)

Adds String Processor (easy), Implementation-Only ExprEval (harness tests), and Debug BST Fix. 3 runs per benchmark at temperature 0.3 to measure consistency.

| Model | Quant | VRAM | Best-of-3 | Average | Consistency | Notes |
|---|---|---|---|---|---|---|
| **gemma-4-31b-it** | Q4_K_M | 22,565 MB | **31/31 (100%)** | **30.7/31 (99%)** | Very high | Perfect best, near-perfect avg. Most consistent S-tier. |
| **harmonic-27b** | Q8_0 | 30,819 MB | **31/31 (100%)** | **28.7/31 (93%)** | High | A* 6/6 every run. ExprEval slightly variable. |
| **harmonic-27b** | Q4_K_M | 19,936 MB | **31/31 (100%)** | **25.7/31 (83%)** | Moderate | Best-of-3 perfect but avg lower. LRU variable (3-6). |
| **qwen3.5-27b-opus** | Q4_K_M | 19,928 MB | **31/31 (100%)** | **25.3/31 (82%)** | Moderate | A* always passes (6-8). ImplOnly variable (1-5). |
| **gemma-4-26b-a4b** | Q6_K | 25,972 MB | **30/31 (97%)** | **30.0/31 (97%)** | Very high | ExprEval 4/5 all 3 runs. Most consistent overall. |
| **qwopus-3.5-27b-v3** | Q6_K | 24,875 MB | **30/31 (97%)** | **24.0/31 (77%)** | Low | LRU: 5/6, 2/6, 0/6. High variance on hard benchmarks. |
| **qwen3.5-35b-a3b** | Q4_K_M | 24,288 MB | **25/31 (81%)** | **24.3/31 (78%)** | High (on what it passes) | LRU: 0/6 all runs. Genuine capability gap. |

### What the Expanded Suite Reveals

**Consistency matters more than peak scores.** Gemma 26B Q6_K scores "only" 30/31 best-of-3, but its average (30.0) nearly matches — it produces the same quality every run. Qwopus scores 30/31 best-of-3 but averages 24.0 — its quality varies wildly between runs.

**New benchmarks differentiate the floor:**
- **String Processor** (easy): Every model passes 5/5, confirming basic coding ability across the board
- **Debug BST Fix**: 4/4 on every run for every model — all can reliably identify and fix bugs
- **Impl-Only ExprEval** (harness tests): Isolates implementation quality from test-writing. Most models score 5/5, confirming that self-test failures are usually about test-writing, not implementation quality

**Temperature 0 vs 0.3:** Single-shot temp 0 is implementation-dependent — FP accumulation order in the inference engine changes the output. Temp 0.3 with 3 runs is more robust and reveals true capability vs lucky sampling.

## LM Studio Baselines (f16 KV, temp 0)

| Tier | Model | Quant | Tok/s | TTFT | Expr Eval (5) | A* Path (6+) | LRU Cache (6) | Total |
|---|---|---|---|---|---|---|---|---|
| **S** | qwen3.5-35b-a3b | Q4_K_M | 92.9 | 2.39s | 5/5 | 8/8 | 6/6 | **19/19 (100%)** |
| **A** | gemma-4-26b-a4b | Q6_K | 161.9 | 2.43s | 5/5 | 6/6 | 5/6 | **16/17 (94%)** |
| **B** | qwen3.5-9b | Q8_0 | 113.2 | 2.20s | 5/5 | 6/7 | 4/6 | **15/18 (83%)** |
| **D** | nemotron-3-nano | Q4_K_M | 78.4 | 4.22s | 0/5 | 3/7 | 0/6 | **3/18 (17%)** |
| **F** | nemotron-3-nano-4b | Q8_0 | 209.8 | 2.65s | 0/5 | -- | -- | -- |

Note: Qwen 35B's S-tier score on LM Studio does not reproduce on llama-server due to FP accumulation order differences at temperature 0. At temp 0.3 with turbo4 KV, it scores 25/31 best-of-3 (81%) — LRU Cache is the persistent failure.

## Recommendations

### For code generation quality: Gemma 4 26B-A4B Q6_K
- 17/17 (100%) single-shot, 97% average across runs
- 142 tok/s — fast enough for interactive use
- Reaches ~230K context with turbo4 KV
- Most consistent model in the expanded suite

### For VRAM-constrained setups: Qwen 3.5 27B Opus-Distilled Q4_K_M
- 17/17 (100%) single-shot, 31/31 best-of-3
- Only 20 GB VRAM at 32K context — leaves 12 GB for context scaling
- 64 tok/s — moderate speed
- Full 262K native context fits easily

### For maximum context: Gemma 4 31B-IT Q4_K_M
- 17/17 (100%) single-shot, 31/31 (100%) best-of-3
- **Requires TurboQuant** — only fits 16K context without KV compression
- 53 tok/s, dense architecture
- Most consistent model in expanded suite (99% average)

### For raw speed: Qwen 3.5 35B-A3B Q4_K_M
- 188 tok/s — 3x faster than Gemma 31B
- 65% code quality (LRU consistently fails)
- Best for quick drafts where speed > correctness

## Full Context Window Analysis

| Model | Weights | KV Config | KV/Token | Max Usable Ctx | Full 262K? |
|---|---|---|---|---|---|
| qwen3.5-27b-opus Q4_K_M | 16 GB | turbo4/turbo4 | ~36 KB | **262K (full)** | Yes |
| gemma-4-26b-a4b Q4_K_M | 16 GB | turbo4/turbo4 | ~36 KB | **262K (full)** | Yes |
| qwen3.5-35b-a3b Q4_K_M | 20 GB | turbo4/turbo4 | ~8 KB | **262K (full)** | Yes |
| harmonic-27b Q4_K_M | 16 GB | turbo4/turbo4 | ~36 KB | **262K (full)** | Yes |
| qwopus-3.5-27b-v3 Q6_K | 22 GB | turbo4/turbo4 | ~36 KB | **262K (full)** | Yes |
| gemma-4-26b-a4b Q6_K | 22 GB | turbo4/turbo4 | ~36 KB | **~230K** | Nearly |
| harmonic-27b Q8_0 | 27 GB | turbo4/turbo4 | ~36 KB | **~60K** | No |
| gemma-4-31b-it Q4_K_M | 18 GB | turbo4/turbo4 | ~260 KB | **~58K** | No |

## Quantization Impact

| Weight Quant | Behavior Under TurboQuant KV |
|---|---|
| **Q6_K** | Consistently outperforms Q4_K_M. Higher weight precision absorbs KV quantization noise. |
| **Q8_0** | Best quality but often doesn't fit. Harmonic Q8_0 = Q4_K_M scores at 2x the VRAM. |
| **Q4_K_M** | Documented risky combination with turbo KV (turboquant_plus). Model-dependent — Gemma 31B handles it, Qwen 35B doesn't. |

## TurboQuant Configuration

### Recommended Server Startup

```bash
# General-purpose (best config for most models)
llama-server -m model.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo4 -ctv turbo4 -np 1 -rea off
```

### Key Flags

| Flag | Purpose | Notes |
|---|---|---|
| `-ctk turbo4 -ctv turbo4` | KV cache quantization | 3.8x compression, +0.23% PPL |
| `-rea off` | Disable thinking | Prevents reasoning loops, 5-10x faster |
| `-fa on` | Flash attention | **Required** for turbo KV types |
| `-np 1` | Single parallel slot | Maximizes context per request |
| `--reasoning-budget N` | Alternative: cap thinking tokens | Use if you need thinking but want a safety net |

See [TURBO3_RESULTS.md](TURBO3_RESULTS.md) for full experimental data across all runs.
See [TURBOQUANT_IMPACT.md](TURBOQUANT_IMPACT.md) for what TurboQuant unlocks on 32 GB VRAM.
