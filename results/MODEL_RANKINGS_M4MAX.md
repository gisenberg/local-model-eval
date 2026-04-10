# Local Model Tier List — MacBook Pro M4 Max 36 GB

**Platform:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X, 410 GB/s
**Inference:** llama.cpp (turboquant fork, build 8590cbff9, Metal backend)
**Standard config:** flash attention (`-fa on`), `max_tokens=16384`, f16 KV (unless noted), thinking off (`--reasoning-budget 0`), `-np 1 --jinja`

Tested April 2026.

## The M4 Max's defining constraint: compute buffer, not weights or KV cache

Three things conspire to limit context length on this machine far below the 5090, despite both having ~30 GB of usable GPU memory:

1. **Metal working set is ~30 GB**, not the full 36 GB. `recommendedMaxWorkingSetSize ≈ 30150 MB`. macOS reserves the rest. Going over produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` immediately, no graceful spillover.

2. **Metal's compute buffer is ~4-5 GB bigger than CUDA's** for the same model + context. This is the surprise. We measured the breakdown for Gemma 4 26B-A4B Q6_K turbo4 at 32K context:
   - Weights: 21,574 MiB
   - KV cache (turbo4): 165 MiB ← negligible
   - **Compute buffer: 8,402 MiB ← the dominant cost**
   - Total: 30,141 MiB → over the 28,753 MiB free budget → OOM

   The compute buffer scales linearly with context at **~260 MiB per 1024 tokens** for this architecture. At 16K it's 4.2 GB, at 32K it's 8.4 GB. The 5090 ranking shows the same model using 25,636 MiB *total* at 32K — which means CUDA's compute buffer for Gemma 4 26B-A4B is roughly **half** the size of Metal's.

3. **No PCIe spill.** When the 5090 hits its 32 GB limit, it spills to system RAM via PCIe (slow but functional — see [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md) for the cliff at 256K). The Mac has no PCIe to spill across; either it fits or it OOMs.

### Practical context limits, by model

These are measured, not calculated:

| Model | Quant | KV | Max ctx that fits | Why |
|---|---|---|---|---|
| Gemma 4 26B-A4B | Q6_K | f16 or turbo4 | **~20K** | Compute buffer at 24K already exceeds the Metal allocator margin |
| Gemma 4 26B-A4B | Q4_K_M | f16 | **32K** | Weights drop from 22 → 16 GB, leaves 6 GB more for compute |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | **16K** | Same issue, but smaller weight headroom |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | f16 | **32K** | Smaller compute buffer (different architecture) |
| Qwen 3.5 9B / Nemotron 4B | Q4_K_M | f16 | 32K | Plenty of headroom |

The 5090 ranking shows Gemma 4 26B-A4B Q6_K turbo4 at "~230K" context. **That number is not achievable on M4 Max at any KV format** because the compute buffer alone would be ~60 GB at 230K context.

Two follow-on findings:

- **TurboQuant KV is *slower* than f16 KV here.** On the 5090, turbo4 buys speed because KV bandwidth is the bottleneck. On the Mac, weights bandwidth is already saturated AND the KV cache is a tiny fraction of the working set anyway, so the dequant compute overhead from turbo4 actively hurts. Use turbo4 only when you need it for capacity (Gemma 4 31B-IT specifically), not for speed.
- **You can run the same model classes the 5090 ranking covers**, but **at 1/4 to 1/3 the throughput** (410 GB/s vs 1,792 GB/s) and with much tighter context windows on dense large models. The trade is portability and ~30 W power draw vs the 5090's 575 W.

---

## S-Tier: The Best Quality That Fits

### Gemma 4 31B-IT Q4_K_M (turbo4 KV)
The highest-quality model on this machine. Perfect single-shot score, but you pay for it in throughput — 11 tok/s is borderline interactive. **Turbo4 KV is mandatory**: dense 31B at f16 KV = ~870 KB/token, and 16K of that is 14 GB on top of 18 GB of weights. The Metal budget can't take it.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Throughput | **11.5 tok/s** |
| Weight size | 18.3 GB (Q4_K_M, unsloth) |
| Context (turbo4) | 16K |
| KV format | turbo4/turbo4 (mandatory) |
| Config | `-c 16384 -ctk turbo4 -ctv turbo4 --reasoning-budget 0 --jinja` |

**Strengths:** Same quality you'd get on a 5090 (also 17/17). Per-benchmark perfect: ExprEval 5/5, A* 6/6, LRU 6/6.
**Weakness:** Throughput. 11 tok/s means a 2K-token generation takes 3 minutes. Acceptable for "give me the right answer once" workflows; painful for chat.
**Why not Q6_K?** 31B Q6_K is ~25 GB which leaves no room for KV. Untested but unlikely to load.

---

### Gemma 4 26B-A4B Q6_K (f16 KV)
The all-around best quality + speed combination on this machine. MoE means only ~4B parameters are active per token, so throughput is much higher than the 31B dense. Quality matches the 5090's S-tier within a single-test single-shot variance (15-16/17 vs 17/17).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **15/17 (88%)** |
| Throughput | **60.3 tok/s** |
| Weight size | 22.6 GB |
| Context | 16K |
| KV format | f16/f16 |
| Config | `-c 16384 -ctk f16 -ctv f16 --reasoning-budget 0 --jinja` |

**Strengths:** Fastest model with high quality. Per-benchmark: ExprEval 3/5, A* 6/6, LRU 6/6.
**Weakness:** Capped at ~20K context. The compute buffer alone is 4.2 GB at 16K and grows to 8.4 GB at 32K — not the KV cache (which is only ~85 MiB at 16K with f16 because Gemma 4 uses sliding-window attention). If you need >20K, drop to Q4_K_M which has 6 GB less weight footprint.
**Same model with turbo4 KV** scored 16/17 at 46 tok/s — *slower* for ~1 test of additional quality. Turbo4 doesn't even buy you context here because the compute buffer dominates, not the KV. On Apple Silicon, prefer f16 KV. The 5090 ranking's "always use turbo4" advice does not transfer.

---

## A-Tier: Best Dense Speed (with caveats)

### Qwen 3.5 27B Opus-Distilled MLX 4bit (mlx_lm)
The fastest *dense* 27B+ model on this machine — and one of only two cases where MLX outperforms llama.cpp on this hardware. **42% faster** than the same model under llama.cpp Q4_K_M (18.5 vs 13.0 tok/s) at ~99% bandwidth utilization. Run via `mlx_lm.server`, not llama.cpp.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **13/17 (76%)** |
| Throughput | **18.5 tok/s** |
| Weight size | ~14 GB (MLX 4-bit) |
| Context | 32K (mlx_lm default) |
| Config | `mlx_lm.server --model mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --port 8766 --temp 0` |

**Per-benchmark:** ExprEval 5/5 (28 tests written, 24 pass — over-engineered), A* 6/6, LRU 2/6.
**Strengths:** Best speed/quality combo for dense 27B+ on this hardware. Beats llama.cpp on the same model in both speed AND code quality (single-shot).
**Weakness:** mlx_lm.server takes 5-15 min to load, vs llama-server's seconds. Slow iteration. LRU cache fell apart at 2/6.

---

### Qwen 3.5 27B Opus-Distilled Q4_K_M (llama.cpp)
The same model under llama.cpp's metal backend. Slower than MLX but easier to set up.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **13.0 tok/s** |
| Weight size | 16.5 GB |
| Context | 32K |
| KV format | f16/f16 |

**Per-benchmark:** ExprEval 5/5, A* 0/6 (model emitted `except ImportError:` outside try block — real syntax error), LRU 6/6.
**Caveat:** Single-shot temp 0 variance. The 5090 saw 100% on this exact GGUF. Use the MLX version above unless you specifically need llama.cpp.

---

### Gemma 4 26B-A4B Q4_K_M (f16 KV, full 32K context)
The "fits everywhere" version. ~16 GB weights leaves enough room for 32K f16 KV. Quality drops noticeably from Q6_K — including a literal incomplete-line truncation in the LRU cache impl (`self.tail.prev = self.` and then EOF). The Q4_K_M quality risk that the 5090 ranking warns about is real and apparently amplified by the Mac inference path.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **58.9 tok/s** |
| Weight size | 16.5 GB |
| Context | 32K |
| KV format | f16/f16 |

**Per-benchmark:** ExprEval 5/5, A* 6/6, LRU 0/6 (truncated impl).
**Bottom line:** Take Q6_K if you don't need >16K context. Take Q4_K_M only if you specifically need long context on this model.

---

### Gemma 4 26B-A4B Q4_K_M (f16 KV, full 32K context)
The "fits everywhere" version. ~16 GB weights leaves enough room for 32K f16 KV. Quality drops noticeably from Q6_K — including a literal incomplete-line truncation in the LRU cache impl (`self.tail.prev = self.` and then EOF). The Q4_K_M quality risk that the 5090 ranking warns about is real and apparently amplified by the Mac inference path.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **58.9 tok/s** |
| Weight size | 16.5 GB |
| Context | 32K |
| KV format | f16/f16 |

**Per-benchmark:** ExprEval 5/5, A* 6/6, LRU 0/6 (truncated impl).
**Bottom line:** Take Q6_K if you don't need >16K context. Take Q4_K_M only if you specifically need long context on this model.

---

## B-Tier: Small + Fast

### Qwen 3.5 9B Q4_K_M (thinking off)
The fastest dense model that produces usable code on this machine. With thinking off, scores are roughly half the 5090's Q8_0 baseline (15/18 → 9/17) — most of the gap is the quant difference (Q4_K_M vs Q8_0), some is single-shot variance.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **9/17 (53%)** |
| Throughput | **35.4 tok/s** (thinking off) |
| Weight size | 5.5 GB |
| Context | 32K |
| Config | `-c 32768 -ctk f16 -ctv f16 --reasoning-budget 0` |

**Thinking on/off:** Thinking ON drops this to 3/17 (18%). Same model, same prompts, same temp — thinking burns tokens then runs out of room for content. Always disable for this model on this hardware.

---

### Nemotron 3 Nano 4B Q4_K_M (thinking on)
The fastest model that produces usable output. 4B parameters, ~58-65 tok/s. **Thinking ON is meaningfully better here**: 7/17 with thinking, 4/17 without — opposite polarity from Qwen 9B.

| Metric | Value |
|---|---|
| Single-shot (temp 0, thinking on) | **7/17 (41%)** |
| Single-shot (temp 0, thinking off) | 4/17 (24%) |
| Throughput | **65.5 tok/s** (thinking on), 58.1 tok/s (thinking off) |
| Weight size | 2.8 GB |
| Context | 32K |

**Why use it:** Lowest VRAM, fastest, runs on battery. If you need *some* model and the rest are too big or too slow, this is the floor.
**Why not:** Hard benchmarks (LRU) are mostly out of reach. Use it for short-form completions, not full implementations.

---

## TurboQuant on M4 Max: When It Helps and When It Doesn't

The 5090 ranking's headline advice is "always use turbo4 KV — same quality, less VRAM, sometimes faster." That advice **does not transfer** to Apple Silicon. The data:

| Model + Config | Tok/s | Score | KV bytes/token (rough) |
|---|---|---|---|
| Gemma 26B-A4B Q6_K f16 16K   | **60.3** | 15/17 | ~190 KB |
| Gemma 26B-A4B Q6_K turbo4 16K | 46.0 | 16/17 | ~50 KB |
| Gemma 31B Q4_K_M turbo4 16K   | 11.5 | 17/17 | ~220 KB |
| Gemma 31B Q4_K_M f16 16K      | — | — | OOM (won't load) |

**The pattern:**
- For models where f16 KV fits → f16 is faster than turbo4. The dequant compute outweighs the KV bandwidth savings on a bandwidth-constrained platform.
- For models where f16 KV won't fit → turbo4 is the only option, regardless of speed cost. Gemma 31B at f16 needs ~14 GB of KV at 16K context, which combined with 18 GB of weights overflows the 30 GB Metal budget.

**Rule of thumb:** Try f16 first. Only switch to turbo4 if the model doesn't load.

---

## MLX vs llama.cpp on Apple Silicon

The HARDWARE_SPECS.md footnote claimed "MLX sometimes outperforms llama.cpp on Apple Silicon by 10–30%." We tested this directly using `mlx_lm.server` 0.31.2 against `mlx-community/*` quants, same prompts, temp 0.

**The answer is "it depends on the model class."** MLX wins on dense ≥27B models, loses on MoE models, and ties on small dense.

| Model | llama.cpp | MLX | Winner |
|---|---|---|---|
| Qwen 3.5 9B (Q4_K_M / 4bit, dense) | 35.4 tok/s, 9/17 | 33.9 tok/s, 10/17 | tie (~4% MLX slower) |
| Gemma 4 26B-A4B (Q6_K f16 / 6bit, **MoE 4B-active**) | **60.3 tok/s**, 15/17 | 33.5–37.4 tok/s, 12/17 | **llama.cpp** (-38% on MLX) |
| Gemma 4 31B-IT (Q4_K_M turbo4 / 4bit, dense) | 11.5 tok/s, 17/17 | 7.9–11.9 tok/s, 17/17 | tie speed, same quality (17/17) |
| Qwen 3.5 27B Opus-Distilled (Q4_K_M / 4bit, **dense**) | 13.0 tok/s, 11/17 | **18.5 tok/s**, 13/17 | **MLX** (+42% speed, +12% quality) |
| Nemotron 3 Nano 4B (Q4_K_M / 6bit) | 65.5 tok/s, 7/17 | **N/A** | mlx_lm 0.31.2 hits transformers `KeyError: '-'` parsing nemotron_h layer pattern |

### The pattern: MoE active params hurt MLX, dense weights help it

Bandwidth utilization (rough) = `(weights+KV bytes per token) × tok/s ÷ peak bandwidth`. Plugging in real numbers:

- **Qwen 27B dense Q4** at 13 tok/s (llama.cpp) ≈ 70% of 410 GB/s peak. Same model at 18.5 tok/s (MLX) ≈ 99% of peak. **MLX is leaving almost nothing on the table here.**
- **Gemma 26B-A4B Q6** (only ~4 GB read per token from MoE) at 60 tok/s (llama.cpp) ≈ 88% of peak. Same model at 33 tok/s (MLX) ≈ 49%. **MLX's MoE kernels under-perform — most likely it's not exploiting the sparse activation pattern as efficiently.**

Plus:
- **Gemma 4 31B-IT 4bit MLX matched llama.cpp on quality (17/17)** — the same 5/5 / 6/6 / 6/6 breakdown. So the 4-bit MLX quantization doesn't lose accuracy on this model.
- **Gemma 4 26B-A4B 6bit MLX had a chat-template regression**: ExprEval went runaway (16K tokens of misclassified `reasoning_content` in the output before any actual code). The mlx_lm.server handling of Gemma 4's reasoning fields is broken; this contributed to the speed loss above and the 0/5 ExprEval score.

### Recommendations

- **Dense ≥27B model on Mac → consider MLX first.** Significant speedup possible.
- **MoE model on Mac → use llama.cpp.** MLX kernels lose 30-40% on this class.
- **Gemma 4 specifically on MLX → wait for the chat-template fix.** Outputs are sometimes garbage due to reasoning misclassification.
- **Gemma 4 31B-IT → either works.** Same quality, similar speed.

---

## Thinking ON vs OFF

| Model | Thinking ON | Thinking OFF | Winner |
|---|---|---|---|
| Qwen 3.5 9B Q4_K_M | 3/17 (18%) | **9/17 (53%)** | **OFF (+6)** — burns budget thinking |
| Nemotron 3 Nano 4B Q4_K_M | **7/17 (41%)** | 4/17 (24%) | **ON (+3)** — actually uses the reasoning |

Two models, two opposite outcomes — same lesson the 5090 ranking found: thinking on/off is **model-dependent**. Default off and flip on per model after testing.

---

## What Couldn't Run

| Model + Config | Why | Workaround |
|---|---|---|
| Gemma 4 26B-A4B Q6_K @ 24K+ (any KV) | Compute buffer >6 GB pushes total over 28.7 GB Metal free budget | Cap at 20K, or switch to Q4_K_M (6 GB less weight footprint) |
| Gemma 4 31B-IT Q4_K_M @ 16K f16 | 18 GB weights + 14 GB f16 KV cache > 30 GB ceiling | Use turbo4 KV (only option) |
| Anything > ~22 GB weight at non-trivial context | Compute buffer takes 4-8 GB on top of weights, leaving no room | Use a 5090 or Spark |
| Long-context (>32K) on any ≥26B model | Compute buffer scales ~260 MiB per 1024 tokens — by 64K it's 16 GB on top of weights | Same as above |

---

## Quick Reference: Choosing a Model

| Priority | Pick | Engine | Quant | KV | Context | Why |
|---|---|---|---|---|---|---|
| **Best quality (slow)** | Gemma 4 31B-IT | llama.cpp turboquant | Q4_K_M | turbo4 | 16K | 17/17, only 11.5 tok/s |
| **Best quality + interactive speed** | Gemma 4 26B-A4B | llama.cpp | Q6_K | f16 | 16K | 15/17, 60 tok/s |
| **Best dense 27B at speed** | Qwen 3.5 27B Opus-Distill | **MLX 4bit** | — | — | 32K | 13/17, 18.5 tok/s (42% > llama.cpp) |
| **Quality + 32K context** | Gemma 4 26B-A4B | llama.cpp | Q4_K_M | f16 | 32K | 11/17, 59 tok/s, real Q4 quality cost |
| **Maximum throughput** | Nemotron 3 Nano 4B | llama.cpp | Q4_K_M | f16 | 32K | 65 tok/s, but only 41% quality |
| **Lowest memory** | Nemotron 3 Nano 4B | llama.cpp | Q4_K_M | f16 | 32K | 2.8 GB weights |

---

## Configuration

### llama.cpp (turboquant fork) — default

```bash
# Standard config — Gemma 26B-A4B Q6_K, the recommended all-rounder
~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server \
  -m gemma-4-26B-A4B-it-Q6_K.gguf --port 8765 \
  -c 16384 -ngl 999 -fa on \
  -ctk f16 -ctv f16 \
  -np 1 --jinja --reasoning-budget 0

# Gemma 31B (turbo4 mandatory)
... -c 16384 -ctk turbo4 -ctv turbo4 ...

# Qwen 9B / Nemotron 4B (full 32K f16, no compression needed)
... -c 32768 -ctk f16 -ctv f16 ...
```

The Docker build at `~/.docker/bin/inference/llama-server` (build 3191462) is too old for Gemma 4 (`unknown model architecture: 'gemma4'`). Use the turboquant fork which is built from a newer base — it supports both f16 and turbo3/turbo4 KV cache types and works as a drop-in replacement when you don't need TurboQuant features.

### MLX (mlx_lm.server) — for dense ≥27B

```bash
# Qwen 3.5 27B Opus-Distilled MLX 4bit — fastest dense 27B+ on this machine
/tmp/mlx-venv/bin/python3 -m mlx_lm server \
  --model mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
  --port 8766 --temp 0 --max-tokens 16384 \
  --chat-template-args '{"enable_thinking": false}'
```

mlx-lm 0.31+ requires Python 3.10+ (system Python 3.9 only has wheels up to mlx 0.29.3 which lacks `qwen3_5.py` / `gemma4.py`). Use a brew Python venv:

```bash
brew install python@3.14
/opt/homebrew/bin/python3.14 -m venv /tmp/mlx-venv
/tmp/mlx-venv/bin/pip install mlx-lm
```

Note that mlx_lm.server takes 5-15 minutes to load a model on first invocation. The OpenAI-style endpoints work after that.

## See also

- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — full hardware comparison with the 5090 and Spark
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — what these same models score on a 5090 with TurboQuant
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — same models on the bandwidth-bound DGX Spark
