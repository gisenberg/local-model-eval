# Local Model Tier List — MacBook Pro M4 Max 36 GB

**Platform:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X, 410 GB/s
**Inference:** llama.cpp (turboquant fork, build 8590cbff9, Metal backend)
**Standard config:** flash attention (`-fa on`), `max_tokens=16384`, f16 KV (unless noted), thinking off (`--reasoning-budget 0`), `-np 1 --jinja`

Tested April 2026.

## The M4 Max's defining constraint: Metal working set, not unified memory

The Mac advertises 36 GB of unified memory but **the GPU can only address ~30 GB** of it (`recommendedMaxWorkingSetSize ≈ 30150 MB`). macOS reserves the rest for the system, page tables, the display, and other Metal clients. This is the hard ceiling for `weights + KV cache + compute buffers + scratch`. Going over it doesn't just slow down — it produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` and the request fails.

What this means in practice on this machine:

| Model class | Weights (Q4) | Weights (Q6) | Practical config |
|---|---|---|---|
| ≤8B dense | ~5 GB | ~7 GB | Anything fits, full 32K f16 KV, 60+ tok/s |
| 26B-A4B MoE | ~16 GB | ~22 GB | Q4_K_M at 32K f16 fits; Q6_K caps at 16K f16 |
| 27B dense | ~16 GB | — | Fits at 32K f16, ~13 tok/s (bandwidth bound) |
| 31B dense (gemma 4) | ~18 GB | — | **Requires turboquant turbo4 KV** — f16 at 16K = 14 GB KV, OOMs |

You can run the same model classes the 5090 ranking covers, but **at 1/4 to 1/3 the throughput** (410 GB/s vs 1,792 GB/s) and with much tighter context windows on dense large models. The trade is portability and ~30 W power draw vs the 5090's 575 W.

A second M4 Max-specific finding: **TurboQuant KV is *slower* than f16 KV here**. On the 5090, turbo4 buys speed because KV bandwidth is the bottleneck. On the Mac, weights bandwidth is already saturated, so the dequant compute overhead from turbo4 actively hurts. Use turbo4 only when you need it for capacity, not for speed.

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

### Gemma 4 26B-A4B Q6_K (turbo4 KV)
The all-around best quality + speed combination on this machine. MoE means only ~4B parameters are active per token, so throughput is much higher than the 31B dense. Quality matches the 5090's S-tier almost exactly (16/17 vs 17/17 — a 1-test variance on a single-shot run).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Throughput | **46.0 tok/s** |
| Weight size | 22.6 GB |
| Context | 16K |
| KV format | turbo4/turbo4 |
| Config | `-c 16384 -ctk turbo4 -ctv turbo4 --reasoning-budget 0 --jinja` |

**Strengths:** Strong quality across all 3 benchmarks (ExprEval 4/5, A* 6/6, LRU 6/6). MoE keeps throughput respectable.
**Weakness:** Capped at 16K context — 32K f16 OOMs even with weights at 22 GB; 32K turbo4 also OOMs because the bottleneck is the *weights*, not the KV cache. If you need >16K, drop to Q4_K_M.

---

## A-Tier: Real Tradeoffs

### Gemma 4 26B-A4B Q6_K (f16 KV)
The same model as above but with stock f16 KV. **Faster than turbo4 here** (60 vs 46 tok/s) — Apple Silicon's Mac-specific quirk: turbo4's dequant overhead exceeds the KV bandwidth savings on a bandwidth-constrained platform. Quality is essentially identical to turbo4 (15/17 vs 16/17, a single-test single-shot difference).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **15/17 (88%)** |
| Throughput | **60.3 tok/s** |
| Weight size | 22.6 GB |
| Context | 16K (32K f16 OOMs against 30 GB Metal budget) |
| KV format | f16/f16 |
| Config | `-c 16384 -ctk f16 -ctv f16 --reasoning-budget 0 --jinja` |

**The takeaway:** On M4 Max, prefer f16 KV unless you have a specific capacity reason to compress it. The 5090 ranking's "always use turbo4" advice does not transfer to Apple Silicon.

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

### Qwen 3.5 27B Opus-Distilled Q4_K_M
Dense 27B that fits cleanly at 32K f16 KV. The 5090 ranking has this at 100% — the M4 Max single-shot variance dropped one full benchmark (A* hit a syntax error: `except ImportError:` block without try). Worth re-running with best-of-3 to see if the gap closes.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **13.0 tok/s** |
| Weight size | 16.5 GB |
| Context | 32K |
| KV format | f16/f16 |

**Per-benchmark:** ExprEval 5/5, A* 0/6 (syntax error), LRU 6/6.
**Caveat:** Single-shot temp 0. The 5090 saw 100% on this exact GGUF; the gap is most likely sampling variance, not a hardware effect.

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
| Gemma 4 26B-A4B Q6_K @ 32K f16 | OOM (22 GB weights + ~12 GB KV > 30 GB) | Use 16K context, or switch to Q4_K_M |
| Gemma 4 26B-A4B Q6_K @ 32K turbo4 | OOM (turbo4 saves on KV but weights still dominate) | Same as above |
| Gemma 4 31B-IT Q4_K_M @ 16K f16 | OOM (18 GB weights + 14 GB f16 KV) | Use turbo4 KV (only option) |
| Anything > ~25 GB weight | Won't fit in 30 GB Metal budget at any context | Use a 5090 or Spark |

---

## Quick Reference: Choosing a Model

| Priority | Pick | Quant | KV | Context | Why |
|---|---|---|---|---|---|
| **Best quality (cost no object)** | Gemma 4 31B-IT | Q4_K_M | turbo4 | 16K | 17/17, only 11 tok/s |
| **Best quality + interactive speed** | Gemma 4 26B-A4B | Q6_K | f16 | 16K | 15-16/17, 60 tok/s |
| **Quality + 32K context** | Gemma 4 26B-A4B | Q4_K_M | f16 | 32K | 11/17, 59 tok/s, real Q4 quality cost |
| **Maximum throughput** | Nemotron 3 Nano 4B | Q4_K_M | f16 | 32K | 65 tok/s, but only 41% quality |
| **Lowest VRAM** | Nemotron 3 Nano 4B | Q4_K_M | f16 | 32K | 2.8 GB weights |

---

## Configuration

```bash
# Standard config — Gemma 26B-A4B Q6_K
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

## See also

- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — full hardware comparison with the 5090 and Spark
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — what these same models score on a 5090 with TurboQuant
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — same models on the bandwidth-bound DGX Spark
