# Generalizing the M4 Max Findings to Other Apple Silicon

**This document is mostly projection, not measurement.** Only the M4 Max 36 GB MacBook Pro row in any throughput table is from real benchmarks (see [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md)). Everything about other Mac configurations is extrapolated from bandwidth math + the qualitative findings we measured. Where we make a projection, we mark it as one. Real measurements on these other machines would change the numbers.

The purpose of this doc is to answer: "I have M4 Max findings — how do they apply to a different Mac?"

## The Mac product matrix (April 2026)

Apple did **not** release an M4 Ultra. The 2025 Mac Studio refresh shipped with the **M4 Max** as the base SKU and the **M3 Ultra** (older generation, but with UltraFusion) as the high-end SKU. M5 Max / M5 Ultra are rumored for mid-2026.

| Chip | CPU/GPU | Memory bandwidth | Memory range | Generation |
|---|---|---|---|---|
| M4 Max (binned, MBP only) | 14C / 32C | **410 GB/s** | 36 GB | M4 (newer Metal) |
| M4 Max (full) | 16C / 40C | **546 GB/s** | 48 / 64 / 128 GB | M4 |
| **M3 Ultra** | 28C / 60C or 32C / 80C | **819 GB/s** | 96 / 256 / 512 GB | M3 (older Metal) |

Sources:
- [Mac Studio (2025) Tech Specs](https://support.apple.com/en-us/122211)
- [Apple's March 2025 Mac Studio launch](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/)

The 36 GB binned M4 Max we benchmarked is the *most* memory-constrained and *least* bandwidth-rich Apple Silicon you can buy in this product family. Everything else is more capable along at least one of the two axes.

## Findings that transfer identically

These are properties of llama.cpp's Metal backend, MLX kernels, and the LM Studio API surface — they don't depend on memory size or bandwidth:

1. **Metal working set ≈ 75% of unified memory** by default (`recommendedMaxWorkingSetSize`). `sudo sysctl iogpu.wired_limit_mb=...` can push it to ~92% if needed. The 75% ratio held on every Apple Silicon generation we know of.

2. **Compute buffer formula** for sliding-window architectures (Gemma 4):
   ```
   compute_buf_MiB ≈ 0.5 × n_ubatch × n_ctx / 1024
   ```
   This is kernel-level, not device-level. Should hold on any Apple Silicon Metal backend running llama.cpp. **Caveat:** verified on M4 Max only. M3 Ultra is one generation older and could have different MPS allocation patterns — needs measurement.

3. **TurboQuant turbo4 KV is slower than f16 KV** on Metal. The dequant compute overhead exceeds the bandwidth savings on bandwidth-constrained platforms — and even high-bandwidth Apple Silicon is bandwidth-constrained relative to its compute budget. The only reason to use turbo4 on any Mac is when the KV cache itself is the bottleneck (essentially just dense full-attention models like Gemma 4 31B-IT at long contexts).

4. **MLX vs llama.cpp is dense-favorable, MoE-unfavorable.** Class-dependent kernel finding. Should transfer to other Apple Silicon. The gap may widen at higher bandwidth budgets where MLX's near-100% bandwidth utilization on dense models matters more in absolute terms.

5. **LM Studio API silently ignores reasoning toggles.** Same `chat_template_kwargs` / `reasoning_effort` / `/no_think` issue regardless of hardware. Use `llama-server --reasoning-budget 0` directly.

6. **mlx_lm.server takes 5-15 minutes to load on first invocation.** Same on every Apple Silicon — it's a Python/MLX setup cost, not bandwidth-bound.

## Findings that *stop being problems* on bigger Macs

Most of the M4 Max 36 GB tuning advice exists *because* of the small memory budget. On bigger Macs these stop mattering:

| Problem on 36 GB M4 Max | M4 Max Studio 64 GB | M4 Max Studio 128 GB | M3 Ultra Studio 256-512 GB |
|---|---|---|---|
| Gemma 4 26B-A4B Q6_K @ 32K f16 OOMs at default `-ub 512` | Fits trivially | Fits trivially | Fits trivially |
| Gemma 4 31B-IT Q4_K_M needs turbo4 KV mandatory | f16 KV fits | f16 KV fits at full ctx | Same |
| Need `-ub 256` to reach 32K | Default `-ub 512` works at 32K | Default works to 64K+ | Default works to full context |
| Can't run anything ≥25 GB of weights | Can run up to ~45 GB models | Can run up to ~95 GB models | Can run up to ~470 GB models (with sysctl) |

The compute buffer formula doesn't change but **as a fraction of available memory it shrinks dramatically.** On a 36 GB Mac, 8.4 GB of compute buffer is 28% of the 30 GB working set — fatal. On a 256 GB Mac Studio, the same 8.4 GB is 4% of ~192 GB — irrelevant.

So on a Mac Studio you can mostly ignore the M4 Max-specific tuning we worked out. Just use the default config and the model will fit.

## Throughput projections (BANDWIDTH MATH ONLY — NOT MEASURED)

Decode throughput is approximately `bandwidth × utilization / bytes_per_token_read`. The M4 Max 36 GB column is measured; the rest are projections at the same utilization fraction we observed.

| Model | M4 Max 36 GB (410 GB/s) | M4 Max Studio 128 GB (546) | M3 Ultra Studio 256-512 GB (819) | RTX 5090 32 GB (1792) |
|---|---|---|---|---|
| Gemma 26B-A4B Q6_K (4B active MoE) | **60 tok/s** ✅ | ~80 ⚠️ | ~120 ⚠️ | **142** ✅ |
| Gemma 31B-IT Q4_K_M (dense) | **11.8 tok/s** ✅ (turbo4) | ~16 ⚠️ (f16) | ~24 ⚠️ (f16) | **50** ✅ |
| Qwen 27B Opus Q4_K_M (dense) | **13 tok/s** ✅ | ~17 ⚠️ | ~26 ⚠️ | not in baseline |
| Qwen 3.5 9B Q4_K_M | **35 tok/s** ✅ | ~47 ⚠️ | ~70 ⚠️ | not in baseline |
| Qwen 3.5 122B-A10B (10B active) | **OOM** | OOM | **~50-80 ⚠️**, fits in 256 GB | **OOM** |
| Qwen3-Coder-Next 80B-A3B (3B active) | OOM | OOM | **~150+ ⚠️**, 3B active | OOM |

✅ = measured / ⚠️ = projected from bandwidth math at the same utilization fraction we observed on M4 Max 36 GB

A few notes on the projections:

- **The bandwidth scaling assumes the same kernel utilization fraction.** We saw ~88% utilization on Gemma 26B-A4B Q6_K and ~70% on Qwen 27B dense Q4 on the M4 Max. Whether MPS hits the same fractions on M3 Ultra (older Metal) and full-bin M4 Max (more compute units) is an open question. Could be slightly higher, slightly lower.
- **M3 Ultra is ~46% of 5090 bandwidth, not 5090-class.** I had this wrong in an earlier draft (claimed ~1092 GB/s based on a non-existent M4 Ultra). Real M3 Ultra is 819 GB/s. So even on the high-end Mac Studio, dense ≥27B models will run noticeably slower than the 5090.
- **M3 Ultra is genuinely 5090-killer for huge MoE models** that the 5090 can't fit. Capacity matters more than bandwidth for these.

## Capacity expansion: what new models become possible

This is the most concrete thing bigger Macs unlock:

**At 64 GB M4 Max Studio (~48 GB working set):**
- Anything we tested on the 36 GB MBP, but at default config (no -ub tuning)
- Gemma 4 31B-IT Q8_0 (~33 GB weights) — top-quality dense at higher precision
- Qwen 3.5 35B-A3B Q4_K_M MoE (~22 GB) — the speed-tier MoE the 5090 ranking shows at 188 tok/s

**At 128 GB M4 Max Studio (~96 GB working set):**
- All of the above
- Qwopus 3.5 27B-v3 Q8_0, Harmonic 27B Q8_0 — high-precision fine-tunes
- Anything in the 5090 ranking, at higher precision than what fits on the 5090

**At 256-512 GB M3 Ultra Studio (~192-384 GB working set):**
- All of the above
- **Qwen 3.5 122B-A10B Q4_K_M** (~72 GB) — A-tier MoE that's Spark-only on the bandwidth-constrained side
- **Qwen3-Coder-Next 80B-A3B** — coding specialist MoE, extremely fast due to small active params
- **MiniMax-M2.5 230B-A10B** (~95 GB at Q3) — large hybrid MoE
- **70B dense models** (Llama 3.1 70B, Qwen 2.5 72B etc) at ~30-40 GB Q4. Will be slow (dense bandwidth math) but fit.
- **405B dense models** at Q4 (~200 GB) — only fits on the 512 GB SKU. Very slow (~3-5 tok/s by bandwidth math), but the only Apple Silicon that can load them.

The 256+ GB Studios are the only Apple Silicon configurations that compete with the [DGX Spark](MODEL_RANKINGS_SPARK.md) in capacity terms, and they have ~3× the bandwidth of the Spark (819 vs 273 GB/s). For MoE models with small active parameter counts, the M3 Ultra Studio is genuinely the most interesting Apple Silicon for inference work.

## What we *don't* know about M3 Ultra specifically

The M3 Ultra is a generation older than the M4 Max we measured. A few things could differ in ways we can't extrapolate:

1. **Compute buffer scaling.** We measured `~0.5 × n_ubatch × n_ctx / 1024 MiB` on M4 Max for Gemma 4. Whether M3 Metal allocates the same way is unverified. M3 doesn't have Metal 4 features (introduced in M4) — could mean different allocation patterns.
2. **MLX kernel performance gap.** The MoE -38% / dense +42% pattern was on M4 Metal with Apple's tensor APIs. M3 lacks some of those APIs. The gap could be smaller (less optimized) or larger (worse MoE handling) on M3 Ultra.
3. **Single-buffer hardware ceiling.** M2 era was reportedly 17.1 GB max single MTL buffer. M4 we haven't probed. M3 Ultra at very large contexts (128K+) could hit a single-buffer wall before it hits the working-set ceiling.
4. **NUMA-like effects across UltraFusion-connected dies.** M3 Ultra is two M3 Max dies fused. For models that are larger than a single die's local memory pool, accesses across the UltraFusion link may have different latency than local memory. Hasn't been characterized for inference workloads in published material we know of.
5. **Whether sliding-window attention compute buffer scaling is the same on M3.** Gemma 4's 5-global / 25-sliding pattern is a recent architecture; M3's MPS implementation may handle it differently than M4 does.

**Anyone who actually buys an M3 Ultra Studio for this kind of work should re-run our `n_ubatch` sweep and our turbo4-vs-f16 comparison to verify the M4 findings transfer.** The qualitative direction should hold; the absolute numbers are a guess.

## Decision matrix for buying Apple Silicon for local inference (April 2026)

| Use case | Best Mac for it | Why |
|---|---|---|
| Run our exact M4 Max model lineup but stop fighting OOMs | **M4 Max Studio 64 GB** | 546 GB/s, 48 GB working set, no `-ub` tuning needed. Cheapest "drop the workarounds" upgrade. |
| Highest-quality coding model at interactive speed | **M4 Max Studio 128 GB** | Plenty of room for any quant of Gemma 31B / Qwen 27B / Harmonic 27B at Q8_0. Probably the sweet spot for "I run the same models, just better." |
| Run the Spark's 122B-A10B MoE faster than the Spark | **M3 Ultra Studio 256 GB** | 819 GB/s on a 72 GB model = ~80 tok/s projected vs Spark's measured 21 tok/s. ~4× speedup on the same model. |
| Run 405B dense models on Apple Silicon at all | **M3 Ultra Studio 512 GB** | The only Apple Silicon with the capacity. Expect ~3-5 tok/s — "loads, but not interactive." |
| Maximum throughput for dense models that fit | **RTX 5090** (not Apple Silicon) | 1792 GB/s and all the CUDA-only optimizations. M3 Ultra is ~46% of 5090 bandwidth. |
| Maximum capacity, accept bandwidth limit | **DGX Spark** (not Apple Silicon) — 128 GB at 273 GB/s | Loses to M3 Ultra on bandwidth, but cheaper and still very capable for small-active MoEs. |
| Portable inference on battery | **M4 Max MBP** (binned 36 GB OK, full 128 GB better) | Only mobile option in this comparison. Tune `-ub 256` if you have the 36 GB binning. |

## What about the M5 Ultra?

Rumored mid-2026. No official specs yet. If Apple follows the historical pattern (Ultra = 2× Max bandwidth) and the M5 Max is in the ~600-700 GB/s range, the M5 Ultra would be ~1200-1400 GB/s — finally approaching 5090 bandwidth territory. But this is rumor, not spec; **no projections from us until Apple ships it**.

## Cross-references

- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — the only measured rankings for any Apple Silicon in this repo
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — where the compute-buffer-scales-with-`n_ubatch` finding came from
- [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md) — why turbo4 KV is a Metal-side speed loss
- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — measured cross-platform comparison (5090, M4 Max 36 GB, DGX Spark)
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — the bandwidth-rich CUDA reference
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — the capacity-rich bandwidth-poor reference
