# Generalizing the M4 Max Findings to Other Apple Silicon

**Three flavors of evidence in this doc:**
1. ✅ **Measured by us** — only the M4 Max 36 GB MacBook Pro rows in any throughput table. See [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md).
2. 📰 **Measured by others** — public benchmarks from people who own M3 Ultra or M5 Max hardware. Cited inline.
3. ⚠️ **Projected** — bandwidth math at the same utilization fractions we observed. The asterisk on the projection: kernel utilization may differ on older (M3) or newer (M5) Metal generations.

The purpose of this doc is to answer: "I have M4 Max findings — how do they apply to a different Mac?"

## The Mac product matrix (April 2026)

Apple **did not release an M4 Ultra**. The 2025 Mac Studio refresh shipped with the M4 Max as the base SKU and the **M3 Ultra** (older generation, but with UltraFusion) as the high-end SKU. The **M5 Max / M5 Pro** then shipped in **March 2026** in MacBook Pros with a new "Apple Fusion Architecture" (two dies bonded into a single SoC). The **M5 Ultra** is expected at WWDC June 2026; the next Mac Studio refresh is expected to ship M5 Max + M5 Ultra at that point.

| Chip | CPU / GPU | Memory bandwidth | Memory range | Generation | Status |
|---|---|---|---|---|---|
| M4 Max (binned, MBP only) | 14C / 32C | **410 GB/s** | 36 GB | M4 | Shipping (this is what we measured) |
| M4 Max (full) | 16C / 40C | **546 GB/s** | 48 / 64 / 128 GB | M4 | Shipping (MBP + Mac Studio entry) |
| **M3 Ultra** | 28C / 60C or 32C / 80C | **819 GB/s** | 96 / 256 / 512 GB | M3 (older Metal) | Shipping (Mac Studio high-end) |
| **M5 Max** (binned) | 18C / 32C | **460 GB/s** | up to 128 GB | M5 (Neural Accelerators) | Shipping March 2026 (MBP only, for now) |
| **M5 Max** (full) | 18C / 40C | **614 GB/s** | up to 128 GB | M5 | Shipping March 2026 (MBP only, for now) |
| M5 Ultra | rumored 36C / 80C | rumored ~1228 GB/s | rumored up to 512 GB+ | M5 | Expected June 2026 (WWDC), unannounced |

The big M5-generation change: each GPU core in M5 has dedicated **Neural Accelerators**, the same idea NVIDIA uses for tensor cores. This matters for inference: Apple's own research paper measured the M5 vs M4 on Qwen3-14B-4bit and found **TTFT 4.06× faster** on M5, **decode 1.19× faster**. The decode improvement is roughly bandwidth-proportional (614/546 = 1.12×), but the TTFT improvement is structural — Neural Accelerators are designed for the matmul-heavy prefill phase. Sources at the bottom.

The 36 GB binned M4 Max we benchmarked is the *most* memory-constrained and *least* bandwidth-rich Apple Silicon you can currently buy in this product family. Everything else is more capable along at least one of the two axes.

## Findings that transfer identically

These are properties of llama.cpp's Metal backend, MLX kernels, and the LM Studio API surface — they don't depend on memory size or bandwidth:

1. **Metal working set ≈ 75% of unified memory** by default (`recommendedMaxWorkingSetSize`). On macOS Sonoma+, the override is `sudo sysctl iogpu.wired_limit_mb=<MB>` (older macOS used `debug.iogpu.wired_limit` in bytes). The change takes effect immediately, no reboot. Leave 4-8 GB for macOS — going to 100% causes hard reboots ([Apple Silicon VRAM increase guide, 2026](https://medium.com/@se.mehmet.baykar/increase-vram-on-apple-silicon-for-local-llms-1b35c453b165)). Resets on reboot unless you set up a Launch Daemon. The 75% default ratio holds across every Apple Silicon generation.

2. **Compute buffer formula** for sliding-window architectures (Gemma 4):
   ```
   compute_buf_MiB ≈ 0.5 × n_ubatch × n_ctx / 1024
   ```
   This is kernel-level, not device-level. Should hold on any Apple Silicon Metal backend running llama.cpp. **Caveat:** verified on M4 Max only. M3 Ultra is one generation older and could have different MPS allocation patterns — needs measurement.

3. **TurboQuant turbo4 KV is slower than f16 KV** on Metal. The dequant compute overhead exceeds the bandwidth savings on bandwidth-constrained platforms — and even high-bandwidth Apple Silicon is bandwidth-constrained relative to its compute budget. The only reason to use turbo4 on any Mac is when the KV cache itself is the bottleneck (essentially just dense full-attention models like Gemma 4 31B-IT at long contexts).

4. **MLX vs llama.cpp on M3/M4: dense-favorable, MoE-mixed.** Our measurement was MLX +42% on Qwen 27B dense, -38% on Gemma 4 26B-A4B MoE. Public reports suggest the MoE story is *model-specific*: MLX is 3× *faster* than llama.cpp on Qwen3-Coder-Next ([llama.cpp issue #19366](https://github.com/ggml-org/llama.cpp/issues/19366)) but slower on Gemma 4 sliding-window models. Documented root causes for the cases where MLX loses:
   - **MLX flash attention** wasn't available until recently — when llama.cpp's `-fa on` is enabled, llama.cpp gains a free 2× on long context that MLX doesn't get ([mlx-lm issue #763](https://github.com/ml-explore/mlx-lm/issues/763)).
   - **Sliding window + global attention** doesn't map cleanly to MLX's matmul primitives, which is exactly what makes Gemma 4 a bad case for MLX.
   - **Prompt caching** is broken for some multimodal MoE setups in the LM Studio MLX runtime — every conversation turn re-processes the entire history.

   So our generalization is: MLX wins on dense models with simple attention, loses on hybrid-attention models, and is sometimes much faster on small-active-MoE coder specialists. **Always benchmark before committing to one engine.**

5. **The MLX vs llama.cpp story changes structurally on M5.** M5 introduced **Neural Accelerators in each GPU core**, which MLX uses but llama.cpp's Metal backend doesn't (yet). Apple's research paper measured MLX on M5 vs M4 at **TTFT 4.06× faster, decode 1.19× faster** for Qwen3-14B-4bit ([Apple research, January 2026](https://machinelearning.apple.com/)). Public benchmarks corroborate that MLX is now ~40-80% faster than llama.cpp on M5 hardware ([Markus Schall's M3 Ultra vs RTX 5090 comparison](https://www.markus-schall.de/en/2025/11/apple-mlx-vs-nvidia-how-local-ki-inference-works-on-the-mac/)). **Until llama.cpp's Metal backend gets Neural Accelerator support, MLX is the right default on M5 hardware** — even on the model classes where llama.cpp won on M4.

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

## Throughput across the Apple Silicon lineup

Three sources of data, marked separately. Decode tok/s unless otherwise noted.

| Model | M4 Max 36 GB (410) | M4 Max Studio 128 GB (546) | M3 Ultra Studio 256-512 GB (819) | M5 Max MBP 128 GB (614) | RTX 5090 (1792) |
|---|---|---|---|---|---|
| Gemma 26B-A4B Q6_K (4B active MoE) | **60** ✅ | ~80 ⚠️ | ~120 ⚠️ | ~90 ⚠️ | **142** ✅ |
| Gemma 31B-IT Q4_K_M (dense) | **11.8** ✅ (turbo4) | ~16 ⚠️ (f16) | ~24 ⚠️ (f16) | ~18 ⚠️ | **50** ✅ |
| Gemma 3 27B Q4 (dense) | not tested | not tested | **~41** 📰 [1] | not tested | not in baseline |
| Qwen 27B Opus Q4_K_M (dense) | **13** ✅ | ~17 ⚠️ | ~26 ⚠️ | ~20 ⚠️ | not in baseline |
| Qwen 3.5 9B Q4_K_M | **35** ✅ | ~47 ⚠️ | ~70 ⚠️ | ~52 ⚠️ | not in baseline |
| Qwen 3 235B FP8 (22B active MoE) | OOM | OOM | **~25-35** 📰 [2] | OOM (>128 GB) | OOM |
| Qwen3.5-35B-A3B Q4_K_M (3B active MoE) | not tested | ~150 ⚠️ | ~225 ⚠️ | ~165 ⚠️ (Ollama 0.18) / **~134** 📰 [3] (Ollama 0.19+MLX) | **188** ✅ |
| Llama 70B / Qwen 72B (dense, Q4) | OOM | OOM | **12-15** 📰 [1] | OOM | OOM |
| DeepSeek-V3 671B (37B active, MLX) | OOM | OOM | **>20** 📰 [2] (mlx-lm only) | OOM | OOM |

**Legend:** ✅ measured by us / 📰 measured by others, see source links / ⚠️ projected from bandwidth math at the same utilization fraction we observed on M4 Max 36 GB

A few notes on what these numbers mean:

- **The third-party Gemma 3 27B Q4 measurement on M3 Ultra (~41 tok/s)** is meaningfully *higher* than our projection of ~26 tok/s for Qwen 27B Opus Q4. Either Gemma 3 27B has lighter per-token bandwidth than Qwen 27B (different attention heads), or the M3 Ultra hits a higher kernel utilization than M4 Max 36 GB does, or both. We haven't been able to disentangle.
- **M3 Ultra at 70B dense Q4 = 12-15 tok/s** is the canonical "this hardware loads the model but runs it slowly" data point. 70B Q4 is ~35 GB → at 819 GB/s the bandwidth ceiling is ~23 tok/s, so 12-15 = 52-65% utilization. Slightly lower than what we see on smaller models on the same hardware.
- **The 5090 is still ~2.2× faster than the M3 Ultra** for models that fit on both. Bandwidth ratio = 1792/819 = 2.19, and the practical ratio (e.g. 142 vs ~120 on Gemma 26B-A4B) tracks that closely.
- **M3 Ultra is genuinely 5090-killer for huge MoE models** that the 5090 can't fit. For Qwen 3 235B / DeepSeek-V3 / Qwen 3.5 122B-A10B, the 5090 is OOM and the M3 Ultra is the only consumer-priced option at any speed. Same story as the [DGX Spark](MODEL_RANKINGS_SPARK.md), but at 3× the bandwidth.
- **M5 Max projections in this table are mostly bandwidth math too**, but the kernel story may have shifted (see next section). Apple's published 1.19× decode speedup for M5 vs M4 on Qwen3-14B-4bit is the main concrete data point we have.

**Sources cited above:**
- [1] [Mac Studio M3 Ultra 96GB LLM Performance, MacRumors Forums](https://forums.macrumors.com/threads/mac-studio-m3-ultra-96gb-28-60-llm-performance.2456559/) and Alex Ziskind YouTube benchmarks
- [2] [Notes on Early Mac Studio AI Benchmarks with Qwen3-235B and Qwen2.5-VL-72B, MacStories](https://www.macstories.net/notes/notes-on-early-mac-studio-ai-benchmarks-with-qwen3-235b-a22b-and-qwen2-5-vl-72b/)
- [3] [Ollama is now powered by MLX on Apple Silicon in preview, Ollama Blog](https://ollama.com/blog/mlx)

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
| Highest-quality coding model on a laptop | **M5 Max MBP 128 GB (full bin)** | March 2026 release. 614 GB/s bandwidth, Neural Accelerators give 4× faster prefill (huge for long-context coding agents) and ~1.2× faster decode than M4 Max. New "best Apple Silicon laptop for inference" winner. |
| Highest-quality coding model at interactive speed (desk) | **M4 Max Studio 128 GB** (today) or wait for M5 Max Studio (June 2026) | Plenty of room for Q8_0 of any of our S-tier models. M5 Max Studio at the same memory tier should be the upgrade target if you can wait. |
| Run the Spark's 122B-A10B MoE faster than the Spark | **M3 Ultra Studio 256 GB** | 819 GB/s on the 72 GB model = ~3× the Spark's bandwidth. Real measurements suggest ~50-80 tok/s on Qwen 3 235B FP8 (22B active) — and that's the *current* generation, M5 Ultra would be faster again. |
| Run 405B+ dense models on Apple Silicon at all | **M3 Ultra Studio 512 GB** | The only Apple Silicon with the capacity for now. Expect single-digit tok/s — "loads, but not interactive." 70B Q4 measured at 12-15 tok/s on this hardware. |
| Maximum throughput for dense models that fit | **RTX 5090** (not Apple Silicon) | 1792 GB/s and all the CUDA-only optimizations. M3 Ultra is ~46% of 5090 bandwidth. M5 Ultra (rumored ~1228 GB/s) will close the gap but probably won't pass it. |
| Maximum capacity, accept bandwidth limit | **DGX Spark** (not Apple Silicon) — 128 GB at 273 GB/s | Loses to M3 Ultra on bandwidth, but typically cheaper and still very capable for small-active MoEs. |
| Portable inference on battery | **M5 Max MBP** (full bin 614 GB/s) | New 2026 winner. The M5's Neural Accelerators are a structural advantage for inference workloads. M4 Max MBP is still fine, just slower. |

## What about the M5 Ultra?

Expected at WWDC June 2026. No official specs yet, but the building blocks are now public:

- **M5 Max full bin: 614 GB/s, 18C / 40C, up to 128 GB** (March 2026 launch)
- **Apple Fusion Architecture** is the new die-bonding tech that replaces UltraFusion. M5 Pro/Max are already 2-die designs internally.
- **If M5 Ultra is 2× M5 Max** in the historical pattern, that's ~1228 GB/s — about 68% of the 5090's 1792 GB/s, the closest Apple has come to NVIDIA bandwidth on a single workstation.
- **If the new Fusion Architecture changes the multiplier** (Apple has discussed allowing 4-die configurations), M5 Ultra could be even more aggressive.

What we *can* say from the M5 Max measurements that already exist: **the Neural Accelerator advantage is the bigger story than the bandwidth bump.** M5 Max + MLX is 1.19× faster decode and 4.06× faster prefill than M4 Max on the same model — and that's just the binned 32-core GPU SKU. For a coding agent with long prompts (where prefill matters), M5 Max + MLX is already a significantly better experience than anything in the M4 generation.

The M5 Ultra Studio at 512 GB and ~1228 GB/s would finally be a desk machine that runs Qwen 3 235B (22B active) at >50 tok/s, handles 405B dense models in the 10-15 tok/s range, and gets within 30% of the 5090 on dense models that fit on both. But all of that is **rumor + bandwidth math**, not measurement.

## Cross-references (other docs in this repo)

- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — the only measured rankings for any Apple Silicon in this repo
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — where the compute-buffer-scales-with-`n_ubatch` finding came from
- [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md) — why turbo4 KV is a Metal-side speed loss
- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — measured cross-platform comparison (5090, M4 Max 36 GB, DGX Spark)
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — the bandwidth-rich CUDA reference
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — the capacity-rich bandwidth-poor reference

## External sources

Apple's product spec sheets:
- [Mac Studio (2025) Tech Specs (M4 Max + M3 Ultra)](https://support.apple.com/en-us/122211)
- [MacBook Pro (M5 Pro / M5 Max, 2026) Tech Specs](https://support.apple.com/en-us/126318)
- [Apple Newsroom — March 2025 Mac Studio launch](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/)
- [Apple Newsroom — March 2026 M5 Pro/Max launch](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)

M3 Ultra inference benchmarks (third-party):
- [Apple's M3 Ultra Mac Studio Misses the Mark for LLM Inference (Billy Newport, Medium)](https://medium.com/@billynewport/apples-m3-ultra-mac-studio-misses-the-mark-for-llm-inference-f57f1f10a56f) — argues GPU compute, not RAM, is the bottleneck even on the high-end Studio
- [Mac Studio M3 Ultra 96GB LLM Performance (MacRumors Forums)](https://forums.macrumors.com/threads/mac-studio-m3-ultra-96gb-28-60-llm-performance.2456559/) — community benchmarks
- [Notes on Early Mac Studio AI Benchmarks with Qwen3-235B-A22B and Qwen2.5-VL-72B (MacStories)](https://www.macstories.net/notes/notes-on-early-mac-studio-ai-benchmarks-with-qwen3-235b-a22b-and-qwen2-5-vl-72b/) — large MoE measurements
- [Apple Mac Studio with M3 Ultra Review (Creative Strategies)](https://creativestrategies.com/mac-studio-m3-ultra-ai-workstation-review/)

M5 Max inference benchmarks (third-party):
- [Benchmarking Open-Weights LLMs on the MacBook Pro M5 Max (Wale Akinfaderin, Medium)](https://medium.com/@WalePhenomenon/benchmarking-open-weights-llms-on-the-macbook-pro-m5-max-d4347e9457af)
- [Apple M5 Max Local LLM Guide: 128GB Benchmarks (AI Productivity)](https://aiproductivity.ai/blog/apple-m5-max-local-llm-guide/)
- [Ollama is now powered by MLX on Apple Silicon in preview](https://ollama.com/blog/mlx) — notes Neural Accelerator integration, claims 40-80% throughput improvement over llama.cpp

MLX vs llama.cpp deep dives:
- [llama.cpp issue #19366: MLX 3× faster on Qwen 3 Coder Next](https://github.com/ggml-org/llama.cpp/issues/19366)
- [mlx-lm issue #763: Long-context generation 50% slower than llama.cpp+FA](https://github.com/ml-explore/mlx-lm/issues/763)
- [Benchmarking Apple's MLX vs llama.cpp (Andreas Kunar, Medium)](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416)
- [A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp (arXiv 2511.05502)](https://arxiv.org/pdf/2511.05502)
- [Mac with M3 Ultra against RTX 5090: Efficiency instead of watts (Markus Schall)](https://www.markus-schall.de/en/2025/11/apple-mlx-vs-nvidia-how-local-ki-inference-works-on-the-mac/)

macOS Metal memory tuning:
- [Apple Developer: MTLDevice.maxBufferLength](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength) — official Apple docs (the specific value for M3 Ultra and M4/M5 Max is not in the public docs and would need to be probed at runtime)
- [Increase VRAM on Apple Silicon for Local LLMs (Mehmet Baykar, Medium, March 2026)](https://medium.com/@se.mehmet.baykar/increase-vram-on-apple-silicon-for-local-llms-1b35c453b165) — current guide to `iogpu.wired_limit_mb`
- [Adjust VRAM/RAM split on Apple Silicon (llama.cpp Discussion #2182)](https://github.com/ggml-org/llama.cpp/discussions/2182)

llama.cpp Apple Silicon performance tracking:
- [Performance of llama.cpp on Apple Silicon M-series (Discussion #4167)](https://github.com/ggml-org/llama.cpp/discussions/4167) — long-running community benchmark thread
- [GPU Benchmarks on LLM Inference (Xiongjie Dai, GitHub)](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) — multi-GPU + Apple Silicon comparison repo
