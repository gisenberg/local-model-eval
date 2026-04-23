# Context Capacity Analysis — MacBook Pro M4 Max 36 GB

Tested April 2026.

For the RTX 5090 equivalent, see [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md). The Mac story is fundamentally different — capacity is limited by Metal's working set ceiling, not VRAM, and there's no graceful degradation when you go over.

> **⚠ UPDATE 2026-04-11 (after rotorquant rebase):** Much of the original analysis below was **base-version specific** to the turboquant fork commit `8590cbff9` we were building from, NOT a fundamental Metal property. Specifically:
>
> 1. **The "compute buffer scales at ~260 MiB per 1024 tokens for sliding-window models" formula was a bug in the old base**, not a Metal characteristic. A newer llama.cpp base (obtained by cherry-picking upstream PR #21309 for Gemma 4 support) has a compute buffer of **~523 MiB at 32K context regardless of `n_ubatch` or model** — a **16× reduction**. The `-ub 256` workaround we documented extensively is now unnecessary on the newer base.
>
> 2. **The claim "Gemma 4 31B-IT has ~870 KB/token KV at f16" was wrong math.** I assumed every layer of the 62-layer network uses full attention; actually Gemma 4 31B uses ISWA (Interleaved Sliding Window Attention) where only ~9 layers are global attention and the rest are sliding-window. Measured actual KV at 32K f16 is **~3.7 GB total** (~120 KB/token averaged), not the 27.8 GB my math predicted. The same "KV cache is the bottleneck" framing was therefore also wrong: Gemma 4 31B f16 KV at 32K fits comfortably at 3.7 GB.
>
> 3. **Turbo4 KV was never "mandatory" for Gemma 31B.** It was mandatory on the old base (because of the compute buffer bug), not because of KV math. On the new base, Gemma 31B f16 fits at 64K with default `-ub`, and turbo4 at 128K+ with default `-ub`.
>
> **What's still true**: the Metal working set ceiling IS ~30 GB (not 36 GB), OOMs are still OOM-not-spill, and there's no PCIe fallback. The per-token KV cost numbers for specific architectures are correct (except the Gemma 4 31B row which was based on bad math). The sections below that discuss "what fits at what context" are accurate for the old-base build at the time of testing — all those measurements were real. But the **mechanism** I attributed them to (compute-buffer-scales-with-`n_ubatch`) was a base-version-specific artifact, and the `-ub 256` workaround advice only applies if you're stuck on that old base.
>
> **If you're on a current llama.cpp base**: stop worrying about `-ub 256`, it's not needed. Use default flags and the context ceilings below that are based on the old compute-buffer formula will roughly **4× as many tokens** at the same memory budget.
>
> The updated numbers for Gemma 4 31B-IT on the new base:
>
> | Config | 32K | 64K | 128K | 256K |
> |---|---|---|---|---|
> | f16 / f16 (default ub) | ✅ 21.7 GB | ✅ 24.3 GB | ❌ 29.4 GB (OOM by ~700 MB) | — |
> | planar3 / f16 K-only (default ub) | ✅ 20.9 GB | — | ❌ 30.5 GB (K-only is deferred — no memory savings at allocation) | — |
> | turbo4 / turbo4 (default ub) | ✅ ~18.5 GB | ✅ — | ✅ 21.0 GB | ✅ 24.1 GB (loads; decode unverified) |
>
> Details in [ROTORQUANT.md](ROTORQUANT.md) under "Context window gains" — the rebase that unblocked Gemma 4 testing also surfaced these base-version issues.

---

## Original analysis (old-base-specific, preserved for historical reference)

## The Defining Constraint: Compute Buffer Sized by `n_ubatch × n_ctx` (Not the KV Cache!)

The Mac advertises 36 GB of unified memory but **the GPU can only address ~30 GB** of it. This is `recommendedMaxWorkingSetSize` from Metal device init, ~30150 MB on this binning. macOS reserves the rest for the system, the display, and other Metal clients.

```
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 30150.67 MB
```

This 30 GB is the **hard ceiling** for `weights + KV cache + compute buffers + scratch`, summed across all loaded models. Going over does NOT produce graceful spillover (like the 5090 does to system RAM via PCIe). It produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` and every chat completion returns `500 Compute error`.

But the surprise is **what dominates that 30 GB budget**. It's not the KV cache. It's the compute buffer — and that buffer is sized by `n_ubatch × n_ctx`, not by anything you'd normally tune.

### TL;DR: lower `-ub` to fit more context

With the default `-ub 512`, Gemma 4 26B-A4B Q6_K needs 8.4 GB of compute buffer at 32K context. With `-ub 256`, it needs **4.2 GB**. Same model, same context, same throughput, half the memory pressure. This is the single most important config knob on this hardware.

| Config | Compute buffer at 32K ctx | Total | Result |
|---|---|---|---|
| `-ub 512` (default) | 8,402 MiB | 30,141 MiB | ❌ OOM |
| `-ub 256` | 4,211 MiB | 26,672 MiB | ✅ comfortable |
| `-ub 128` | 2,121 MiB | 24,584 MiB | ✅ even more headroom |

The scaling is linear in *both* `n_ubatch` and `n_ctx`: roughly **`compute_buf_MiB ≈ 0.5 × n_ubatch × n_ctx / 1024`** for sliding-window models like Gemma 4. So to push to 64K, halve ubatch: `-ub 128`. To push to 128K, halve again: `-ub 64`. Each halving roughly doubles the prefill chunking on max-length prompts (from 4 chunks to 8 to 16 for a 2048-token prompt) but keeps decode throughput identical.

**Important: lower `-ub` only, not `-b`.** `-b` (default 2048) is the maximum prompt length; reducing it below your typical prompt size will reject long prompts. `-ub` is only the physical batch size for forward passes — long prompts get chunked, not rejected.

### The compute buffer is much bigger than the KV cache on Metal

For Gemma 4 26B-A4B Q6_K with turbo4 KV at 32K context, the actual breakdown llama-server reports with default `-ub 512`:

```
load_tensors:  MTL0_Mapped model buffer size  = 21574.57 MiB    ← weights
llama_kv_cache:    MTL0 KV buffer size        =   170.13 MiB    ← turbo4 KV (TINY)
llama_kv_cache:    MTL0 KV buffer size        =    79.81 MiB    ← second buffer
sched_reserve:     MTL0 compute buffer size   =  8402.37 MiB    ← THIS is the dominant cost
```

The KV cache is **170 MiB**. Three orders of magnitude smaller than the compute buffer. Compressing the KV cache further (turbo4 already gives 3.8x compression) saves nothing meaningful — there's nothing left to compress.

### How the compute buffer scales with `n_ubatch` and `n_ctx`

Two sweeps, holding everything else constant:

**Vary n_ubatch at fixed 32K context:**

| n_ubatch | Compute buffer | Total projected | Fits? |
|---|---|---|---|
| 1 | 17.83 MiB | 22,482 MiB | ✅ (but n_batch=1 caps prompts at 1 token) |
| 64 | 1,077 MiB | 23,541 MiB | ✅ |
| 128 | 2,121 MiB | 24,584 MiB | ✅ |
| **256** | **4,210 MiB** | **26,672 MiB** | **✅ recommended** |
| 512 (default) | 8,389 MiB | 30,898 MiB | ❌ OOM |
| 1024 | 16,749 MiB | 39,353 MiB | ❌ way over |

**Vary n_ctx at fixed `-ub 512` (the default that we're trying to avoid):**

| Context | KV cache | Compute buffer | Total | Fits? |
|---|---|---|---|---|
| 16K | 165 MiB | 4,242 MiB | ~26.0 GB | ✅ |
| 20K | 212 MiB | 5,282 MiB | ~27.1 GB | ✅ |
| 24K | 254 MiB | 6,322 MiB | ~28.1 GB | ❌ Metal allocator fails |
| 32K | 339 MiB | 8,402 MiB | ~30.2 GB | ❌ |

Combining both: the formula is roughly `compute_buf_MiB ≈ 0.5 × n_ubatch × n_ctx / 1024` for sliding-window architectures. **This is the single most important number to know on this hardware.** The default `n_ubatch=512` was designed for GPUs with much more headroom; on the M4 Max, halving it is essentially free (decode throughput unchanged) and unlocks the actual context window that the working set can support.

### Why `-ub 256` is "free" — the prefill cost

`n_ubatch` is the *physical* batch size (tokens processed per forward pass), not the *logical* batch size (`n_batch`, the maximum prompt length). When you submit a 2000-token prompt with `-b 2048 -ub 256`, llama.cpp chunks it into 8 micro-batches of 256 tokens and processes them sequentially. With `-ub 512` it would be 4 chunks. So prefill is ~2× slower for max-length prompts at `-ub 256`.

**Decode throughput is unchanged.** Decode processes 1 token at a time anyway, so the ubatch size is irrelevant during generation.

For our coding benchmarks (200-400 token prompts), the prefill cost is invisible: a 200-token prompt is one chunk regardless of whether ub=256 or ub=512. We measured 61 tok/s decode on Gemma 26B Q6_K f16 32K with `-ub 256`, vs 60 tok/s on the same model at 16K with default `-ub 512` — within variance.

### Why the 5090 doesn't have this problem in the first place

The 5090 ranking shows the same Gemma 4 26B-A4B Q6_K turbo4 model using 25,636 MB *total* VRAM at 32K context with default settings. M4 Max projects 30,244 MiB at the same configuration. **The difference is ~4.6 GB, all of it in the compute buffer.**

CUDA's compute buffer for this model class is roughly half the size of Metal's at the same `n_ubatch`. Possible reasons:

- cuBLAS/cuDNN kernels can stream tensors through working memory in smaller chunks; Metal Performance Shaders may pre-allocate full intermediate tensors.
- Flash attention on CUDA has been heavily optimized for memory efficiency over multiple years; the Metal flash attention path is newer.
- The sliding window attention pattern in Gemma 4 (15 global + 15 sliding layers) may need re-materialization buffers on Metal that CUDA avoids via different kernel fusion.

The exact root cause would require profiling both backends, but the *practical* implication is clear: **on Metal you have to actively reduce `n_ubatch` to recover the context window the 5090 gets for free at default settings.**

## What Fits at What Context (Measured)

Each row is a real run from the M4 Max benchmark. ❌ = OOM during inference, ✅ = generates successfully.

All measured. The "ubatch" column matters: most rows below use `-ub 256`, the recommended config. Default is `-ub 512`.

| Model | Quant | KV | Weights | Context | ubatch | Result |
|---|---|---|---|---|---|---|
| Nemotron 3 Nano 4B | Q4_K_M | f16 | 2.8 GB | 32K | default | ✅ 65 tok/s |
| Qwen 3.5 9B | Q4_K_M | f16 | 5.5 GB | 32K | default | ✅ 35 tok/s |
| Gemma 4 26B-A4B | Q4_K_M | f16 | 16.5 GB | 32K | default | ✅ 59 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 16K | default | ✅ 60 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 32K | **256** | **✅ 61 tok/s** (compute buf 4.2 GB) |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 64K | **128** | ✅ verified loads (4.2 GB compute buf, 27.3 GB total) |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 32K | default 512 | ❌ OOM (compute buf 8.4 GB) |
| Gemma 4 26B-A4B | Q6_K | turbo4 | 22.6 GB | 32K | **256** | **✅ 45 tok/s** (slower than f16, no benefit) |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | f16 | 16.5 GB | 32K | default | ✅ 13 tok/s |
| Gemma 4 31B-IT | Q4_K_M | f16 | 18.3 GB | 16K | any | ❌ OOM (KV cache itself is 14 GB at f16, mandatory turbo4) |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 32K | **256** | **✅ 11.8 tok/s** |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 64K | 256 | ❌ OOM (compute buf 16.7 GB at -ub 256, would need -ub 128) |

**Key findings:**

1. **Gemma 4 26B-A4B Q6_K reaches 32K context with `-ub 256`**, not 16K as we initially documented. The compute buffer at default `-ub 512` is 8.4 GB at 32K (OOM), but at `-ub 256` it's 4.2 GB and 32K fits comfortably. Same throughput (61 vs 60 tok/s). To go further: 64K with `-ub 128`, ~128K with `-ub 64`. The cost is slower prefill on max-length prompts (more chunks of smaller batch), nothing else.

2. **Gemma 4 31B-IT reaches 32K with turbo4 KV + `-ub 256`** (we previously had it pinned at 16K). Two constraints stack here: turbo4 KV is mandatory because dense f16 KV is 14 GB at 16K, AND `-ub 256` is needed to fit the compute buffer at 32K context.

3. **Qwen 3.5 27B Opus-Distilled fits at 32K f16 with default ubatch.** Qwen 3.5 uses GQA (8 KV heads vs Gemma 4 31B's 16) — smaller KV per token. And dense models without sliding window have a different (smaller) compute buffer scaling pattern than Gemma 4's 5-global-15-sliding architecture.

4. **The default `-ub 512` was the wrong default for this hardware.** It was designed for GPUs with much larger compute-buffer headroom. On the M4 Max, the right default for any model with ≥20 GB weights is `-ub 256`. There's no decode throughput cost (the only cost is ~2× slower prefill on max-length prompts, which is invisible for short prompts).

## Per-Token KV Cost (Measured Architectures)

Same formula as the 5090 page:

```
KV bytes/token = 2 (K+V) × num_attn_layers × num_kv_heads × head_dim × 2 (FP16)
```

Approximations for the models in this benchmark:

| Model | Layers | KV Heads | Head Dim | KV/token (f16) | KV @ 16K | KV @ 32K |
|---|---|---|---|---|---|---|
| nemotron-3-nano-4b | 42 (4 attn) | 8 | 128 | ~16 KB | 0.25 GB | 0.5 GB |
| qwen3.5-9b | 32 | 4 | 256 | ~128 KB | 2.0 GB | 4.0 GB |
| gemma-4-26b-a4b | 30 (15 global + 15 sliding) | 8 | 256 | ~120 KB (global only) | 1.9 GB | 3.8 GB |
| qwen3.5-27b-opus | 32 | 8 | 256 | ~256 KB | 4.0 GB | 8.0 GB |
| gemma-4-31b-it (dense) | 62 | 16 | 256 | ~870 KB | **13.6 GB** | 27.2 GB |

The Gemma 4 31B-IT KV cost is the canonical "this model can only run with KV compression on a 30 GB platform" case — even at 16K f16 the KV alone is 14 GB, leaving no room for the 18 GB of weights.

## What Couldn't Be Tested Here

- **Anything with weight size > ~25 GB** at any context. Examples: Gemma 4 31B-IT Q6_K, Qwen 3.5 27B Q8_0, Gemma 4 26B-A4B Q8_0. The Metal working set ceiling is the hard constraint.
- **Long-context scaling sweeps** (32K → 64K → 112K → 192K → 256K like the 5090 page). On the M4 Max most models that run at 32K already use a significant fraction of the budget, so 64K+ is moot for anything bigger than ~9B dense or the small-active MoEs.

## TurboQuant on M4 Max: Capacity Yes, Speed No

Per the [M4 Max rankings](MODEL_RANKINGS_M4MAX.md), TurboQuant turbo4 KV is *slower* than f16 KV on this hardware (the dequant compute overhead exceeds the bandwidth savings). But it does still buy real capacity. Concretely:

- Gemma 4 31B-IT Q4_K_M @ 16K: f16 = 32 GB total = OOM, turbo4 = 22 GB total = ✅
- Gemma 4 26B-A4B Q6_K @ 16K: f16 = 26 GB ✅, turbo4 = 24 GB ✅ (no benefit from turbo4)
- Gemma 4 26B-A4B Q6_K @ 32K: f16 = 32 GB ❌, turbo4 = 28 GB ❌ (still over because weights are 22 GB)

**Rule of thumb:** Use turbo4 if and only if the model literally won't load with f16 KV. There's no speed benefit on this platform.

## Why the Mac Doesn't Have a "Spill Cliff" Like the 5090

The 5090 page documents a famous failure mode: a model loads fine, runs at full speed for short prompts, then drops from 150 to 55 tok/s when the context fills up because portions of the KV cache silently spilled to system RAM via PCIe. The Mac has no equivalent because:

1. **There's no PCIe.** Memory is unified — the GPU and CPU share the same physical pool. There's no remote bus to spill across.
2. **The Metal working set is a hard limit, not a soft budget.** macOS doesn't transparently page GPU buffers in and out of system RAM. You're either inside the budget or you OOM.

So on the Mac, "loads successfully" actually does mean "runs at full speed". The downside is that you don't get a warning when you're approaching the limit — the limit just hits.

## Future Work

- **Context scaling sweep** for Qwen 3.5 9B and Nemotron 4B (small enough to fit at very long contexts). Would map out the speed-vs-context curve at 32K / 64K / 96K / 128K to see if there's any degradation under heavy KV usage on Metal.
- **iogpu.wired_limit override.** macOS allows raising the GPU memory ceiling from the default 75% to ~92% via `sudo sysctl iogpu.wired_limit_mb=33024`. Would let larger models load (e.g. Gemma 4 26B-A4B Q6_K at 32K f16). Untested in this run because it requires a sudo session and a reboot on revert.
- **Flash attention KV cache effects.** All measurements above use `-fa on`. Whether disabling FA changes the working set profile is untested.
