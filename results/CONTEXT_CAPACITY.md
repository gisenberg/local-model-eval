# Context capacity — cross-platform

Context capacity is **not** bounded by model weights — it's bounded by the KV cache + compute buffer, both pre-allocated at model load time. What constrains you depends on your platform:

| Platform | Dominant constraint | Failure mode | First lever to reach more context |
|---|---|---|---|
| **RTX 5090** (32 GB GDDR7) | KV cache fitting in VRAM | **Silent spill to system RAM via PCIe** — throughput crashes when context actually fills | KV compression (TurboQuant turbo4) |
| **M4 Max** (36 GB unified, ~30 GB usable) | Compute buffer (much bigger than KV on Metal) | **Hard OOM, no spill** — allocator fails, server returns 500 | Lower `-ub` (physical batch size) |

**Shared formula** for per-token KV cost (applies to every architecture that has a KV cache):

```
KV bytes/token = 2 (K+V) × num_attn_layers × num_kv_heads × head_dim × 2 (FP16)
```

Hybrid architectures (Mamba, DeltaNet, sliding window attention) dramatically reduce this because non-attention layers don't allocate a KV cache. The Qwen 3.5 35B-A3B has the smallest per-token KV cost in our lineup despite being the largest model, because DeltaNet layers need no KV cache.

## Architecture reference

| Model | Total / attn layers | KV heads | Head dim | KV/token (f16) | Notes |
|---|---|---|---|---|---|
| nemotron-3-nano-4b | 42 / 4 | 8 | 128 | ~16 KB | Mamba-2 hybrid (4 attn) |
| qwen3.5-35b-a3b | 40 / 10 | 2 | 256 | ~20 KB | DeltaNet hybrid (10 attn) |
| nemotron-3-nano | 52 / ~23 | 2 | 128 | ~23 KB | Mamba-2 hybrid |
| qwen3.5-9b | 32 / 32 | 4 | 256 | ~128 KB | Dense transformer |
| gemma-4-26b-a4b | 30 / 15 global + 15 sliding | 8 | 256 | ~120 KB global only | Sliding window + global |
| qwen3.5-27b-opus | 32 / 32 | 8 | 256 | ~256 KB | Dense GQA |
| gemma-4-31b-it | 62 / 10 global + 52 SWA | 4-16 | 256-512 | ~22 KB turbo4 average† | ISWA (interleaved SW attn) |

† Original math said 870 KB/token f16 assuming all 60 layers use full attention. Real number is ~22 KB/token turbo4 averaged because only 10 layers are non-SWA with ~4 shared-KV heads each — see [Gemma 4 31B correction](#correction-gemma-4-31b-it-kv-math-was-off-by-10x).

---

# RTX 5090 (32 GB GDDR7)

The platform has a graceful-but-dangerous failure mode: models at large contexts silently spill KV cache to system RAM. They load successfully and run at full speed on short prompts — only when the context actually fills do you pay the PCIe round-trip penalty.

## Verified context loading

Each model loaded at increasing context sizes, tested for successful generation:

| Model | 32K | 64K | 112K | 192K | 256K |
|---|---|---|---|---|---|
| qwen3.5-35b-a3b | ✅ | ✅ | ✅ | ✅ | **✅** |
| qwen3.5-35b-a3b (parallel=1) | ✅ | ✅ | ✅ | ✅ | **✅** |
| gemma-4-26b-a4b Q6_K | ✅ | ✅ | ✅ | ✅ | ✅† |

† Gemma loads at 256K but estimates 41 GB total — exceeds 32 GB VRAM, portions spill to system RAM.

## Throughput vs context (the real test)

"Loads successfully" does not mean "fits in VRAM":

**Gemma 4 26B-A4B Q6_K** — f16 KV, default ubatch:
```
  32K: 152.6 tok/s
  64K: 150.6 tok/s
 112K: 150.1 tok/s
 192K: 145.2 tok/s    ← slight degradation, ~4 GB spill to RAM
 256K:  54.9 tok/s    ← severe cliff, ~9 GB spill to RAM
```

**Qwen 3.5 35B-A3B** — f16 KV, default ubatch:
```
  32K:  70.9 tok/s
  64K:  69.9 tok/s
 112K:  69.4 tok/s
 192K:  69.5 tok/s
 256K:  72.2 tok/s    ← no degradation, fully in VRAM
```

Qwen 35B fits entirely in VRAM at 256K despite heavier weights (22.1 vs 20 GB) because its KV cache per token is 6× smaller than Gemma's.

## Why RAM spill isn't always visible

At 192K Gemma estimates 36.4 GB (4 GB over VRAM), yet throughput only drops ~5%. Reasons:

1. The pre-allocated KV buffer spills to RAM, but on short prompts the model never accesses the distant portions.
2. Only **actively used** KV regions cause PCIe round-trips.
3. At 256K (9 GB over VRAM), enough actively-used memory is in RAM that every token generation pays the penalty.

**Rule of thumb**: if your actual prompt + output stays within the in-VRAM portion of the KV cache, you won't see degradation even with some spill. But filling the context window will hit the cliff.

## Parallel slots and VRAM

LM Studio defaults to `parallel=4`, allocating separate KV cache per slot. At our tested context sizes this didn't significantly change VRAM, but at maximum context it can multiply KV allocation by up to 4×. Set `parallel=1` when maximizing context for single-user workloads.

## TurboQuant KV compression as the first lever

[TurboQuant](TURBOQUANT.md) turbo4 compresses KV cache to 3.5 bits/channel at +0.23% PPL. What it unlocks on the 5090:

- **Gemma 4 26B-A4B Q6_K at 256K**: drops from ~41 GB to ~27 GB VRAM — fits entirely on GPU at full speed.
- **Gemma 4 31B-IT Q4_K_M**: fits the full 262K native window at 28 GB with 4 GB headroom.
- **70B models at Q3**: become viable with ~45K usable context.

Requires llama.cpp's TurboQuant fork. Details in [TURBOQUANT.md](TURBOQUANT.md).

## Tuning experiments

Parallel slots, eval_batch_size, and KV cache quantization tested on 3 models:

| Parameter | Effect on throughput | Effect on quality | Notes |
|---|---|---|---|
| parallel (1 vs 4) | <1% | **Different output at temp=0** | FP accumulation order changes token selection |
| eval_batch_size (512 / 1024 / 2048) | <1% | None | Only matters for long prompt ingestion (>10K tokens) |
| KV cache quant (q8 / q4) | Not testable via LM Studio | — | LM Studio API doesn't expose `cache_type_k` / `cache_type_v` |

---

# MacBook Pro M4 Max (36 GB unified)

> **⚠ Update 2026-04-11 (after rotorquant rebase)**: Much of the original analysis below was **base-version specific** to the turboquant fork commit `8590cbff9`, NOT a fundamental Metal property. Specifically:
>
> 1. **The "compute buffer scales at ~260 MiB per 1024 tokens for sliding-window models" formula was a bug in the old base**, not a Metal characteristic. A newer llama.cpp base (via cherry-picking upstream PR #21309 for Gemma 4 support) has compute buffer ~523 MiB at 32K regardless of `n_ubatch` or model — a **16× reduction**. The `-ub 256` workaround documented below is unnecessary on the newer base.
> 2. **The claim "Gemma 4 31B-IT has ~870 KB/token KV at f16" was wrong math.** Assumed every layer of the 62-layer network uses full attention; actually Gemma 4 31B uses ISWA where only ~9 layers are global attention. Measured actual KV at 32K f16 is ~3.7 GB total (~120 KB/token averaged), not 27.8 GB.
> 3. **Turbo4 KV was never "mandatory" for Gemma 31B** — it was mandatory on the old base because of the compute-buffer bug, not because of KV math. On the new base, Gemma 31B f16 fits at 64K with default `-ub`, and turbo4 at 128K+.
>
> **What's still true**: the Metal working set ceiling IS ~30 GB (not 36 GB), OOMs are still OOM-not-spill, there's no PCIe fallback. The per-token KV costs for specific architectures are correct (except the Gemma 4 31B row which was based on bad math). The "what fits at what context" measurements are accurate for the old-base build at the time of testing. But the **mechanism** attributed to them (compute-buffer-scales-with-`n_ubatch`) was a base-version-specific artifact.
>
> **On a current llama.cpp base**: don't worry about `-ub 256`. Use default flags and the old-base ceilings below work out to **~4× as many tokens** at the same memory budget.
>
> Updated numbers for Gemma 4 31B-IT on the new base:
>
> | Config | 32K | 64K | 128K | 256K |
> |---|---|---|---|---|
> | f16 / f16 (default ub) | ✅ 21.7 GB | ✅ 24.3 GB | ❌ 29.4 GB (OOM by ~700 MB) | — |
> | planar3 / f16 K-only (default ub) | ✅ 20.9 GB | — | ❌ 30.5 GB (deferred quant — no allocation savings) | — |
> | turbo4 / turbo4 (default ub) | ✅ ~18.5 GB | ✅ — | ✅ 21.0 GB | ✅ 24.1 GB |
>
> Full story in [ROTORQUANT.md](ROTORQUANT.md).

## The defining constraint (still true): Metal working set ceiling ~30 GB

The Mac advertises 36 GB of unified memory but the GPU can only address ~30 GB. `recommendedMaxWorkingSetSize` from Metal device init is ~30,150 MB on this binning; macOS reserves the rest for system, display, and other Metal clients.

```
ggml_metal_device_init: recommendedMaxWorkingSetSize = 30150.67 MB
```

This 30 GB is the **hard ceiling** for `weights + KV cache + compute buffers + scratch`, summed across all loaded models. Going over does **not** produce graceful spillover (like the 5090). It produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` and every chat completion returns `500 Compute error`.

## What couldn't be tested on Metal

- **Weight size > ~25 GB** at any context. Examples: Gemma 4 31B-IT Q6_K, Qwen 3.5 27B Q8_0, Gemma 4 26B-A4B Q8_0. Metal working set ceiling is the hard constraint.
- **Long-context scaling sweeps** (32K → 64K → 112K → 192K → 256K) like the 5090 page. On M4 Max most models at 32K already use a significant fraction of the budget; 64K+ is moot for anything bigger than ~9B dense or the small-active MoEs.

## Why the Mac doesn't have a spill cliff

The 5090's famous failure mode — model loads fine, runs at full speed on short prompts, drops from 150 to 55 tok/s when context fills — doesn't exist here because:

1. **No PCIe.** Memory is unified; GPU and CPU share the same physical pool. No remote bus to spill across.
2. **Metal working set is a hard limit, not a soft budget.** macOS doesn't transparently page GPU buffers in and out of system RAM. You're inside the budget or you OOM.

On Mac, "loads successfully" actually does mean "runs at full speed." Downside: no warning when you're approaching the limit — it just hits.

---

## Original old-base analysis (historical, preserved — see correction above)

The sections below document the pre-rebase mechanism where the compute buffer scaled at ~0.5 × `n_ubatch` × `n_ctx` / 1024 MiB for sliding-window models, which was a base-version bug rather than an Apple-Silicon property. Specific numbers are real measurements from that pinned commit; current llama.cpp doesn't exhibit this.

### Compute buffer dominated the 30 GB budget (old base)

For Gemma 4 26B-A4B Q6_K with turbo4 KV at 32K context with default `-ub 512`:

```
load_tensors:  MTL0_Mapped model buffer size  = 21,574 MiB    ← weights
llama_kv_cache:    MTL0 KV buffer size         =    170 MiB   ← turbo4 KV (TINY)
llama_kv_cache:    MTL0 KV buffer size         =     80 MiB   ← second buffer
sched_reserve:     MTL0 compute buffer size    =  8,402 MiB   ← DOMINANT cost
```

KV cache was 170 MiB — three orders of magnitude smaller than the compute buffer. Compressing KV further saved nothing meaningful.

### Lower `-ub` was the first lever (old base)

With default `-ub 512`, Gemma 4 26B-A4B Q6_K needed 8.4 GB compute buffer at 32K. With `-ub 256`, it needed 4.2 GB. Same model, same context, same throughput, half the memory pressure.

| Config | Compute buffer at 32K | Total | Result |
|---|---|---|---|
| `-ub 512` (default) | 8,402 MiB | 30,141 MiB | ❌ OOM |
| `-ub 256` | 4,211 MiB | 26,672 MiB | ✅ comfortable |
| `-ub 128` | 2,121 MiB | 24,584 MiB | ✅ more headroom |

Scaling was linear in both `n_ubatch` and `n_ctx`: `compute_buf_MiB ≈ 0.5 × n_ubatch × n_ctx / 1024` for sliding-window models. To push to 64K: halve ubatch → `-ub 128`. To push to 128K: halve again → `-ub 64`.

**`-ub` is physical batch size, not `-b` (prompt length).** A 2000-token prompt submitted with `-b 2048 -ub 256` is chunked into 8 micro-batches of 256 and processed sequentially; with `-ub 512` it's 4 chunks. Prefill is ~2× slower at max-length prompts with `-ub 256`; decode throughput is unchanged. For 200-400 token coding prompts this is invisible.

### n_ubatch sweep at 32K context (old base)

| n_ubatch | Compute buffer | Total projected | Fits? |
|---|---|---|---|
| 1 | 18 MiB | 22,482 MiB | ✅ but n_batch=1 caps prompts at 1 token |
| 64 | 1,077 MiB | 23,541 MiB | ✅ |
| 128 | 2,121 MiB | 24,584 MiB | ✅ |
| **256** | **4,210 MiB** | **26,672 MiB** | ✅ recommended |
| 512 (default) | 8,389 MiB | 30,898 MiB | ❌ OOM |
| 1024 | 16,749 MiB | 39,353 MiB | ❌ way over |

### n_ctx sweep at default `-ub 512` (old base)

| Context | KV cache | Compute buffer | Total | Fits? |
|---|---|---|---|---|
| 16K | 165 MiB | 4,242 MiB | ~26.0 GB | ✅ |
| 20K | 212 MiB | 5,282 MiB | ~27.1 GB | ✅ |
| 24K | 254 MiB | 6,322 MiB | ~28.1 GB | ❌ |
| 32K | 339 MiB | 8,402 MiB | ~30.2 GB | ❌ |

### What fit at what context on the old base

Each row is a measurement from M4 Max benchmarks. ❌ = OOM during inference, ✅ = generates successfully.

| Model | Quant | KV | Weights | Ctx | `-ub` | Result |
|---|---|---|---|---|---|---|
| Nemotron 3 Nano 4B | Q4_K_M | f16 | 2.8 GB | 32K | default | ✅ 65 tok/s |
| Qwen 3.5 9B | Q4_K_M | f16 | 5.5 GB | 32K | default | ✅ 35 tok/s |
| Gemma 4 26B-A4B | Q4_K_M | f16 | 16.5 GB | 32K | default | ✅ 59 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 16K | default | ✅ 60 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 32K | **256** | ✅ 61 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 64K | **128** | ✅ (4.2 GB compute buf, 27.3 GB total) |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 32K | default 512 | ❌ OOM (compute buf 8.4 GB) |
| Gemma 4 26B-A4B | Q6_K | turbo4 | 22.6 GB | 32K | **256** | ✅ 45 tok/s (slower than f16) |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | f16 | 16.5 GB | 32K | default | ✅ 13 tok/s |
| Gemma 4 31B-IT | Q4_K_M | f16 | 18.3 GB | 16K | any | ❌ OOM (but see correction below) |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 32K | **256** | ✅ 11.8 tok/s |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 64K | 256 | ❌ OOM at compute buf 16.7 GB (would need `-ub 128`) |

### Correction: Gemma 4 31B-IT KV math was off by 10×

Original: "KV bytes/token = 2 × 62 layers × 16 KV heads × 256 head_dim × 2 = 870 KB." That assumed every layer uses full attention.

Reality: Gemma 4 31B uses **ISWA** (interleaved sliding-window attention). Only ~10 of 62 layers are global; the rest are SWA at a fixed 1536-cell window. From the live gemma4 loader on the rebased fork, non-SWA layers have ~4 shared-KV heads each. Real per-token cost averaged across all layers at turbo4 is **~22 KB/token** (measured from a VRAM sweep at 96K / 160K / 262K), not 870 KB.

At 262K context, turbo4 KV is 5.6 GB — well within the budget. Gemma 4 31B fits the full native window at 28 GB total on a 32 GB card, not the "~58K ceiling" the bad math predicted.

### Why the 5090 didn't have this problem in the first place

The 5090 uses 25,636 MB total VRAM for the same Gemma 4 26B-A4B Q6_K turbo4 at 32K with default settings. M4 Max projected 30,244 MiB at the same config. Delta: ~4.6 GB, all in the compute buffer. Plausible reasons:

- cuBLAS/cuDNN kernels can stream tensors through working memory in smaller chunks; Metal Performance Shaders may pre-allocate full intermediate tensors.
- Flash attention on CUDA has been optimized for memory efficiency over years; Metal's FA path is newer.
- Gemma 4's sliding-window pattern may need re-materialization buffers on Metal that CUDA avoids via different kernel fusion.

Exact root cause would require backend profiling. The practical implication for old-base users: you had to actively reduce `n_ubatch` to recover the context window the 5090 got for free at default settings. On the new base this asymmetry is much smaller.

## TurboQuant on M4 Max: capacity yes, speed no

Per [TURBOQUANT.md](TURBOQUANT.md): turbo4 KV is *slower* than f16 KV on Metal (the dequant compute overhead exceeds bandwidth savings) but does buy real capacity on models where KV is a first-class line item:

- Gemma 4 31B-IT Q4_K_M @ 16K (old base): f16 = 32 GB = OOM, turbo4 = 22 GB = ✅
- Gemma 4 26B-A4B Q6_K @ 16K: f16 = 26 GB ✅, turbo4 = 24 GB ✅ (no benefit from turbo4)
- Gemma 4 26B-A4B Q6_K @ 32K: f16 = 32 GB ❌, turbo4 = 28 GB ❌ (still over — weights are 22 GB)

**Rule**: use turbo4 if and only if the model literally won't load with f16 KV. No speed benefit on this platform.

## Future experiments (M4 Max)

- **Context scaling sweep** for Qwen 3.5 9B and Nemotron 4B (small enough to fit at very long contexts). Would map the speed-vs-context curve at 32K / 64K / 96K / 128K to check for any degradation under heavy KV usage on Metal.
- **`iogpu.wired_limit` override.** `sudo sysctl iogpu.wired_limit_mb=33024` raises GPU memory ceiling from default 75% to ~92%. Would let larger models load (e.g. Gemma 4 26B-A4B Q6_K at 32K f16). Untested (requires sudo + reboot on revert).
- **Flash-attention off profile.** All measurements use `-fa on`. Whether disabling FA changes the working set profile is untested.

## See also

- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — full 5090 tier list
- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — full M4 Max tier list
- [TURBOQUANT.md](TURBOQUANT.md) — KV compression across platforms
- [ROTORQUANT.md](ROTORQUANT.md) — K-only Givens-rotation KV quantizer
