# Context Capacity Analysis — MacBook Pro M4 Max 36 GB

Tested April 2026.

For the RTX 5090 equivalent, see [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md). The Mac story is fundamentally different — capacity is limited by Metal's working set ceiling, not VRAM, and there's no graceful degradation when you go over.

## The Defining Constraint: Metal's Compute Buffer (Not the KV Cache!)

The Mac advertises 36 GB of unified memory but **the GPU can only address ~30 GB** of it. This is `recommendedMaxWorkingSetSize` from Metal device init, ~30150 MB on this binning. macOS reserves the rest for the system, the display, and other Metal clients.

```
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 30150.67 MB
```

This 30 GB is the **hard ceiling** for `weights + KV cache + compute buffers + scratch`, summed across all loaded models. Going over does NOT produce graceful spillover (like the 5090 does to system RAM via PCIe). It produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` and every chat completion returns `500 Compute error`.

But the surprise is **what dominates that 30 GB budget**. It's not the KV cache. It's the compute buffer.

### The compute buffer is much bigger than the KV cache on Metal

For Gemma 4 26B-A4B Q6_K with turbo4 KV at 32K context, the actual breakdown llama-server reports on this Mac:

```
load_tensors:  MTL0_Mapped model buffer size  = 21574.57 MiB    ← weights
llama_kv_cache:    MTL0 KV buffer size        =   170.13 MiB    ← turbo4 KV (TINY)
llama_kv_cache:    MTL0 KV buffer size        =    79.81 MiB    ← second buffer
sched_reserve:     MTL0 compute buffer size   =  8402.37 MiB    ← THIS is the problem
```

The KV cache is **170 MiB**. Three orders of magnitude smaller than the compute buffer. Compressing the KV cache further (turbo4 already gives 3.8x compression) saves nothing meaningful — there's nothing left to compress.

The compute buffer scales nearly linearly with context length:

| Context | KV cache | Compute buffer | Total | Fits? |
|---|---|---|---|---|
| 16K | 165 MiB | 4,242 MiB | ~26.0 GB | ✅ |
| 20K | 212 MiB | 5,282 MiB | ~27.1 GB | ✅ |
| 24K | 254 MiB | 6,322 MiB | ~28.1 GB | ❌ Metal allocator fails |
| 28K | 296 MiB | 7,362 MiB | ~29.2 GB | ❌ |
| 32K | 339 MiB | 8,402 MiB | ~30.2 GB | ❌ |

That's roughly **260 MiB of compute buffer per 1024 tokens** for this architecture. By 64K context the compute buffer alone would be ~16 GB. By 230K (the 5090 ranking's number for this model) it would be ~58 GB.

### Why the 5090 doesn't have this problem

The 5090 ranking shows the same Gemma 4 26B-A4B Q6_K turbo4 model using 25,636 MB *total* VRAM at 32K context. M4 Max projects 30,244 MiB at the same configuration. **The difference is ~4.6 GB, all of it in the compute buffer.**

CUDA's compute buffer for this model class is roughly half the size of Metal's. Possible reasons:

- cuBLAS/cuDNN kernels can stream tensors through working memory in smaller chunks; Metal Performance Shaders may pre-allocate full intermediate tensors.
- Flash attention on CUDA has been heavily optimized for memory efficiency over multiple years; the Metal flash attention path is newer.
- The sliding window attention pattern in Gemma 4 (15 global + 15 sliding layers) may need re-materialization buffers on Metal that CUDA avoids via different kernel fusion.

The exact root cause would require profiling both backends, but the *practical* result is clear: the M4 Max gets 4-5 GB less of usable context budget than the 5090 for the same model, on top of the 2 GB nominal memory difference (32 GB CUDA vs 30 GB Metal).

## What Fits at What Context (Measured)

Each row is a real run from the M4 Max benchmark. ❌ = OOM during inference, ✅ = generates successfully.

| Model | Quant | KV | Weights | Context | Result |
|---|---|---|---|---|---|
| Nemotron 3 Nano 4B | Q4_K_M | f16 | 2.8 GB | 32K | ✅ 65 tok/s |
| Qwen 3.5 9B | Q4_K_M | f16 | 5.5 GB | 32K | ✅ 35 tok/s |
| Gemma 4 26B-A4B | Q4_K_M | f16 | 16.5 GB | 32K | ✅ 59 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 16K | ✅ 60 tok/s (compute buf 4.2 GB) |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 20K | ✅ (compute buf 5.3 GB) |
| Gemma 4 26B-A4B | Q6_K | f16 or turbo4 | 22.6 GB | 24K | ❌ OOM (compute buf 6.3 GB > Metal margin) |
| Gemma 4 26B-A4B | Q6_K | f16 or turbo4 | 22.6 GB | 32K | ❌ OOM (compute buf 8.4 GB) |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | f16 | 16.5 GB | 32K | ✅ 13 tok/s |
| Gemma 4 31B-IT | Q4_K_M | f16 | 18.3 GB | 16K | ❌ OOM (KV cache itself is 14 GB at f16) |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 16K | ✅ 11.5 tok/s (KV → 3.5 GB, fits) |

**Key findings:**

1. **Gemma 4 26B-A4B Q6_K is the most context-constrained S-tier model.** Practical max ~20K context regardless of KV format. The bottleneck is the **compute buffer** (~260 MiB per 1024 tokens of context), which on top of 22 GB of weights leaves no room. The KV cache itself is only 170 MiB even at 32K — turbo4 vs f16 makes essentially no difference here.

2. **Gemma 4 31B-IT only runs with turbo4 KV** — but for a different reason. This is the one model where the KV cache really is the bottleneck. Dense 31B has ~870 KB/token of KV at f16 (vs Qwen's GQA which is ~150 KB/token), so 16K f16 = 14 GB of KV on top of 18 GB of weights = 32 GB > 30 GB ceiling. With turbo4 KV (~3.5 GB), it fits.

3. **Qwen 3.5 27B Opus-Distilled fits at 32K f16** despite being dense and similar size to Gemma 31B, because Qwen 3.5 uses GQA (8 KV heads, not 16) — a 6× smaller per-token KV cost. Same weight class, very different capacity profile. Also: a different compute buffer pattern, since dense models without sliding window don't have the re-materialization overhead.

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
