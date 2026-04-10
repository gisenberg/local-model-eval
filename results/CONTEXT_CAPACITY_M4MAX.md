# Context Capacity Analysis — MacBook Pro M4 Max 36 GB

Tested April 2026.

For the RTX 5090 equivalent, see [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md). The Mac story is fundamentally different — capacity is limited by Metal's working set ceiling, not VRAM, and there's no graceful degradation when you go over.

## The Defining Constraint: Metal Working Set ≠ Unified Memory

The Mac advertises 36 GB of unified memory but **the GPU can only address ~30 GB** of it. This is `recommendedMaxWorkingSetSize` from Metal device init, ~30150 MB on this binning. macOS reserves the rest for the system, the display, and other Metal clients.

```
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 30150.67 MB
```

This 30 GB is the **hard ceiling** for `weights + KV cache + compute buffers + scratch`, summed across all loaded models. Going over does NOT produce graceful spillover (like the 5090 does to system RAM via PCIe). It produces:

```
ggml_metal_synchronize: error: command buffer 0 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
process_ubatch: failed to compute graph, compute status: -1
llama_decode: failed to decode, ret = -3
```

The model loads fine, the server starts, /health returns ok — and then every chat completion returns `500 Compute error`. So you have to know your math up front; you don't get a slow degradation warning.

## What Fits at What Context (Measured)

Each row is a real run from the M4 Max benchmark. ❌ = OOM during inference, ✅ = generates successfully.

| Model | Quant | KV | Weights | Context | Result |
|---|---|---|---|---|---|
| Nemotron 3 Nano 4B | Q4_K_M | f16 | 2.8 GB | 32K | ✅ 65 tok/s |
| Qwen 3.5 9B | Q4_K_M | f16 | 5.5 GB | 32K | ✅ 35 tok/s |
| Gemma 4 26B-A4B | Q4_K_M | f16 | 16.5 GB | 32K | ✅ 59 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 16K | ✅ 60 tok/s |
| Gemma 4 26B-A4B | Q6_K | f16 | 22.6 GB | 32K | ❌ OOM |
| Gemma 4 26B-A4B | Q6_K | turbo4 | 22.6 GB | 16K | ✅ 46 tok/s |
| Gemma 4 26B-A4B | Q6_K | turbo4 | 22.6 GB | 32K | ❌ OOM |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | f16 | 16.5 GB | 32K | ✅ 13 tok/s |
| Gemma 4 31B-IT | Q4_K_M | f16 | 18.3 GB | 16K | ❌ OOM (won't load) |
| Gemma 4 31B-IT | Q4_K_M | turbo4 | 18.3 GB | 16K | ✅ 11.5 tok/s |

**Key findings:**

1. **Gemma 4 26B-A4B Q6_K is the most context-constrained S-tier model.** 22 GB of weights leaves only ~8 GB for KV+compute. 16K f16 fits at ~3 GB KV. 32K f16 needs ~6 GB KV plus a few GB of compute scratch — over the line. **Even turbo4 KV doesn't help at 32K** because the bottleneck is the *weights*, not the KV cache.

2. **Gemma 4 31B-IT only runs with turbo4 KV.** The dense 31B has ~870 KB/token of KV at f16 (vs Qwen's ~150 KB/token with GQA), so 16K f16 = 14 GB of KV on top of 18 GB of weights. That's 32 GB, over the 30 GB limit. With turbo4 KV (~3.5 GB), the total is ~22 GB and it fits.

3. **Qwen 3.5 27B Opus-Distilled fits at 32K f16** despite being dense and similar size to Gemma 31B, because Qwen 3.5 uses GQA (8 KV heads, not 16) — a 6× smaller per-token KV cost. Same weight class, very different capacity profile.

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
