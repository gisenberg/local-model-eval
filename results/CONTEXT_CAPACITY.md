# Context Capacity Analysis

Tested April 2026 on RTX 5090 32GB.

## The Key Constraint: KV Cache

Context capacity is NOT limited by model weights — it's limited by the KV cache, which is pre-allocated at model load time. The formula:

```
KV bytes/token = 2 (K+V) x num_attn_layers x num_kv_heads x head_dim x 2 (FP16)
```

Hybrid architectures (Mamba, DeltaNet, sliding window) dramatically reduce this because only attention layers need KV cache.

## Architecture Comparison

| Model | Total Layers | Attention Layers | KV Heads | Head Dim | KV/Token | Architecture |
|---|---|---|---|---|---|---|
| nemotron-3-nano-4b | 42 | 4 | 8 | 128 | ~16 KB | Mamba-2 hybrid (4 attn) |
| qwen3.5-35b-a3b | 40 | 10 | 2 | 256 | ~20 KB | DeltaNet hybrid (10 attn) |
| nemotron-3-nano | 52 | ~23 | 2 | 128 | ~23 KB | Mamba-2 hybrid |
| gemma-4-26b-a4b | 30 | 15 global + 15 sliding | 8 | 256 | ~120 KB (global only) | Sliding window + global attn |
| qwen3.5-9b | 32 | 32 (all) | 4 | 256 | ~128 KB | Dense transformer |

The Qwen 35B has the smallest per-token KV cost despite being the largest model, because DeltaNet layers need no KV cache.

## Verified Context Loading (RTX 5090 32GB)

Each model was loaded at increasing context sizes and tested for successful generation:

| Model | 32K | 64K | 112K | 192K | 256K |
|---|---|---|---|---|---|
| qwen3.5-35b-a3b | OK | OK | OK | OK | **OK** |
| qwen3.5-35b-a3b (parallel=1) | OK | OK | OK | OK | **OK** |
| gemma-4-26b-a4b (Q6_K) | OK | OK | OK | OK | OK* |

*Gemma loads at 256K but UI estimates 41 GB — exceeds 32 GB VRAM, spills to RAM.

## Throughput vs Context Size (Measured)

This is the critical test — "loads successfully" does not mean "fits in VRAM":

### Gemma 4 26B Q6_K
```
  32K: 152.6 tok/s
  64K: 150.6 tok/s
 112K: 150.1 tok/s
 192K: 145.2 tok/s    <-- slight degradation, ~4GB spill to RAM
 256K:  54.9 tok/s    <-- severe cliff, ~9GB spill to RAM
```

### Qwen 3.5 35B-A3B
```
  32K:  70.9 tok/s
  64K:  69.9 tok/s
 112K:  69.4 tok/s
 192K:  69.5 tok/s
 256K:  72.2 tok/s    <-- no degradation, fully in VRAM
```

**Key insight**: Qwen 35B fits entirely in VRAM at 256K despite heavier weights (22.1 vs ~20 GB) because its KV cache per token is 6x smaller than Gemma's.

## Why RAM Spill Isn't Always Visible

At 192K, Gemma's UI estimates 36.4 GB (4 GB over VRAM), yet throughput only drops ~5%. This is because:

1. The **pre-allocated** KV buffer spills to RAM, but on short prompts the model never accesses the distant portions
2. Only **actively used** KV cache regions cause PCIe round-trips
3. At 256K (9 GB over), enough actively-used memory is in RAM that every token generation pays the penalty

**Rule of thumb**: if your actual prompt + output stays under the in-VRAM portion of the KV cache, you won't see degradation even with some spill. But filling the context window will hit the cliff.

## Future: TurboQuant KV Cache Compression

TurboQuant (arXiv:2504.19874) compresses KV cache to 3.5 bits/channel with zero quality loss (verified on needle-in-a-haystack up to 104K tokens). This would:

- **Gemma at 256K**: drops from ~41 GB to ~27 GB VRAM — fits entirely on GPU at full speed
- **70B models at Q3**: become viable with ~45K usable context
- **All models**: could run at full trained context length without RAM spill

Requires a llama.cpp fork with TurboQuant KV cache support.

## Parallel Slots and VRAM

LM Studio defaults to `parallel=4`, allocating separate KV cache per slot. In our testing, this didn't significantly change VRAM usage at the context sizes we tested, but at maximum context it could multiply the KV allocation by up to 4x. Setting `parallel=1` is recommended when maximizing context for single-user workloads.

## Tuning Experiments

Tested parallel slots, eval_batch_size, and KV cache quantization parameters across 3 models:

| Parameter | Effect on Throughput | Effect on Quality | Notes |
|---|---|---|---|
| parallel (1 vs 4) | <1% difference | Different output at temp=0 | FP accumulation order changes token selection |
| eval_batch_size (512/1024/2048) | <1% difference | No effect | Only matters for long prompt ingestion (>10K tokens) |
| KV cache quant (q8/q4) | Not testable | Not testable | LM Studio API doesn't expose `cache_type_k`/`cache_type_v` |
