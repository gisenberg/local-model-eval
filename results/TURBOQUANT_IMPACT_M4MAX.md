# TurboQuant Impact: What It Unlocks on M4 Max (36 GB)

The short version: **TurboQuant on Apple Silicon is the opposite of TurboQuant on the 5090.** On the 5090 it's a free win (same quality, more context, sometimes faster). On the M4 Max it costs throughput everywhere except one model where it's the only thing that lets it run at all.

For the 5090 story, see [TURBOQUANT_IMPACT_5090.md](TURBOQUANT_IMPACT_5090.md).

## TL;DR

| Model | Without TurboQuant (f16 KV) | With TurboQuant (turbo4 KV) | Net |
|---|---|---|---|
| **Gemma 4 31B-IT Q4_K_M @ 16K** | OOM (14 GB KV + 18 GB weights = 32 GB > 30 GB limit) | ✅ 11.5 tok/s, 17/17 | **Required to run** |
| Gemma 4 26B-A4B Q6_K @ 16K | ✅ **60 tok/s**, 15/17 | 46 tok/s, 16/17 | **-23% speed**, ~tied quality |
| Gemma 4 26B-A4B Q6_K @ 32K | OOM | OOM | No help (weights are the bottleneck) |
| Anything else we tested | f16 fits at full context | turbo4 hurts speed | **don't use turbo4** |

**The decision rule on this hardware:** use turbo4 KV if and only if the model literally won't load with f16. There's no other use case where it's the right choice.

## Why TurboQuant Helps the 5090 But Mostly Not the Mac

Two platform asymmetries combine to make turbo4 a much weaker tool on Metal than on CUDA.

### Asymmetry 1: Bandwidth bottleneck (speed)

**RTX 5090:** 1,792 GB/s bandwidth, 32 GB VRAM. KV cache is a meaningful fraction of per-token bandwidth for dense models. Compressing the KV cache saves bandwidth → faster decoding.

**M4 Max:** 410 GB/s bandwidth, ~30 GB working set. Weights bandwidth (16-22 GB read every token) already dominates the decode budget. KV cache is a small fraction; compressing it doesn't free up enough bandwidth to matter, and the dequantization compute overhead actively hurts. Measured: -23% speed on Gemma 26B-A4B Q6_K (60 tok/s f16 → 46 tok/s turbo4 at the same 16K context).

### Asymmetry 2: Compute buffer dominates the working set (capacity)

This is the bigger surprise. On Metal, the **compute buffer** is the dominant working-set cost for context — much bigger than the KV cache itself. From an actual run of Gemma 4 26B-A4B Q6_K turbo4 at 32K:

```
weights:           21,574 MiB
KV cache (turbo4):    250 MiB    ← 0.8% of the budget
compute buffer:     8,402 MiB    ← 28% of the budget
total:             30,244 MiB    → exceeds 28,753 MiB free, OOMs
```

The KV cache is **less than 1%** of the working set. **Compressing the KV cache further saves you almost nothing** — there's nothing left to compress. Any context-capacity benefit from turbo4 is in the noise.

Compare this to the 5090, which according to its ranking uses 25,636 MiB *total* VRAM for the same model at 32K. **Metal's compute buffer is roughly 4-5 GB bigger than CUDA's** for this architecture. That's why the 5090 hits 230K context on this model and the M4 Max can't even hit 24K — it has nothing to do with the KV cache, which both platforms keep tiny under turbo4.

**The result:** turbo4 helps the M4 Max in exactly one scenario (the small fraction of cases where the *KV cache itself* is a meaningful share of memory — i.e. dense models with full attention and many KV heads, like Gemma 4 31B-IT), and is a pure speed loss everywhere else. Most of the time, what's eating your memory budget is the compute buffer, which turbo4 cannot help with.

## The One Case Where TurboQuant Is Mandatory: Gemma 4 31B-IT

Gemma 4 31B-IT is dense (no GQA, no sliding window — full attention on every layer). Per-token KV cost at f16 is enormous:

```
KV bytes/token = 2 (K+V) × 62 layers × 16 KV heads × 256 head_dim × 2 (FP16) ≈ 870 KB
```

At 16K context that's **13.6 GB of KV cache**. Plus 18 GB of Q4_K_M weights. Total: 32 GB. The Metal working set ceiling is ~30 GB. **It won't load**.

| Config | Weights | KV @ 16K | Total | Fits? |
|---|---|---|---|---|
| Q4_K_M f16 | 18 GB | 13.6 GB | **31.6 GB** | ❌ OOM |
| Q4_K_M turbo4 | 18 GB | 3.5 GB | **21.5 GB** | ✅ |
| Q4_K_M f16 @ 8K (stretch) | 18 GB | 6.8 GB | 24.8 GB | ✅ but tiny ctx |

So for Gemma 4 31B-IT specifically, TurboQuant turbo4 is the difference between "best dense model on this Mac, 17/17 quality, 11.5 tok/s" and "doesn't run at all". That's worth the 23% speed cost on this one model.

The 5090 ranking has this same observation but with different framing: turbo4 there extends Gemma 31B from 16K → 58K context. Same root cause (enormous per-token KV), different solution profile.

## What TurboQuant Doesn't Save You From

For most M4 Max OOMs, the **compute buffer** is the bottleneck, not the KV cache. TurboQuant compresses the KV cache; it does nothing for the compute buffer.

- **Gemma 4 26B-A4B Q6_K @ 24K+ context** — 22 GB of weights + 6+ GB of compute buffer is already at the limit. The KV cache (with or without turbo4) adds <1 GB at 32K, which is the noise floor. **Both f16 and turbo4 OOM at 24K and beyond.** Measured.
- **Anything Q6_K above ~22 GB at non-trivial context** — Q8_0 of these models is roughly 30 GB by itself, leaving zero room for compute or KV.
- **Long contexts (>32K) on any ≥26B model** — the compute buffer scales linearly at ~260 MiB per 1024 tokens for sliding-window architectures. By 64K it's 16 GB on top of weights. Turbo4 doesn't help.

The "just compress the KV more" instinct is a CUDA habit that doesn't transfer. On the M4 Max, more aggressive **weight** quantization (Q4 instead of Q6) is usually a better lever — it frees up ~6 GB of memory by shrinking the dominant cost (the weights), which makes room for the second-largest cost (the compute buffer). KV compression is fighting over the smallest line item.

## The Speed Cost in Numbers

The only direct apples-to-apples comparison we have is Gemma 4 26B-A4B Q6_K at 16K context, both KV configs:

| Config | Weight read/token | KV read/token (rough) | Tok/s | % of 410 GB/s peak |
|---|---|---|---|---|
| f16 KV | ~4 GB (MoE 4B-active) | ~3 GB | **60.3** | ~88% util |
| turbo4 KV | ~4 GB | ~0.8 GB | 46.0 | ~54% util |

Same model, same prompts, same context — the only difference is the KV cache format. f16 is 31% faster and hits 88% bandwidth utilization. turbo4 drops to 54% because the dequant compute eats more cycles than the smaller KV reads save in bandwidth.

**Quality** in this comparison: f16 = 15/17, turbo4 = 16/17. Single-shot single-test variance, not a real quality difference.

## Practical Rule of Thumb

When picking a KV format on this Mac:

1. **Default to `-ctk f16 -ctv f16`.**
2. **First lever for OOMs is `-ub 256`, not turbo4 KV.** The compute buffer is usually the actual bottleneck, not the KV cache. Halving `-ub` (the physical batch size) cuts the compute buffer in half with zero decode-throughput cost. See [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) for the full sweep.
3. **Use turbo4 KV only when the KV cache itself is the bottleneck.** That's basically just Gemma 4 31B-IT, where dense full attention puts the KV at 14 GB at 16K f16. For most other models, KV is <1% of the working set on Metal — turbo4 saves nothing meaningful and slows you down ~25%.
4. **Don't expect a speedup from turbo4 on Apple Silicon.** It's purely a capacity workaround when needed.
5. **If you're hitting OOM and `-ub 256` + turbo4 don't help, the bottleneck is the weights themselves.** Drop to a smaller quant (Q6 → Q4), or switch to a smaller model class.

## Future Experiments

- **turbo3 vs turbo4** — only turbo4 was tested on M4 Max in this run. The 5090 ranking found turbo3 to be faster but riskier; whether the same tradeoff exists on Metal is unknown.
- **Asymmetric `-ctk f16 -ctv q8_0`** — the standard llama.cpp KV-quant flag (per memory: only quantize V, never K, for tool-calling safety). Might offer a milder speed-vs-capacity tradeoff than turbo4.
- **iogpu.wired_limit override** — bumping macOS's GPU memory ceiling from 75% to ~92% via `sudo sysctl iogpu.wired_limit_mb=33024` would let Gemma 31B run with f16 KV at 16K (32 GB total fits in 33 GB ceiling). Untested in this session.

## See also

- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — full per-model tier list with all configs
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — what fits at what context, in detail
- [TURBOQUANT_IMPACT_5090.md](TURBOQUANT_IMPACT_5090.md) — the opposite story on a CUDA platform
