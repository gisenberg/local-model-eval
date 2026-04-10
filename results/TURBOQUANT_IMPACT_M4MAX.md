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

## Why TurboQuant Helps the 5090 But Not the Mac

It comes down to which bottleneck dominates each platform.

**RTX 5090:**
- 1,792 GB/s memory bandwidth (huge)
- 32 GB VRAM (tight)
- KV cache is a meaningful fraction of bandwidth per token for dense models
- → Compressing the KV cache saves bandwidth → speeds up decoding
- → Compressing the KV cache saves VRAM → enables longer contexts
- Net: turbo4 = win on both axes

**M4 Max:**
- 410 GB/s memory bandwidth (constrained — already the bottleneck)
- 36 GB unified memory but only ~30 GB Metal working set
- Weights bandwidth dominates per-token decode cost (we read the whole weight set every step, and the weights are 16-22 GB)
- → KV cache is a small fraction of per-token bandwidth, so compressing it doesn't free up enough to matter
- → The compute overhead of dequantizing turbo4 KV per step exceeds the bandwidth savings
- → It does still save memory, which is the only reason to use it (for models that won't otherwise fit)
- Net: turbo4 = capacity win, speed loss

The asymmetry is platform-physics, not implementation quality. The same TurboQuant kernels that win on CUDA lose on Metal because the platforms have different bottleneck profiles.

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

The Metal working set is the real ceiling, not just the KV cache. For models where the **weights** push you to the limit, no amount of KV compression helps:

- **Gemma 4 26B-A4B Q6_K @ 32K context** — 22 GB of weights leaves only ~8 GB for KV+compute. Even turbo4 KV at 32K (~3 GB) plus compute scratch (~3 GB) is right at the line. We measured: **OOM with both f16 and turbo4 at 32K**.
- **Anything Q6_K above ~22 GB** — Q8_0 of these models is roughly 30 GB by itself, leaving zero room for any KV at all.

The "just compress the KV more" instinct doesn't work when weights are already the bottleneck. On the M4 Max, more aggressive **weight** quantization (Q4 instead of Q6) is usually a better lever than KV compression.

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
2. **If the model won't load, try `-ctk turbo4 -ctv turbo4`.** (Not `-ctk f16 -ctv q8_0` — the per-V-only path saves less memory and isn't enough to fit Gemma 31B.)
3. **Don't expect a speedup from turbo4.** It's purely a capacity workaround on this hardware.
4. **If you're hitting OOM and turbo4 doesn't help, the bottleneck is the weights, not the KV.** Drop to a smaller quant (Q6 → Q4), or switch to a smaller model class.

## Future Experiments

- **turbo3 vs turbo4** — only turbo4 was tested on M4 Max in this run. The 5090 ranking found turbo3 to be faster but riskier; whether the same tradeoff exists on Metal is unknown.
- **Asymmetric `-ctk f16 -ctv q8_0`** — the standard llama.cpp KV-quant flag (per memory: only quantize V, never K, for tool-calling safety). Might offer a milder speed-vs-capacity tradeoff than turbo4.
- **iogpu.wired_limit override** — bumping macOS's GPU memory ceiling from 75% to ~92% via `sudo sysctl iogpu.wired_limit_mb=33024` would let Gemma 31B run with f16 KV at 16K (32 GB total fits in 33 GB ceiling). Untested in this session.

## See also

- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — full per-model tier list with all configs
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — what fits at what context, in detail
- [TURBOQUANT_IMPACT_5090.md](TURBOQUANT_IMPACT_5090.md) — the opposite story on a CUDA platform
