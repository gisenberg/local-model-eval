# RotorQuant on DGX Spark — Pre-experiment Hypothesis

**Date:** 2026-04-11
**Hardware:** NVIDIA DGX Spark (GB10 Blackwell, compute cap 12.1, 128 GB LPDDR5X @ ~273 GB/s)
**Stack:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` (commit `20efe75cf`), CUDA 13.0, sm_121a, aarch64
**Comparator:** mainline llama.cpp results already in [MODEL_RANKINGS_SPARK.md](results/MODEL_RANKINGS_SPARK.md)

RotorQuant ships three symmetric KV cache quantizers (`iso3`, `planar3`, `planar4`, `iso4`) plus a deferred-K mode where K stays FP16 during prefill and only gets quantized post-prefill. The headline claim, from [scrya.com/rotorquant](https://www.scrya.com/rotorquant.pdf), is **better PPL + 28% faster decode + 5× faster prefill than TurboQuant's WHT at the same 10.3× compression ratio**, measured on Llama 3.1 8B Instruct Q4_K_M on an RTX 5090.

We are running it on our **S and A tier MoE models** on the DGX Spark, a radically different host. This document records the prediction *before* we see any results.

## What's different about the Spark

Three facts from our existing [Spark rankings](results/MODEL_RANKINGS_SPARK.md) set the prior:

1. **It is bandwidth-bound by LPDDR5X, not compute.** ~273 GB/s unified memory — 6.5× lower than a 5090. MoE decode saturates at 50–70% of peak bandwidth. The dominant cost per token is reading the active weight set, not attention.

2. **Every S/A tier model on the Spark is a small-active-param MoE with tiny per-token KV cost.** Qwen3.5-122B-A10B uses a DeltaNet hybrid: 36 linear-attention layers + 12 full-attention layers, ~24 KB KV per token. GLM-4.5-Air uses standard attention but with only 12B active of 106B params. For both, the 32K-context KV cache is under ~1 GB — **well under 10% of the per-token bandwidth budget**. There is simply not much KV-cache bandwidth left to save, because there wasn't much to begin with.

3. **Nauful-style V-only q8_0 KV quantization on Spark cost us −46% throughput** with zero quality change (Qwen3.5-122B bartowski, f16K/q8V: 25.8 → 14.0 tok/s at identical 13/17 score). Scalar q8 is the *lightest possible* dequant work — RotorQuant does *more* compute per element (2D Givens or 4D quaternion inverse rotation on top of scalar dequant). The clean negative result on asymmetric q8 is our best data point for what KV quantization *does* on this hardware: compute overhead massively exceeds any bandwidth saving because the KV cache is already tiny and the dequant path competes with bandwidth-bound decode for the same compute units.

## Why the rotorquant paper's 5090 result does not carry over

On the RTX 5090 running dense Llama 3.1 8B, the rotorquant table itself shows:

| Config | Decode tok/s | Δ vs f16 |
|---|---:|---:|
| f16 / f16 | 140 | baseline |
| iso3 / iso3 | 118 | **−16%** |
| planar3 / planar3 | 119 | **−15%** |
| turbo3 / turbo3 | 93 | −34% |
| planar3 / f16 | 134 | **−4%** |

The "+28% faster decode" headline is relative to TurboQuant's `turbo3`, not relative to the f16 baseline. **Even on a 5090, rotorquant is slower than f16 in absolute terms.** Its advertised win is "less slow than the WHT butterfly."

On the Spark, three things make the 5090 comparison optimistic at best:

- **Worse compute-to-bandwidth ratio.** The 5090 has slack compute to absorb dequant overhead cheaply; the Spark does not.
- **MoE active-weight bandwidth dominates.** On the 5090's dense 8B, KV accounts for a larger share of per-token reads, so compression has more room to matter. On Spark MoE, KV is a 10% tax at most.
- **mainline llama.cpp CUDA dequant path is already expensive here.** Our own q8_0 V-cache result is the empirical proof.

## Hypotheses

### H1 (throughput, primary) — symmetric 3-bit is net-negative on Spark

**iso3/iso3 and planar3/planar3 will slow down decode by 20–50%** on both Qwen3.5-122B and GLM-4.5-Air relative to their f16 baselines. Direction: confident. Magnitude: bracketed by the asymmetric q8 result (−46%, V-only) and the 5090 symmetric result (−15%, both). The Spark's compute bottleneck on mainline CUDA should push us closer to the −46% end than the −15% end.

- Qwen3.5-122B bartowski + mainline: 25.8 → **14–20 tok/s** expected
- GLM-4.5-Air: 21.7 → **11–17 tok/s** expected

### H2 (throughput, secondary) — K-only planar3/f16 is the least-bad config

`planar3 / f16` is rotorquant's "zero PPL loss at 5.1× K compression" mode. It halves the dequant burden (no V-side inverse rotation), and because `f16 V` is the native Flash Attention V path, there is no compute added to the V-cache read at all. This should be the config with the **smallest throughput penalty and the lowest quality noise**.

Prediction: within **5–15% slower** than f16 baseline on both models. Not faster — there is no physical mechanism by which compression helps throughput when KV bandwidth is already a rounding error. But close enough to be livable if quality holds.

### H3 (quality) — KV numerical noise will flip code paths, direction unpredictable

At temperature 0 on Spark, we have hard evidence that **kernel-level numerical noise is large enough to steer deterministic generation down completely different code paths**. The same Qwen3.5-122B bartowski file scores:
- **17/17** on ik-llama (different attention/MoE kernels)
- **13/17** on mainline llama.cpp f16 KV
- **8/17** on mainline llama.cpp f16 KV + thinking on
- **14/17** on mainline llama.cpp f16K/q8V asymmetric

That is a ±4-point swing from pure numerical noise, not model differences. RotorQuant inserts extra quantization error (rotate → quantize → dequantize → inverse rotate) at every KV read, which is a *bigger* perturbation than the q8_0 scalar case. We should expect:

- **Qwen3.5-122B bartowski + mainline + iso3**: could land anywhere in 8–16/17. Direction unknowable. If it beats the 13/17 baseline it is *not* because rotorquant improved the model — it would just mean the noise happened to steer onto a cleaner path.
- **GLM-4.5-Air + iso3**: similar noise band around 15/17 — probably 11–17/17.

The **only defensible quality claim** we can make pre-experiment: rotorquant on Spark at temp 0 is **high-variance** and any single-shot result should be treated as a sample of one, not a quality judgment on the technique. A temp=0.3 best-of-3 would be needed for a real comparison, and that is not what we are running here.

### H4 (combined verdict) — net-negative on Spark for MoE

Throughput will be worse. Quality will be noise-dominated. **Rotorquant's target hardware is the 5090, not the Spark**, and its target model class is dense-ish with meaningful KV cost, not small-active-param MoE. The technique is sound; the hardware and model class we are testing are the wrong fit for it.

### H5 (where rotorquant *would* matter on Spark — not in this experiment)

The real Spark use case for rotorquant is not S/A tier. It is **enabling long contexts on the F-tier dense models** that otherwise drown in KV cache bandwidth:

- **Gemma 4 31B Q8_0 (F-tier, 6.7 tok/s)** has ~870 KB/token KV. At 128K context that's 114 GB of KV cache alone — it cannot fit. 10.3× compression drops that to ~11 GB, which *does* fit, and because Gemma is already bandwidth-saturated (81% of peak), the dequant overhead has less headroom to hurt.
- Dense 70B models in the 40+ GB weight class, same argument.

We are **not** running Gemma 31B + rotorquant in this experiment — the user asked for S and A tier. But if this experiment's negative result holds as predicted, the follow-up is not "rotorquant is dead on Spark" — it is "rotorquant belongs on dense long-context workloads on Spark, not MoE decode."

## The experiment

Four runs, all on the 3-benchmark coding suite (Expression Evaluator + A* + LRU-TTL, 17 tests), temp 0, single-shot, 32K context, `-np 1 --no-mmap --jinja`:

| # | Model | KV config | Baseline to compare |
|---|---|---|---|
| 1 | Qwen3.5-122B-A10B Q4_K_M (bartowski) | `iso3 / iso3` | 13/17 @ 25.8 tok/s (mainline f16) |
| 2 | Qwen3.5-122B-A10B Q4_K_M (bartowski) | `planar3 / f16` | same as above |
| 3 | GLM-4.5-Air Q4_K_M (bartowski) | `iso3 / iso3` | 15/17 @ 21.7 tok/s (mainline f16) |
| 4 | GLM-4.5-Air Q4_K_M (bartowski) | `planar3 / f16` | same as above |

Engine: `johndpope/llama-cpp-turboquant` `feature/planarquant-kv-cache @ 20efe75cf`, built for CUDA 13 / sm_121a. Comparison baselines are the same quant file on the same hardware with `-ctk f16 -ctv f16` on mainline llama.cpp — already in the rankings doc, not rebenched.

## Success criteria for "the hypothesis held"

H1 and H4 hold if the iso3 runs land at least 15% below the f16 baseline on **both** models with scores within the noise band. The hypothesis is wrong if any iso3 run **lands at or above the f16 baseline throughput** — which would mean Spark has more compute headroom than the asymmetric q8 result suggested, and the rotate kernels are more efficient than q8_0 dequant despite doing more work. That would be a genuinely surprising finding and worth documenting.

H2 holds if `planar3/f16` K-only beats `iso3/iso3` on throughput on both models.

H3 is unfalsifiable by a single-shot run — we will report scores but interpret them as one draw from a noisy distribution.

## Not tested (explicit scope cuts)

- **ik-llama + rotorquant**: the planarquant kernels only exist in the johndpope mainline fork. Comparing bartowski-on-mainline-planar3 to the S-tier bartowski-on-ik-llama-f16 result would confound engine choice with quantization. We stay on mainline for a clean comparison.
- **Gemma 31B long-context (the real use case)**: out of scope for this request but noted above as the logical follow-up.
- **4-bit variants (planar4/iso4)**: rotorquant's own docs say the symmetric-4bit FA dispatch still crashes on CUDA. Skipping.
- **turbo3/turbo4 comparison**: TurboQuant data already exists for Qwen3.5-122B (the separate experiment) and rotorquant is strictly expected to beat turbo3 per the paper. Independent axis.
- **PPL measurement**: would take hours per config and would not distinguish the numerical-noise effect from a real quality signal at temp 0.
