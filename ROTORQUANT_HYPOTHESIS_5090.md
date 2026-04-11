# RotorQuant on RTX 5090 — Pre-experiment Hypothesis

**Date:** 2026-04-11
**Hardware:** NVIDIA RTX 5090 32 GB GDDR7 @ 1,792 GB/s, Blackwell sm_120a, Windows 11 + CUDA 13.2
**Stack:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` (commit `20efe75cf`)
**Comparator:** our existing 5090 llama.cpp+TurboQuant results in [MODEL_RANKINGS_5090.md](results/MODEL_RANKINGS_5090.md) (re-measured from WSL2 Linux client, April 10)

> **Hard limitation discovered during build:** the johndpope fork is based on an older upstream llama.cpp commit (April 1, 2026) that **does not have Gemma 4 architecture support**. Loading any `gemma4` GGUF crashes with `unknown model architecture: 'gemma4'`. This eliminates 3 of our 6 originally-planned models (Gemma 26B-A4B Q6_K, Gemma 26B-A4B Q4_K_M, Gemma 31B-IT Q4_K_M) — including the **one model where rotorquant would have materially mattered**, Gemma 31B-IT for the +99K context unlock described in H5. This is recorded here rather than buried in the results writeup because it significantly changes what this experiment can answer. Actual tests run only on the Qwen-family 27B models. Also fought four MSVC build issues that needed patches to compile on Windows at all: `M_PI` not defined in two `.c` files, one `extern` in C++ inside function scope, one symbol needing `GGML_API` for cross-DLL export, one `extern bool g_innerq_finalized` cross-DLL reference that had to be stubbed out because Windows won't cross-link non-dllexport symbols. None of these change behavior for the non-Gemma models we actually tested.

RotorQuant ships two symmetric KV cache quantizers: `iso3` (4D quaternion rotation) and `planar3` (2D Givens rotation), plus asymmetric modes where K is rotated/quantized and V stays at f16 or q8_0. The headline claim, from [scrya.com/rotorquant](https://www.scrya.com/rotorquant.pdf), is **better PPL + 28% faster decode + 5× faster prefill than TurboQuant's WHT at the same 10.3× compression ratio**, measured on Llama 3.1 8B Instruct Q4_K_M on an RTX 5090.

The headline "28% faster" is relative to TurboQuant's `turbo3`, NOT relative to f16 baseline. The paper's own numbers show rotorquant is *slower* than f16 in absolute terms (iso3: −16%, planar3: −15%, turbo3: −34%). The win is "less slow than the WHT butterfly."

We're testing it on the **same hardware class the paper targeted** (Blackwell 5090), so unlike the Spark hypothesis we are **not** predicting a complete collapse. But our model class is different from the paper's: we run **26-31B MoE and dense models**, not dense 8B. That's a meaningful shift.

## What's different about our 5090 setup vs the paper's

Three facts from our existing [5090 rankings](results/MODEL_RANKINGS_5090.md) set the prior:

1. **We already use TurboQuant turbo4 as standard, not f16.** The paper compares rotorquant vs f16 and vs turbo3. Our published baselines are all **TurboQuant turbo4 KV**, which sits between the two on the compression ↔ quality tradeoff (3.8× compression vs turbo3's 5.1× and rotorquant iso3's 10.3×). The right comparison for us is: does rotorquant beat turbo4, not whether it beats f16. The paper is silent on turbo4.

2. **Every 5090 S/A tier model is in the 26-31B class.** The paper tested dense 8B. For a dense 8B, KV cache per token is a meaningful share of per-token memory traffic (~5% by our math). For dense 31B, KV is a larger fraction (~10%) AND the dequant cost per token is the same — so the compression savings have more room to matter. For MoE 26B-A4B, KV is a smaller fraction (~3%) because MoE decode reads only ~4B of active weights per token but still reads the full KV cache.

3. **The 5090 is compute-rich, not bandwidth-bound.** On our measurements, Gemma 26B-A4B Q6_K decode at 139 tok/s uses ~3.3 GB/token × 139 = ~460 GB/s of the 1,792 GB/s available bandwidth — only 26% utilization. There's slack headroom for dequant overhead to hide in, which is exactly what makes the paper's 5090 results less dire than they'd be on the bandwidth-bound Spark.

## Why the paper's Llama 3.1 8B result may or may not transfer

The paper's key 5090 data point (from our reading of the rotorquant README):

| Config (K/V) | Decode tok/s | vs f16 |
|---|---:|---:|
| f16 / f16 | 140 | baseline |
| **iso3 / iso3** | **118** | **−16%** |
| **planar3 / planar3** | **119** | **−15%** |
| turbo3 / turbo3 | 93 | −34% |
| **planar3 / f16** | **134** | **−4%** |

If those numbers scale linearly to our models:
- Gemma 26B-A4B Q6_K (139 tok/s baseline): expect ~117 with iso3, ~133 with planar3/f16
- Gemma 31B-IT Q4_K_M (50 tok/s baseline): expect ~42 with iso3, ~48 with planar3/f16
- Qwen 35B-A3B Q4_K_M (174 tok/s baseline): expect ~146 with iso3, ~167 with planar3/f16

But there are reasons to expect *non*-linear behavior:
- **Bigger models have different bandwidth math.** Dense 31B reads ~14 GB/token of weights. 8B reads ~4 GB/token. KV cost is a bigger slice for the smaller model.
- **Turbo4 vs f16 as baseline matters.** The paper compared rotorquant to f16 (their baseline). We'd be comparing to **turbo4** (our baseline), which already does 3.8× compression. The incremental win going from turbo4 → iso3 (10.3×) is smaller than going from f16 → iso3.
- **The paper only tested one dense 8B model.** MoE architectures have completely different per-token memory patterns and the rotorquant results may not generalize.

## Hypotheses

### H1 (throughput, primary) — iso3/iso3 will be 10-20% slower than f16, 20-30% slower than turbo4

The paper's 5090 measurement (−16% vs f16) should roughly hold for our models *compared to f16*. Compared to our **turbo4** baseline, rotorquant will look worse because turbo4 is already faster than f16 in most cases due to better cache locality and less total memory traffic.

Predictions (decode tok/s, compared to our published turbo4 results):

| Model | turbo4 baseline | iso3/iso3 prediction | Delta |
|---|---:|---:|---:|
| Gemma 4 26B-A4B Q6_K | 138.9 | **105-115** | −20 to −24% |
| Gemma 4 26B-A4B Q4_K_M | 149.5 | **115-125** | −18 to −23% |
| Gemma 4 31B-IT Q4_K_M | 50.3 | **38-43** | −15 to −25% |
| Qwen 3.5 27B Opus-Distilled | 60.0 | **46-52** | −13 to −23% |
| Qwopus 3.5 27B-v3 Q6_K | 49.6 | **38-43** | −14 to −23% |
| Harmonic 27B Q4_K_M | 61.3 | **47-53** | −14 to −23% |

I'm **fairly confident** on direction (slower) and **moderately confident** on magnitude. The range is set by the paper's −16% on dense 8B as the floor and our expectation that bigger models amplify the dequant overhead.

The hypothesis is **wrong** if any iso3 run lands within 5% of the turbo4 baseline. That would mean rotorquant's 2D Givens dequant is faster than turbo4's WHT butterfly on 5090 kernels — plausible per the paper's claims, but surprising if it holds for 26-31B models.

### H2 (throughput, secondary) — planar3/f16 is the sweet spot on 5090, not iso3/iso3

`planar3 / f16` halves the dequant burden (K-side only, V stays native). The paper measured −4% vs f16 on 8B for this config. On our 5090 this should be the **best throughput config** among rotorquant options, possibly landing within 5% of turbo4/turbo4.

Predictions:
- Gemma 26B-A4B Q6_K: **130-140 tok/s** (within 10% of turbo4)
- Gemma 31B-IT Q4_K_M: **46-50 tok/s** (within 10% of turbo4)

If planar3/f16 beats turbo4 on any model, that's genuinely interesting and worth a closer look at kernel choice. I don't expect it to, because turbo4 is also a low-overhead config (3.8× compression, well-tuned WHT).

### H3 (quality) — rotorquant will be within noise on the 5090

The 5090 has enough compute that numerical noise is less of a concern than on Spark. At temp=0 the paper measured PPL within 5-6% of f16 on Llama 3.1 8B for iso3/iso3. Our temp=0.3 multi-run suite is **more forgiving** than temp=0 PPL — it averages out sampling variance.

Predictions for best-of-3 coding scores (our standard metric):

| Model | turbo4 baseline | iso3/iso3 prediction | planar3/f16 prediction |
|---|---:|---:|---:|
| Gemma 26B-A4B Q6_K | 30/31 | **28-31/31** | **29-31/31** |
| Gemma 31B-IT Q4_K_M | 31/31 | **28-31/31** | **30-31/31** |
| Qwen 27B Opus-Distilled | 31/31 | **27-31/31** | **29-31/31** |

I expect **within 2-3 tests of turbo4** for iso3/iso3, **within 1-2 tests** for planar3/f16. The failure mode I'm watching for: Harmonic 27B specifically needs stable thinking traces and is sensitive to KV perturbation. If any model regresses hard, Harmonic is the most likely candidate.

The hypothesis is **wrong** if best-of-3 drops more than 4 tests below the turbo4 baseline on any model. That would suggest the extra quantization error from rotation + scalar quant is materially worse than turbo4's WHT on these specific architectures.

### H4 (combined verdict) — turbo4 remains the default, rotorquant is an option for long context

For single-stream coding on the 5090, **turbo4 will stay the default** because it's faster AND the quality is equivalent at the compression ratio we need. Rotorquant's advantage kicks in only when:

1. You need the 10.3× compression ratio (e.g., 200K+ context on dense 31B) that turbo4's 3.8× can't reach
2. You're in a regime where the WHT butterfly kernel is actively slow (CPU, certain embedded GPUs) — not our case

For our workload, **iso3/iso3 is strictly worse than turbo4** and **planar3/f16 is at best marginally competitive**. This is the opposite of the Spark hypothesis which predicted rotorquant would be net-negative for different reasons (bandwidth-bound, small KV fraction). On the 5090, rotorquant will be net-neutral-to-slightly-negative because we already have a good alternative that rotorquant's paper doesn't measure against.

### H5 (where rotorquant *would* win on 5090 — not in this experiment)

The real 5090 use case for rotorquant is **extending dense Gemma 31B beyond 58K context** (our current turbo4 ceiling). At 10.3× compression, Gemma 31B's 870 KB/token KV drops to ~84 KB/token, which could push the 5090 to 150-200K usable context on that model. Turbo4's 3.8× compression only gets us to ~230 KB/token, capping at 58K.

We are **not** testing long-context rotorquant in this experiment — the user asked for S/A tier coding benchmarks at 32K context, where turbo4 already fits fine. If the short-context experiment shows "rotorquant is 20% slower but quality matches," the follow-up question becomes "is the quality still matched at 150K context where turbo4 can't even fit?" — that's a different and interesting experiment we're deferring.

## Context math per model

The throughput hypothesis is only half the story — the other half is whether rotorquant buys us any *context* we don't already have. The compression ratios of each config:

| Config | Compression vs f16 | Relative to turbo4 |
|---|---|---|
| f16 / f16 | 1.0× | 0.26× |
| planar3 / f16 (K-only) | **~1.67×** effective | **0.44× — worse than turbo4** |
| turbo4 / turbo4 | 3.8× | 1.0× (our baseline) |
| iso3 / iso3 | **10.3×** | **2.71×** |

planar3/f16 is surprising — it compresses K by 5.1× but leaves V unchanged at f16, so the *total* KV cache shrinks by only ~1.67× vs baseline. That is **less** than turbo4's 3.8×, meaning planar3/f16 gives you *less* context headroom than our current standard. It exists purely as a "zero PPL loss" reference config, not as a context-maximizing one.

### Per-model context budget on the 5090

Using our measured turbo4 max context as the anchor, projecting each rotorquant config:

| Model | Native trained | Turbo4 max (measured) | iso3/iso3 projected | planar3/f16 projected | Useful gain |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B Q6_K | 262K | ~230K | **262K** (capped by native) | ~101K | **+32K** |
| Gemma 4 26B-A4B Q4_K_M | 262K | 262K (full) | 262K (already full) | ~115K | **none** |
| Gemma 4 31B-IT Q4_K_M | 262K | **~58K** | **~157K** | ~26K | **+99K** ← only big win |
| Qwen 27B Opus-Distilled | 262K | 262K (full) | 262K (already full) | ~115K | **none** |
| Qwopus 27B Q6_K | 262K | 262K (full) | 262K (already full) | ~115K | **none** |
| Harmonic 27B Q4_K_M | 262K | 262K (full) | 262K (already full) | ~115K | **none** |

**The headline finding (pre-experiment):** **5 of 6 models already hit the native 262K context ceiling with turbo4.** Rotorquant cannot exceed the trained context window regardless of compression ratio. For these models, iso3/iso3 is a pure throughput tax with zero capability upside — it just makes the same context slower.

**Only Gemma 31B-IT materially benefits.** Its dense architecture burns 870 KB/token on the KV cache, so even turbo4 caps it at 58K. iso3/iso3 should unlock ~157K context — a **+99K** gain that takes this model from "medium-context only" to "true long-context". This is the one result in the experiment where a rotorquant win would matter.

**planar3/f16 is negative across the board.** It gives less context than turbo4 AND is slower. The paper's "zero PPL loss" framing is irrelevant when the alternative (turbo4) has roughly the same PPL impact at higher compression. I include it in the experiment only to validate H2 (that it's the least-slow rotorquant config), not because any model benefits from using it in production.

### What this means for the experiment's verdict

The context math changes the question from "is rotorquant faster than turbo4" to "is rotorquant's **context unlock** on Gemma 31B worth the throughput tax?" For the other 5 models, the right answer is "stay on turbo4 regardless of throughput results" — they don't need more context and rotorquant can't give them more anyway.

The specific decision after this experiment:
- **If iso3/iso3 on Gemma 31B lands within 20% of turbo4 throughput** (38-40 tok/s vs 50) with ≤3 test quality drop → iso3 becomes the recommended config for Gemma 31B when you need context >58K, which is the only scenario it helps.
- **If iso3/iso3 is slower than that** → keep using turbo4 at 58K, accept the context ceiling, or switch models if you need longer context (Qwen 27B Opus-Distilled already runs at 262K on turbo4 for the cost of slightly lower quality).
- **For the other 5 models** → keep turbo4. No configuration of rotorquant can improve their status.

## The experiment (revised scope)

Originally planned: 6 S/A tier models × 2 rotorquant configs × 4 benchmarks × 3 runs = 144 runs. **After the Gemma 4 build blocker, actual scope is 3 models × 2 configs × 4 benchmarks × 3 runs = 72 runs.** The 3 Gemma 4 models are excluded entirely because the johndpope fork's base upstream predates Gemma 4 architecture support.

Two configs per testable model, all on the 4-benchmark coding suite (Expression Evaluator + A* + LRU-TTL + String Processor, 22 tests), temp 0.3, **3 runs best-of-3** (matching our existing methodology), 32K context, `-np 1`, thinking off (`-rea off`) unless the model's existing A-tier ranking uses thinking on.

All benchmarks run from a **WSL2 Linux client** hitting the Windows llama-server. This matches our corrected April 10 methodology and avoids the ~2s Python urllib3-on-Windows TTFT bug.

Models (all tested with iso3/iso3 and planar3/f16):

| # | Model | Tier | Baseline (turbo4) | File | Testable? |
|---|---|---|---:|---|---|
| 1 | Gemma 4 26B-A4B Q6_K | S | 138.9 tok/s, 30/31 avg | `gemma-4-26B-A4B-it-Q6_K.gguf` | **No — gemma4 arch unsupported** |
| 2 | Gemma 4 26B-A4B Q4_K_M | A | 149.5 tok/s, n/a | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | **No — gemma4 arch unsupported** |
| 3 | Gemma 4 31B-IT Q4_K_M | S | 50.3 tok/s, 30.7/31 avg | `gemma-4-31B-it-Q4_K_M.gguf` | **No — gemma4 arch unsupported** (the big miss — only model that would gain context from H5) |
| 4 | Qwen 3.5 27B Opus-Distilled Q4_K_M | S | 60.0 tok/s, 25.3/31 avg | `Qwen3.5-27B.Q4_K_M.gguf` | Yes |
| 5 | Qwopus 3.5 27B-v3 Q6_K | A | 49.6 tok/s, 24.0/31 avg | `Qwopus3.5-27B-v3-Q6_K.gguf` | Yes |
| 6 | Harmonic 27B Q4_K_M (thinking on) | A | 61.3 tok/s, 30.7/31 avg | `Harmonic-27B-Q4_K_M.gguf` | Yes |

**Actual runs: 3 models × 2 configs × 4 benchmarks × 3 runs = 72 runs.** ~1-1.5 hours.

Engine: `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache @ 20efe75cf`, built on Windows with CUDA 13.2, Visual Studio 2022 (with four patches noted at the top of this document). Comparison baselines are the published turbo4 results from MODEL_RANKINGS_5090.md, not re-run for this experiment.

### What we can and can't learn

Three dense Qwen-family 27B variants all derived from the same base model test **H1 (throughput penalty)** and **H3 (quality noise)** but don't illuminate **H4/H5** meaningfully. All three already hit 262K native context on turbo4, so the "does rotorquant unlock more context" question is pre-answered as no. The one model where rotorquant could actually prove its worth on the 5090 (Gemma 31B-IT) is the one we can't run. This experiment becomes a test of "does rotorquant hurt on models where it can't help" — which is the strictly less interesting half of the hypothesis.

## Success criteria for "the hypothesis held"

- **H1 holds** if every iso3/iso3 run is within 10-25% slower than its turbo4 baseline. Wrong if any model is faster on iso3/iso3 than turbo4, OR slower by more than 35%.
- **H2 holds** if planar3/f16 throughput is within 5% of turbo4 on every model. Wrong if planar3/f16 is slower than iso3/iso3 on any model (that would imply a dispatch bug in the fork).
- **H3 holds** if best-of-3 scores are within 3 tests of the turbo4 baseline on every model. Wrong if any model drops more than 4 tests or *improves* by more than 2 tests (the improvement would suggest our turbo4 baseline has a subtle issue).
- **H4 holds** if no single (model, rotorquant config) pair beats the turbo4 baseline on **both** throughput and quality simultaneously. Wrong if any configuration becomes the new default for any model.

## Not tested (explicit scope cuts)

- **NVFP4 models (Gemma 31B-IT NVFP4-turbo, Gemma 26B-A4B NVFP4)**: these run on vLLM, not llama.cpp. Rotorquant kernels only exist in the llama.cpp fork.
- **vs TurboQuant turbo3**: we already use turbo4, not turbo3. The paper's "28% faster than turbo3" claim is against a config we don't use and can't directly verify without running a third baseline.
- **Long context (>32K)**: out of scope. 32K is the standard context in our methodology. The "rotorquant for long context" use case is documented in H5 as a follow-up.
- **4-bit variants (planar4/iso4)**: the johndpope README and CLAUDE.md both note the symmetric 4-bit FA dispatch crashes during prefill. Skipping per upstream recommendation.
- **PPL measurement**: our methodology uses coding benchmark pass rates, not PPL. Adding PPL would not distinguish numerical noise from real quality signal at this scale.
- **Qwen 3.5 35B-A3B (C-tier)**: the user asked specifically for S-A tier models. Qwen 35B is C-tier on our ranking despite raw speed because of the LRU Cache capability gap.
