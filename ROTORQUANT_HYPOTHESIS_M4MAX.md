# RotorQuant on M4 Max — Pre-experiment Hypothesis

**Date:** 2026-04-11
**Hardware:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X @ 410 GB/s, ~30 GB Metal working set
**Stack:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` built for Metal (`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`)
**Comparator:** our own measured M4 Max baselines in [results/MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md)

RotorQuant ships four symmetric KV cache quantizers (`iso3`, `planar3`, `planar4`, `iso4`) and an asymmetric/K-only `planar3 / f16` mode with deferred K-quantization (K stays FP16 during prefill, compresses post-prefill). The technique is Givens 2D or quaternion 4D block-diagonal rotations with ~64× fewer FMAs than TurboQuant's Walsh-Hadamard butterfly — [scrya.com/rotorquant paper](https://www.scrya.com/rotorquant.pdf) claims "better PPL, 28% faster decode, 5× faster prefill" than TurboQuant at the same 10.3× compression ratio, measured on Llama 3.1 8B Q4_K_M on an RTX 5090.

We're running it on our **S and A tier models from MODEL_RANKINGS_M4MAX.md**, on a piece of hardware where we have a surprising prior: **TurboQuant's turbo4 KV was *slower* than f16 KV** in our measurements. This document is the prediction before we see any results.

## What's different about the M4 Max

Three findings from [results/MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md), [results/CONTEXT_CAPACITY_M4MAX.md](results/CONTEXT_CAPACITY_M4MAX.md), and [results/TURBOQUANT_IMPACT_M4MAX.md](results/TURBOQUANT_IMPACT_M4MAX.md) set the prior:

1. **TurboQuant turbo4 KV is slower than f16 KV on Metal.** Measured, not projected: Gemma 4 26B-A4B Q6_K at 16K context runs at 60.3 tok/s with f16 KV and 46.0 tok/s with turbo4 KV on the same machine. **−23% throughput for a KV format that's supposed to save bandwidth.** The 5090 sees the opposite sign on this comparison. Our working explanation is that Apple Silicon is bandwidth-constrained on a different axis than CUDA — the dominant per-token bandwidth cost is reading the ~22 GB of active weights, so the dequantization compute overhead on each KV read exceeds the bandwidth savings, and the bigger the dequant work per element the worse the tradeoff. **This is the central finding rotorquant is being tested against.**

2. **The KV cache is <1% of the working set on our S/A tier models.** Measured breakdown for Gemma 4 26B-A4B Q6_K at 32K context with turbo4 KV:
   - Weights: 21,574 MiB (~72%)
   - **KV cache: 250 MiB (<1%)**
   - Compute buffer: 8,402 MiB (~28%)
   - Total: 30,244 MiB → OOM without `-b 2048 -ub 256` workaround

   Everything we're testing is a small-active-param MoE or a GQA dense model. The KV cache was never meaningful memory pressure to begin with. There is very little for a KV quantizer to meaningfully save.

3. **The compute buffer, not the KV cache, is the working-set bottleneck.** Context capacity on M4 Max is determined by `~0.5 × n_ubatch × n_ctx / 1024 MiB` for sliding-window architectures like Gemma 4. Halving `n_ubatch` from the default 512 to 256 frees ~4 GB of working set and unlocks 32K context for Gemma 4 26B-A4B Q6_K. **No amount of KV compression helps with the compute buffer.** RotorQuant cannot fix this problem.

There is one important exception to (2) and (3): **Gemma 4 31B-IT (dense, full attention)**. Its f16 KV at 16K context is ~14 GB on top of 18 GB of weights, blowing the 30 GB working set. It is the one model in our benchmark suite where the KV cache itself is a first-class line item. Today we work around this with `-ctk turbo4 -ctv turbo4`, which fits but costs the 23% throughput hit we saw on Gemma 4 26B-A4B. **Gemma 31B is the cleanest test case for rotorquant on Metal** — it's the model where we genuinely need KV compression AND where a lighter dequant kernel would pay off most.

## Why the rotorquant paper's 5090 result doesn't straightforwardly carry over

The rotorquant README's headline table is the 5090 result:

| Config | Decode tok/s | Δ vs f16 |
|---|---:|---:|
| f16 / f16 | 140 | baseline |
| iso3 / iso3 | 118 | **−16%** |
| planar3 / planar3 | 119 | **−15%** |
| turbo3 / turbo3 | 93 | −34% |
| planar3 / f16 | 134 | −4% |

The "+28% faster decode" headline is planar3 vs turbo3, not planar3 vs f16. **Even on a 5090, rotorquant is slower than f16 in absolute terms.** Its advertised win is "less slow than the WHT butterfly." Same structure as the Spark hypothesis document.

On the M4 Max, the extrapolation could go either way:

- **Pessimistic direction (same story as Spark):** Apple Silicon has even less compute headroom for dequant per element than the 5090. Our empirical turbo4 data point (−23% vs f16) is already worse than the 5090's turbo3 data point (−34%) in *relative recovery room*. Rotorquant's compute savings over turbo might not be enough to close that gap on Metal.

- **Optimistic direction (the interesting one):** Our turbo4 result was −23% on Metal. If rotorquant's 64× reduction in FMAs-per-dequant produces proportional relief, planar3 could recover much of that gap — not necessarily beating f16, but landing close enough to be useful. **This is the most interesting possible finding** because it would invert the "TurboQuant is a speed loss on Metal" conclusion we documented earlier. The mechanism is plausible: turbo4's slowdown on Metal is compute-bound on the dequant kernel; planar3 does ~64× less compute per element; therefore less of the slowdown should transfer.

The direction is genuinely uncertain. Unlike the Spark case where we had strong priors that KV quantization is net-negative, on M4 Max we have a mechanism by which rotorquant specifically could be net-positive relative to turbo4 and possibly competitive with f16.

## Metal backend caveats (the unknown unknown)

The [rotorquant CLAUDE.md file](../../../scrya-com/rotorquant/CLAUDE.md) lists **"Port symmetric V dequant fix to Metal backend (Mac M4)"** as an open TODO. This is the `6e5a4aa` fix that took CUDA PPL from 15,369 down to 7.05 by adding the inverse Givens/quaternion rotation on V dequant. **If Metal doesn't have this fix, symmetric configs (`planar3/planar3`, `iso3/iso3`) may produce broken output or catastrophic PPL on M4 Max** — the same way early-CUDA rotorquant produced garbage before the fix landed.

Practical implication: **the K-only `planar3 / f16` config is far more likely to work on Metal** than the symmetric configs, because the K-only path doesn't need the V inverse rotation at all. Our experiment plan has to be able to gracefully handle symmetric modes producing broken output.

We'll find out empirically whether the Metal path is in a usable state. Either way, this is a data point rotorquant itself could use (the paper team would want to know).

## Hypotheses

### H1 (throughput, primary) — planar3 could recover most of the turbo4 slowdown

On Gemma 4 26B-A4B Q6_K at 16K context (the model where we have a clean f16 vs turbo4 baseline):

- **f16/f16**: 60.3 tok/s (measured baseline)
- **turbo4/turbo4**: 46.0 tok/s (measured, −23%)
- **planar3/planar3**: **projected 52-58 tok/s** (−4% to −14% vs f16; 13-27% improvement over turbo4)
- **planar3/f16**: **projected 56-60 tok/s** (−1% to −7% vs f16)

Direction: modestly confident on the *order* (planar3/f16 > planar3/planar3 > turbo4 > [if symmetric Metal is broken, planar3/planar3 = 0]). Magnitude: much less confident.

**This is falsifiable.** If any rotorquant config lands between 46 and 60 tok/s on this model at 16K context, H1 is supported. If it lands below 46, rotorquant is worse than turbo4 on Metal and our "compute overhead dominates" explanation is wrong (or the Metal kernel quality is poor). If it lands above 60, rotorquant is beating f16 on Metal, which would be the strongest possible result.

### H2 (context) — RotorQuant does NOT fix the `-ub 256` problem, but it unlocks new context tiers on dense models where KV cache IS the bottleneck

The compute buffer is sized by `n_ubatch × n_ctx`, independent of KV format. KV savings from planar3 (maybe 200 MiB at 32K on the Gemma 4 26B-A4B class) are two orders of magnitude smaller than the compute buffer (4.2 GB at `-ub 256` 32K). **Gemma 4 26B-A4B Q6_K at 32K will still need `-ub 256`** regardless of KV format. We should test at 32K with `-ub 256` to confirm this isn't accidentally affected.

**But** — and this is the part we missed initially — the KV cache IS a meaningful share of the working set on the *dense* models in our lineup, and those are the ones where the Metal working set ceiling bites hardest today. Rotorquant's 10.3× KV compression (same as turbo4) unlocks real context headroom there.

#### Per-model context math (projections)

Memory budget: ~28,753 MiB of free Metal working set (`recommendedMaxWorkingSetSize = 30,150 MB` minus macOS overhead). For each model, the per-token KV cost + compute buffer + weights must stay under this.

| Model | Weights | KV/tok (f16) | Compute @ `-ub 256` | **f16 max ctx** | **turbo4 max ctx** | **planar3 max ctx** |
|---|---|---|---|---|---|---|
| Nemotron 3 Nano 4B Q4 | 2.8 GB | ~16 KB | scales | 32K+ | 32K+ | 32K+ |
| Qwen 3.5 9B Q4 | 5.5 GB | ~128 KB | scales | 32K+ | 32K+ | 32K+ |
| **Gemma 4 26B-A4B Q6_K** (MoE) | 22.6 GB | ~120 KB global-only (sliding window for others) | **dominates** | 32K (measured) | 32K (measured, slower) | 32K (no gain) |
| **Gemma 4 26B-A4B Q4_K_M** (MoE) | 16.5 GB | same as Q6 | scales | 32K (measured) | ~48K projected | ~48K projected (no material improvement over turbo4) |
| **Qwen 3.5 27B Opus-Distilled Q4** (dense GQA) | 16.5 GB | ~150 KB | scales | **~32K** (measured) | ~48K projected | **~128K projected with `-ub 128`** |
| **Gemma 4 31B-IT Q4_K_M** (dense full attention) | 18.3 GB | ~870 KB | scales | **won't fit at 16K** | 32K (measured) | **~48K at `-ub 256`, ~64K at `-ub 128` projected** |

Math for the two headline cases:

**Gemma 4 31B-IT Q4_K_M**:
- Weights: 18.3 GB fixed
- f16 KV at 16K: 870 KB × 16,384 = 13.9 GB — plus 18.3 weights + 4.2 compute = 36.4 GB → OOM (matches our earlier measurement)
- turbo4 KV at 32K: ~2.7 GB — plus weights + compute (at `-ub 256`) = ~25.2 GB → fits (our current config)
- **planar3 KV at 32K**: ~2.7 GB (same ratio as turbo4) → fits at the same budget as turbo4
- **planar3 KV at 48K, `-ub 256`**: 4.1 GB KV + 6.3 GB compute + 18.3 weights = **28.7 GB — just under the free budget**
- **planar3 KV at 64K, `-ub 128`**: 5.4 GB KV + 4.2 GB compute + 18.3 weights = **27.9 GB — comfortable**

Practical gain for Gemma 31B: **32K → 48-64K** if rotorquant Metal backend works. That's enough to fit a realistic coding task with 40 KB of repo context plus room for a diff response, at a model where quality is measured-best (17/17).

**Qwen 3.5 27B Opus-Distilled Q4_K_M**:
- Weights: 16.5 GB fixed
- f16 KV at 32K: ~4.8 GB — plus 16.5 + 4.2 = 25.5 GB → fits (our current config)
- f16 KV at 64K, `-ub 128`: 9.6 + 4.2 + 16.5 = 30.3 GB → OOM (over by 1.5 GB)
- **planar3 KV at 64K, `-ub 128`**: 0.96 GB KV + 4.2 + 16.5 = **21.7 GB — huge headroom**
- **planar3 KV at 128K, `-ub 128`**: 1.92 + 8.4 + 16.5 = **26.8 GB — fits comfortably**
- **planar3 KV at 256K, `-ub 64`**: 3.84 + 8.4 + 16.5 = **28.7 GB — at the edge**

Practical gain for Qwen 27B Opus: **32K → 128K-256K**, a 4-8× expansion of effective context. This is the biggest context win in the entire M4 Max model lineup if rotorquant works — and it's the model where we have a clean f16 baseline to compare against.

#### What rotorquant does NOT help with

- **Gemma 4 26B-A4B** (either Q6_K or Q4_K_M): the KV cache is <1% of the working set because sliding window + MoE. Compression saves literally nothing. Context is bounded by compute buffer, which is unrelated to KV format. Rotorquant is context-neutral here.
- **Any small model (≤9B)**: already has 20+ GB of headroom at 32K. Rotorquant doesn't move the ceiling.
- **Any model above ~22 GB of weights**: compute buffer is already bleeding into the margin. Compressing KV saves 1-2 GB at best, not enough to buy a context tier.

#### Summary table (context tiers if rotorquant works on Metal)

| Model | Current max ctx | With planar3 (projected) | Practical win? |
|---|---|---|---|
| Nemotron 4B | 32K | 32K | No (already huge) |
| Qwen 3.5 9B | 32K | 32K | No (already huge) |
| Gemma 4 26B-A4B Q6_K | 32K | 32K | **No** (compute buffer, not KV) |
| Gemma 4 26B-A4B Q4_K_M | 32K | 32K-48K | Minor |
| Qwen 3.5 27B Opus Q4 | 32K | **128K-256K** | **Yes, biggest win** |
| Gemma 4 31B-IT Q4 | 32K (turbo4) | **48-64K** | **Yes, with quality trade** |

**The actionable story**: rotorquant's context benefit on M4 Max is concentrated in the **dense 27-31B weight class**, which is also the class where we measured the "KV cache actually matters" effect. Everything else is either already big enough or bounded by something rotorquant can't touch.

### H3 (the Gemma 31B case) — planar3/iso3 as turbo4 replacement

Gemma 4 31B-IT Q4_K_M is the only S-tier model on M4 Max where we genuinely need KV compression to fit. Today: 18 GB weights + 14 GB f16 KV @ 16K = 32 GB > 30 GB working set → OOM. Workaround: turbo4 KV → 18 GB + ~3.5 GB = ~22 GB, fits, runs at 11.8 tok/s.

Projection for rotorquant on Gemma 31B:

- **If Metal symmetric works**: `iso3/iso3` or `planar3/planar3` should match turbo4's compression (10.3× vs turbo4's 9.1×) and beat its throughput. **Projection: 13-15 tok/s** (10-30% improvement over turbo4's 11.8).
- **If Metal symmetric is broken (TODO item from CLAUDE.md)**: `planar3/f16` won't fit because f16 V is still 7 GB on top of planar3 K at ~3 GB. Total: 18 + 3 + 7 = 28 GB — might fit with `-ub 256`, or might not.
- **If neither symmetric nor K-only fits**: Gemma 31B on Metal stays turbo4-only. This would be the "Metal port isn't ready" finding.

This is the highest-stakes test. Gemma 31B at 11.8 tok/s is borderline unusable today; pushing it to 14+ tok/s would genuinely change the daily-use story for "run a top-quality model on a laptop."

### H4 (quality) — noise band similar to Spark's, may have M4-specific flavor

Our M4 Max experiments so far have **not** surfaced the same kernel-numerical-noise-steers-generation-paths effect the Spark showed (where Qwen3.5-122B scored 17/17 on ik-llama and 13/17 on mainline at the same temp 0). The reason is we haven't swapped engines on M4 Max yet — everything has been the same turboquant-fork llama-server with different flags. Rotorquant brings a different set of kernels from the same fork family, but with a different branch and a potentially unfixed Metal V-dequant path.

Expected quality outcomes at temp 0 on our existing code benchmarks:

- **planar3/f16 on Gemma 4 26B-A4B Q6_K** (K-only, lowest perturbation): scores should land in the 14-16/17 band (current f16 is 15/17).
- **planar3/planar3 on Gemma 4 26B-A4B Q6_K** (symmetric): if Metal V-dequant fix is missing, this could score anywhere from 0/17 (garbage output) to 16/17 (noise-favored path). Unfalsifiable at n=1.
- **Gemma 4 31B-IT**: the 17/17 score is unusually robust (hit at multiple configs). Single-test deviations would most likely be single-shot variance rather than a rotorquant signal.

Nothing we can conclude from a single run. We report the scores as "one draw from a noisy distribution, not a quality judgment on the technique."

### H5 (Qwen 27B Opus-Distilled) — least interesting M4 Max test

Qwen 3.5 27B Opus-Distilled Q4_K_M runs at 13 tok/s on M4 Max (dense, GQA, 32K f16 context). KV at 32K is ~5 GB — not the compute buffer bottleneck, not the KV cache bottleneck. It's purely bandwidth-bound on weights. **Rotorquant shouldn't meaningfully move this number in either direction**, because KV reads aren't a meaningful share of per-token bandwidth and the weights are the dominant cost.

Including it in the experiment set anyway for completeness, with low expectations of a signal. Projection: within ±5% of the 13 tok/s baseline.

### H6 (the Metal port readiness) — real probability this is the actual finding

The strongest prediction this document can make is meta: **there is a real chance the Metal backend in the planarquant branch is not in a working state for symmetric configs on Gemma 4 sliding-window architectures**. Gemma 4's attention pattern has 5 global + 25 sliding-window layers with different KV head counts (8 or 2) per layer — not a simple dense attention pattern.

The risk hierarchy, from lowest to highest risk:

1. **`planar3 / f16` on Qwen 27B Opus-Distilled** (dense, GQA, no sliding window): lowest risk. Most likely to just work.
2. **`planar3 / f16` on Gemma 4 26B-A4B Q6_K** (sliding window + global): moderate risk. K-only sidesteps the V-dequant issue but the K dequant still has to handle the alternating attention pattern.
3. **`planar3 / planar3` on Gemma 4 26B-A4B Q6_K**: higher risk. Needs the V-inverse-rotation fix on Metal which is a known TODO.
4. **`planar3 / planar3` on Gemma 4 31B-IT** (dense full attention, no sliding window): moderate risk on rotation mechanics, but the model-level fit is the best case.

We'll run the low-risk cases first to establish a working baseline, then move to the higher-risk symmetric configs. If everything crashes, **the contribution of this experiment is "the Metal port needs work — here's what fails"**, which is itself useful data for the rotorquant authors.

## The experiment

Four primary runs + one backup, same 3-benchmark coding suite (Expression Evaluator + A* Pathfinding + LRU Cache with TTL, 17 tests total), temp 0, single-shot, `-np 1 --jinja --reasoning-budget 0`. Contexts chosen to match our existing baselines.

| # | Model | Context | KV config | Baseline to compare | Expected risk |
|---|---|---|---|---|---|
| 1 | Gemma 4 26B-A4B Q6_K | 16K | `planar3 / f16` | 15/17 @ 60.3 tok/s (f16/f16) | Low |
| 2 | Gemma 4 26B-A4B Q6_K | 16K | `planar3 / planar3` | same, and 16/17 @ 46 tok/s (turbo4/turbo4) | Medium (Metal V-dequant fix TODO) |
| 3 | Gemma 4 31B-IT Q4_K_M | 16K | `planar3 / planar3` | 17/17 @ 11.8 tok/s (turbo4/turbo4) | Medium — and highest payoff if it works |
| 4 | Qwen 3.5 27B Opus-Distilled Q4_K_M | 32K | `planar3 / f16` | 11/17 @ 13 tok/s (f16/f16) | Lowest risk, lowest interest |
| 5 (fallback) | Gemma 4 26B-A4B Q6_K | 16K | `iso3 / iso3` | if planar3/planar3 crashes — same model, different rotation block size | — |

Engine: `johndpope/llama-cpp-turboquant` at `feature/planarquant-kv-cache`, built for Metal (`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`). Comparison baselines are from [results/MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md) — already in the rankings doc, not rebenched.

We'll use the same `tools/m4max_bench.py` script with a new list of configs, the `-b 2048 -ub 256` workaround where needed for 32K runs, and write outputs to `experiments/rotorquant_m4max_bench/`.

## Success criteria for "the hypothesis held"

- **H1** holds if `planar3/planar3` on Gemma 4 26B-A4B Q6_K at 16K lands in the 52-58 tok/s band (net positive vs turbo4, net negative vs f16). Falsified below 46 or above 60.
- **H2** holds trivially if Gemma 26B-A4B Q6_K at 32K still OOMs without `-ub 256`. Only way to falsify is if rotorquant somehow fixes the compute buffer, which would be surprising.
- **H3** (Gemma 31B) holds if `planar3/planar3` beats turbo4's 11.8 tok/s at the same 17/17 quality. This is the highest-value positive outcome.
- **H6** (Metal port readiness) is validated if symmetric configs crash, produce zero-score output, or the server fails to launch. That outcome is *also* useful — it tells the rotorquant authors exactly what needs to be ported.

## Not tested (explicit scope cuts)

- **MLX + rotorquant**: rotorquant's kernels are in the llama.cpp fork family, not in `mlx_lm`. No direct comparison there.
- **Dense 70B or larger**: we haven't benchmarked those on M4 Max (they don't fit), so no baseline to compare against.
- **TurboQuant turbo3 comparison**: turbo3 wasn't in our original M4 Max benchmarks (we jumped straight to turbo4 because the rankings said that's the production config). We'd need to rebench the turbo3 baseline for a clean comparison. Out of scope.
- **4-bit rotorquant variants (`planar4/planar4`, `iso4/iso4`)**: the rotorquant README and CLAUDE.md both call out that 4-bit symmetric FA dispatch still crashes on CUDA. Metal is almost certainly worse. Skipping.
- **PPL measurement**: would take hours per config and would not distinguish numerical-noise effects from real quality signal at temp 0. Code-benchmark scores are a noisier but faster signal.
- **Multi-run variance**: single-shot only, same as the existing M4 Max benchmarks. A temp=0.3 best-of-3 for each config would take ~12× longer and isn't in the budget for this experiment.

## See also

- [ROTORQUANT_HYPOTHESIS.md](ROTORQUANT_HYPOTHESIS.md) — the DGX Spark version of this document (written first). That one predicts rotorquant is net-negative on Spark MoE; this one is genuinely uncertain for M4 Max because of the turbo4-is-slower-than-f16 prior.
- [results/MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md) — baseline measurements we're comparing against.
- [results/TURBOQUANT_IMPACT_M4MAX.md](results/TURBOQUANT_IMPACT_M4MAX.md) — the underlying story for *why* we expect rotorquant to have a chance on Metal where turbo4 didn't.
- [results/CONTEXT_CAPACITY_M4MAX.md](results/CONTEXT_CAPACITY_M4MAX.md) — the compute-buffer finding that bounds what any KV quantizer can achieve here.
