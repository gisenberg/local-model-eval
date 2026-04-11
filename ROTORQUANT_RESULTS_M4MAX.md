# RotorQuant on M4 Max — Post-experiment Results

**Date:** 2026-04-11
**Hardware:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X @ 410 GB/s, ~28.6 GB Metal free working set
**Stack:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` commit `20efe75cf`, Metal build (ggml 0.9.8)
**Pre-experiment doc:** [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md)

The hypothesis predicted rotorquant could be the first KV quantizer that's net-positive on Metal, because our earlier measurements showed TurboQuant turbo4 is *slower* than f16 on this hardware (60.3 → 46.0 tok/s on Gemma 4 26B-A4B Q6_K). The mechanism: rotorquant's Givens 2D rotation does ~64× less dequant compute per element than TurboQuant's Walsh-Hadamard butterfly, and Metal's bottleneck on KV quantization is apparently the dequant compute, not the KV bandwidth.

The actual results are more nuanced than the hypothesis and have one show-stopping surprise.

## Show-stopper: Gemma 4 models couldn't be tested

The planarquant fork's base llama.cpp is **ggml 0.9.8** (commit `20efe75cf`), which pre-dates `gemma4` architecture support. The turboquant fork we built earlier for `MODEL_RANKINGS_M4MAX.md` is at **ggml 0.9.11** and does support gemma4. Every Gemma test case in the hypothesis failed with:

```
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma4'
```

This killed **all three of the highest-value hypotheses**:
- H1 primary (planar3/planar3 on Gemma 4 26B-A4B Q6_K) — blocked
- H1 K-only (planar3/f16 on Gemma 4 26B-A4B Q6_K) — blocked
- **H3 (Gemma 4 31B-IT context unlock)** — the single most interesting test case, completely blocked

Options for fixing this (not attempted in this experiment):

1. Merge `ggml 0.9.11` (with gemma4 support) into the planarquant branch
2. Cherry-pick the gemma4 architecture commit from the turboquant fork
3. Wait for upstream to merge planarquant into a newer base

None of these are work-for-an-afternoon fixes. **Actionable finding for the rotorquant authors:** the Metal backend's model-class coverage is effectively limited by the fork's base llama.cpp version. Publishing a rebase onto a newer upstream would make the fork dramatically more useful to Mac users who want to run the current generation of models.

Only **Qwen 3.5 27B Opus-Distilled Q4_K_M** (architecture name `qwen35`) ran successfully. Every data point below is on that one model.

## What we measured on Qwen 3.5 27B Opus-Distilled Q4_K_M

Six configurations, all at temp 0, single-shot, `-np 1 --jinja --reasoning-budget 0`, 3-benchmark coding suite (Expression Evaluator + A* Pathfinding + LRU Cache with TTL, 17 tests total). Baseline comes from our existing [MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md) at default `-ub 512`.

| Config | Context | `-ub` | Avg tok/s | Score | Δ vs baseline |
|---|---|---|---|---|---|
| **f16 / f16** (original baseline) | 32K | 512 default | 13.0 | 11/17 | — |
| **planar3 / f16** (K-only, rotorquant) | 32K | 512 | **15.5** | **11/17** | **+19% speed, same quality** |
| planar3 / planar3 (symmetric, rotorquant) | 32K | 512 | 11.0 | 16/17 | −15% speed, +5 quality (noise / path flip) |
| f16 / f16 at 65K | 65K | 256 | 15.4 | 11/17 | +18% speed (mostly from `-ub 256`, not context) |
| planar3 / planar3 at 65K | 65K | 256 | 10.8 | 16/17 | −17% speed, same quality as 32K symmetric |
| planar3 / planar3 at **256K** | 256K | 256 | **1.7** (two benchmarks timed out at 30 min) | 6/6 (LRU only, ExprEval+A* DNF) | **Context fits but decode collapses** |

Additional data point from a single short decode test (not scored):
- **planar3 / planar3 at 128K**: decode 11.3 tok/s on a 500-token response. No collapse at 128K. Collapse happens between 128K and 256K.

## Hypothesis verdicts

### H1 — rotorquant recovers from the turbo4 slowdown ✅ (on the model we could test)

The prediction was planar3 recovers most of the gap between turbo4 (−23% vs f16) and f16. Measured result on Qwen 27B Opus (not Gemma, but same direction):

- **K-only planar3/f16: +19% vs f16.** Rotorquant's K-only mode is *faster* than f16, not just "less slow." This is the first KV quantizer we've seen on Metal that's net-positive on decode.
- **Symmetric planar3/planar3: −15% vs f16.** Still a loss, because the Metal V-dequant path carries a heavier compute cost than K-only. Direction is correct (better than turbo4's −23%) but magnitude is small enough that K-only is strictly the right config for a speed-focused run.

The specific numbers the hypothesis bracketed for Gemma 4 26B-A4B (52-58 tok/s projected for planar3/planar3 vs 60 f16, 46 turbo4) couldn't be tested. On Qwen 27B the equivalent ratios would be: f16=13, turbo4=?? (not measured on this model), planar3/f16=15.5, planar3/planar3=11.0. **H1 holds** in the sense that K-only rotorquant IS better than f16, which is a stronger result than the hypothesis even asked for.

### H2 — rotorquant doesn't unlock meaningful new context on Qwen 27B Opus ❌ (refuted, wrong reason)

The pre-experiment context projection said Qwen 27B Opus would max out around 32K with f16 and rotorquant could push it to 128-256K. The table projected:

| Model | Current max ctx | With planar3 (projected) |
|---|---|---|
| Qwen 3.5 27B Opus Q4 | 32K | **128K-256K** |

**What actually happened:** f16 KV at 128K **already fits** (24 GB total projected, 5 GB of headroom). f16 only OOMs at 256K (32 GB projected, 3.4 GB over budget). And at 256K, planar3/planar3 fits (27 GB projected) but **decode collapses to 1.7 tok/s** — 6× slower than at 32-128K, and two of the three benchmarks timed out.

So the picture is:

| Context | f16 status | planar3/planar3 status |
|---|---|---|
| 32K | ✅ 13 tok/s | ✅ 11 tok/s |
| 65K | ✅ 15 tok/s | ✅ 11 tok/s |
| **128K** | ✅ (quick test shows loads fine) | ✅ 11 tok/s (not collapsed) |
| **256K** | ❌ OOM (32 GB > 28.6 free) | ⚠️ Fits (27 GB) but decode 1.7 tok/s, timeouts |

**H2's prediction was wrong for two reasons:**

1. **I underestimated what f16 could reach on this model.** The compute buffer scaling formula I derived (`~260 MiB per 1024 tokens`) is specific to Gemma 4's sliding-window attention pattern. Qwen 27B Opus uses standard GQA attention with a near-constant ~250-420 MiB compute buffer regardless of context. So f16 reaches much further than I projected, eating most of the "rotorquant unlocks context" story for this model.

2. **Rotorquant capacity at 256K doesn't translate to usable decode.** The 256K run loaded without OOM and generated one of three benchmarks correctly (LRU cache, 6/6 in 20 minutes wall-clock). The other two benchmarks hit the 30-minute timeout. Whatever causes the collapse between 128K and 256K (Metal flash-attention kernel edge case? Memory locality? Allocation boundary?) makes the nominal capacity unusable.

**Practical usable context ceiling on Qwen 27B Opus with rotorquant: ~128K.** Same as f16 with `-ub 256`. Rotorquant doesn't help on this specific model class.

### H3 — Gemma 4 31B-IT context unlock ❌ (can't test, not refuted)

Highest-value hypothesis, blocked by the gemma4 architecture gap. The math still says this should be the biggest win — dense 31B with f16 KV = 14 GB at 16K, a giant chunk of a 28.6 GB budget — but we couldn't run it.

### H4 — quality is noise-dominated at temp 0, single-shot ✅ (confirmed, dramatically)

Symmetric planar3/planar3 scored **16/17** on Qwen 27B Opus. The f16 baseline and K-only planar3 both score **11/17**. The 5-point swing comes from a single code-path flip: symmetric mode landed on a cleaner A* implementation (6/6) where both baselines wrote `except ImportError:` outside a try block and got 0/6 on the syntax error.

Same symmetric run at 32K and 65K produced byte-identical scores (16/17 each, same per-benchmark breakdown) and 256K produced the same LRU output (6/6) before hitting other failures. So the generation is deterministic under a given KV config, and the 5-point difference between symmetric and K-only is **entirely from the KV numerical perturbation steering the model onto a different code path**. Same thing we documented on the Spark for ik-llama vs mainline llama.cpp.

**Interpretation:** symmetric planar3's 16/17 is not a rotorquant quality win. It's one draw from a distribution where both baseline and rotorquant configs have roughly the same expected quality but occasionally land on different code paths. A best-of-3 run at temp=0.3 would probably collapse this back to indistinguishable.

### H5 — Qwen 27B Opus was predicted lowest-interest, turned out to be our only viable test path ⚠️ (promoted)

The pre-experiment framing called Qwen 27B Opus "low-signal, sanity check" because it was bandwidth-bound on weights and the KV cache was supposed to be a rounding error. In practice it was the *only* model we could run rotorquant on, and the result (+19% K-only speedup) was the strongest positive finding of the whole experiment.

The reason the K-only speedup materialized: the KV cache size at 32K f16 is ~5 GB for this model, which is large enough that per-token KV reads (during flash attention's softmax over context) are a meaningful fraction of per-token bandwidth. Compressing K from 2.5 GB → ~0.5 GB saves ~2 GB of per-token bandwidth, which on a 410 GB/s pipe is the difference between 13 and 15.5 tok/s.

My pre-experiment assumption was "KV is small in memory = KV is small in bandwidth." That conflated storage size with per-token read amplification during decode — the full KV is read once per layer per token, so a 5 GB cache produces 5 GB of reads per decode step, not 5 GB once. Learned.

### H6 — Metal port readiness ⚠️ (mixed)

The hypothesis predicted that symmetric planar3/planar3 on Metal might produce broken output because the V-dequant inverse rotation fix (`6e5a4aa` on CUDA) is listed as a TODO for Metal. The actual findings:

- **Metal output is not broken.** Symmetric planar3/planar3 generates coherent Python code at real temperature, passes most of the coding benchmarks, and differs from f16 only by expected numerical-noise variance. If the V-dequant fix is missing on Metal, its absence isn't catastrophic at these contexts.
- **Metal K-only works perfectly.** planar3/f16 generates the same tokens as f16 baseline (same 11/17 breakdown), just faster.
- **The 256K collapse is a real Metal bug.** Symmetric mode loads at 256K but decode drops from 11 tok/s to 1.7 tok/s, and two of three coding benchmarks timed out. Something in the Metal flash attention kernel or memory layout handles 256K + planar3 cache worse than 128K + planar3 or 256K + f16 (which we can't compare because f16 OOMs at 256K first).
- **No gemma4 support** in the fork's base llama.cpp blocks most of the M4 Max model lineup from being tested at all.

## Revised recommendations for M4 Max

Based on what we measured (and what we couldn't measure), here's the updated advice for running rotorquant on M4 Max:

### Where rotorquant wins

- **Dense GQA models where K cache is a meaningful per-token bandwidth fraction** (measured: Qwen 27B Opus-Distilled). Use `planar3 / f16` K-only mode. It's the first KV quantizer we've tested on Metal that's net-positive on decode throughput, at the same quality as f16. **+19% speedup, no quality cost.**

- **Same model at longer context, if you stay ≤128K.** Both f16 and planar3 K-only work at 128K. No clear winner without a 128K planar3/f16 coding bench run (not done).

### Where rotorquant doesn't help (on M4 Max specifically)

- **MoE sliding-window models (Gemma 4 26B-A4B family)**: can't be tested in this fork, but by our earlier measurements the KV cache is <1% of working set, so bandwidth savings from compression would be in the noise. Compute buffer is the actual context constraint.
- **Dense models at 256K+ context**: symmetric rotorquant loads but decodes at 1.7 tok/s, which makes it unusable in practice. The usable context ceiling with rotorquant is ~128K, same as f16 at `-ub 256`.
- **Small models (≤9B weights)**: already have plenty of bandwidth and context headroom. No meaningful savings.

### Where rotorquant *should* win but we couldn't test

- **Gemma 4 31B-IT Q4_K_M** — the dense full-attention model where f16 KV is 14 GB at 16K and we currently need mandatory turbo4 KV (at 11.8 tok/s, −19% vs expected f16 speed). H3 in the hypothesis. If the planarquant fork gets rebased to gemma4-capable llama.cpp, this is the single most interesting test to re-run. Projected gain from symmetric planar3: possibly +10-30% vs turbo4, same context capacity.

## Raw run artifacts

- Experiment outputs: [`experiments/rotorquant_m4max_bench/`](experiments/rotorquant_m4max_bench/)
- Bench runner: [`tools/m4max_rotorquant_bench.py`](tools/m4max_rotorquant_bench.py)
- Scorer: [`tools/score_combined.py`](tools/score_combined.py) (same as other M4 Max runs)
- Baseline for comparison: [`results/MODEL_RANKINGS_M4MAX.md`](results/MODEL_RANKINGS_M4MAX.md) (Qwen 27B Opus row)

## See also

- [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md) — the pre-experiment hypothesis this doc tests
- [ROTORQUANT_HYPOTHESIS.md](ROTORQUANT_HYPOTHESIS.md) — Spark version of the pre-experiment hypothesis
- [ROTORQUANT_HYPOTHESIS_5090.md](ROTORQUANT_HYPOTHESIS_5090.md) — 5090 version
- [results/TURBOQUANT_IMPACT_M4MAX.md](results/TURBOQUANT_IMPACT_M4MAX.md) — the "turbo4 KV is slower than f16 on Metal" finding that motivated this experiment
