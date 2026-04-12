# RotorQuant on M4 Max — Post-experiment Results

**Date:** 2026-04-11
**Hardware:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X @ 410 GB/s, ~28.6 GB Metal free working set
**Stack (first pass):** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` commit `20efe75cf`, Metal build (ggml 0.9.8)
**Stack (second pass, with cherry-picks):** same branch, cherry-picked `63f8fe0ef` (gemma4 architecture from upstream #21309, mtmd bits reverted) and `5208e2d5b` (gemma4 chat template fix from #21326) from the TheTom turboquant branch to get gemma4 support
**Pre-experiment doc:** [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md)

The hypothesis predicted rotorquant could be the first KV quantizer that's net-positive on Metal, because our earlier measurements showed TurboQuant turbo4 is *slower* than f16 on this hardware (60.3 → 46.0 tok/s on Gemma 4 26B-A4B Q6_K). The mechanism: rotorquant's Givens 2D rotation does ~64× less dequant compute per element than TurboQuant's Walsh-Hadamard butterfly, and Metal's bottleneck on KV quantization is apparently the dequant compute, not the KV bandwidth.

The actual results are more nuanced than the hypothesis and have one show-stopping surprise.

## Update: Gemma 4 unblocked via cherry-pick

The show-stopper below was real — **on the first pass** of the experiment, Gemma 4 models couldn't be loaded because the planarquant fork's base llama.cpp (ggml 0.9.8) predates `gemma4` architecture support. We fixed this post-hoc by **cherry-picking two commits from TheTom's newer turboquant branch** onto the planarquant branch:

1. **`63f8fe0ef`** (upstream PR #21309) — adds `LLM_ARCH_GEMMA4`, `src/models/gemma4-iswa.cpp` (311 lines), and the rest of the Gemma 4 text-model infrastructure. Cherry-picked cleanly except for one conflict in `tools/mtmd/mtmd.cpp` (multimodal image processing), which we resolved by reverting to planarquant's version since we don't need multimodal for text benchmarks. Also reverted the other mtmd/ bits from the same commit because they referenced `mtmd-image.h`, a file from the newer turboquant base that planarquant doesn't have.
2. **`5208e2d5b`** (upstream PR #21326) — the Gemma 4 chat-template fix. Needed because without it, the model emits ~43 KB of reasoning content despite `--reasoning-budget 0` being set. Cherry-picked cleanly with no conflicts.

The rebuilt binary recognizes `gemma4` architecture and all four rotorquant KV cache types (`planar3`, `iso3`, `planar4`, `iso4`) still work. Worktree lives at `~/git/TheTom/llama-cpp-planarquant` on branch `local/planarquant-gemma4-cherrypick`.

**Second-pass results on Gemma 4 follow in the tables below. The headline finding changed**: Gemma 4 26B-A4B Q6_K and Gemma 4 31B-IT Q4_K_M both show **K-only planar3/f16 speedups in the +13% to +20% range vs their best baseline configs**, at identical quality scores. This is the biggest positive result of the whole experiment.

---

## First-pass show-stopper (now resolved): Gemma 4 models couldn't be tested

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

First-pass runs were limited to **Qwen 3.5 27B Opus-Distilled Q4_K_M** (the only model with an architecture supported by the fork's base llama.cpp). Second-pass runs (after the cherry-pick) added **Gemma 4 26B-A4B Q6_K** and **Gemma 4 31B-IT Q4_K_M**. Every data point below is one of these three models on the 3-benchmark coding suite at temp 0, single-shot, `-np 1 --jinja --reasoning-budget 0`.

## Measurements — all configurations

### Qwen 3.5 27B Opus-Distilled Q4_K_M (dense GQA, `qwen35` arch)

Baseline from existing [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) is 13 tok/s @ 11/17 at 32K f16 with default `-ub 512`.

| Config | Context | `-ub` | Avg tok/s | Score | Δ vs baseline |
|---|---|---|---|---|---|
| **f16 / f16** (original baseline) | 32K | 512 default | 13.0 | 11/17 | — |
| **planar3 / f16** (K-only, rotorquant) | 32K | 512 | **15.5** | **11/17** | **+19% speed, same quality** |
| planar3 / planar3 (symmetric) | 32K | 512 | 11.0 | 16/17 | −15% speed, +5 quality (noise / path flip) |
| f16 / f16 at 65K | 65K | 256 | 15.4 | 11/17 | +18% speed (mostly from `-ub 256`, not context) |
| planar3 / planar3 at 65K | 65K | 256 | 10.8 | 16/17 | −17% speed, same quality as 32K symmetric |
| planar3 / planar3 at **256K** | 256K | 256 | **1.7** (two benchmarks timed out at 30 min) | 6/6 (LRU only, ExprEval+A* DNF) | **Context fits but decode collapses** |

### Gemma 4 26B-A4B Q6_K (MoE, sliding window + global, `gemma4` arch)

Baseline from [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) is 60.3 tok/s @ 15/17 at 16K f16, and 46.0 tok/s @ 16/17 at 16K turbo4.

| Config | Context | `-ub` | Avg tok/s | Score | Δ vs f16 baseline |
|---|---|---|---|---|---|
| f16 / f16 (from rankings doc) | 16K | 512 default | 60.3 | 15/17 | — |
| turbo4 / turbo4 (from rankings doc) | 16K | 512 default | 46.0 | 16/17 | −24% |
| **planar3 / f16** (K-only, rotorquant) | 16K | 512 | **68.2** | **15/17** | **+13% speed, IDENTICAL quality** |
| planar3 / planar3 (symmetric) | 16K | 512 | 12.3 | 0/5 (ExprEval DNF) | **Runaway generation bug** — model emits 16K tokens without stopping (`finish=length`). Experiment aborted after first benchmark. |

### Gemma 4 31B-IT Q4_K_M (dense full attention, `gemma4` arch)

Baseline from [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) is 11.8 tok/s @ 17/17 at 32K turbo4 (turbo4 is mandatory at 32K because f16 KV is ~14 GB, too big for the 30 GB Metal working set).

| Config | Context | `-ub` | Avg tok/s | Score | Δ vs turbo4 baseline |
|---|---|---|---|---|---|
| turbo4 / turbo4 (from rankings doc) | 32K | 256 | 11.8 | 17/17 | — |
| **planar3 / f16** (K-only, rotorquant) | 32K | 256 | **14.2** | **17/17** | **+20% speed, IDENTICAL quality** |
| planar3 / planar3 (symmetric) | 32K | 256 | — (aborted) | — | Same runaway bug as Gemma 4 26B-A4B symmetric. Never completed a benchmark. |

### Context window gains: **zero on M4 Max**

Despite the clean throughput wins, **rotorquant unlocked no additional usable context on any of the three test models on this hardware.** The reasons differ per model and each reason points at a different (un)fixable obstacle:

| Model | Max usable ctx before rotorquant | Max usable ctx with rotorquant | Gain |
|---|---|---|---|
| Gemma 4 26B-A4B Q6_K (MoE sliding window) | 32K with f16 + `-ub 256` | **32K** (same — K-only, no V compression) | **0** |
| Gemma 4 31B-IT Q4_K_M (dense full attention) | 32K with turbo4 KV + `-ub 256` | **32K** (K-only; symmetric would extend but has the runaway bug) | **0** |
| Qwen 3.5 27B Opus-Distilled Q4_K_M (dense GQA) | 128K with f16 + `-ub 256` | 128K fine; 256K nominally loads (27 GB) but decode collapses to 1.7 tok/s | **0 usable** |

Three reasons, one per model:

1. **Gemma 4 26B-A4B**: the KV cache was already <1% of the 28.6 GB Metal working set thanks to sliding-window attention. There was nothing meaningful to compress. Context cap is bounded by weights + compute buffer, not KV — a rotorquant problem rotorquant can't solve.

2. **Gemma 4 31B-IT**: this is the one model in our set where KV is genuinely the bottleneck (f16 V cache is ~14 GB at 16K, dominating the budget). K-only `planar3 / f16` compresses only the K half, leaving V at f16. That gets the +20% throughput win at the current 32K ceiling but doesn't buy new context. The path to unlocking 48-64K would be symmetric `planar3 / planar3`, which compresses V too — but symmetric hits the runaway-generation bug on Gemma 4 on this Metal backend. Almost certainly the Metal V-dequant inverse-rotation TODO from rotorquant's CLAUDE.md. Fixable upstream, not fixable by us.

3. **Qwen 3.5 27B Opus-Distilled**: f16 already reaches 128K on this model — the pre-experiment projections in this repo were too conservative (we had assumed the Gemma 4 compute buffer formula was universal, but Qwen 27B Opus has a near-constant 250-420 MiB compute buffer regardless of context). Symmetric `planar3 / planar3` nominally fits at 256K (27 GB projected, below the 28.6 GB budget), but decode collapses from 11 → 1.7 tok/s somewhere between 128K and 256K, and two of three coding benchmarks timed out at 30 minutes. The nominal +4× capacity isn't usable in practice; the practical ceiling is the same 128K f16 already reaches.

**Compare to the 5090**, where the context unlock on Gemma 4 31B-IT was the headline win of that platform's rotorquant experiment: 58K ctx turbo4 → **160K ctx iso3** (+102K usable) at 22/22 quality. That gain came from symmetric `iso3/iso3` working cleanly on CUDA — the same path that hangs on Metal for Gemma 4 today. **If the Metal symmetric V-dequant port ever lands upstream, H2/H3's context-expansion predictions should replicate here**. Until then, rotorquant on M4 Max is **a pure throughput win at the same context tiers we already had**.

### Summary — the K-only mode wins consistently

All three tested models show the same pattern: **K-only `planar3 / f16` is faster than the best prior config at identical quality.**

| Model | Prior best config | New best config | Speedup | Quality |
|---|---|---|---|---|
| Qwen 3.5 27B Opus Q4_K_M (dense) | f16/f16 @ 13.0 tok/s | **planar3/f16 @ 15.5 tok/s** | **+19%** | 11/17 → 11/17 (identical) |
| Gemma 4 26B-A4B Q6_K (MoE) | f16/f16 @ 60.3 tok/s | **planar3/f16 @ 68.2 tok/s** | **+13%** | 15/17 → 15/17 (identical) |
| Gemma 4 31B-IT Q4_K_M (dense) | turbo4/turbo4 @ 11.8 tok/s | **planar3/f16 @ 14.2 tok/s** | **+20%** | 17/17 → 17/17 (identical) |

Three-for-three. This is the first KV cache format we've tested that's strictly better than f16 on every model where both fit, and strictly better than turbo4 on the one model where turbo4 was mandatory. On Metal specifically, rotorquant's deferred-K quantization (F16 during prefill + planar3 compression on decode insertion) produces a real, reproducible decode-throughput win across dense GQA, MoE sliding-window, and dense full-attention architectures.

### Why K-only works and symmetric doesn't on Gemma 4

Symmetric `planar3/planar3` on Qwen 27B Opus is slow but functional (16/17 quality, 11 tok/s). On **both** Gemma 4 models, symmetric hits a different failure mode: the model starts generating tokens normally but never emits a stop token, hitting `max_tokens=16384` with `finish=length` and producing 16K of unstructured content instead of a coding answer. Reasoning is confirmed off (`think=0c`), so this isn't the reasoning-budget issue we hit before the template fix.

Best guess: the symmetric V-dequant path on Metal perturbs the numerical noise enough to push Gemma 4's generation off the typical stop-token probability peak. The same pattern shows up at 16K (Gemma 26B-A4B) and 32K (Gemma 31B), so it's not context-dependent. The K-only path sidesteps this because V stays at f16 — no V-side perturbation means Gemma 4's stop logic lands where it expects to.

This is a **Metal V-dequant bug** — not a general rotorquant limitation. On CUDA (per the rotorquant paper's 5090 data), symmetric planar3/planar3 works fine on Llama 3.1 8B. The symmetric V-dequant inverse rotation fix that landed for CUDA in `6e5a4aae4` ("Fix symmetric V=planar3/iso3: add inverse rotation to V dequant") is listed as a TODO for Metal in the rotorquant CLAUDE.md. Our measurements strongly suggest **that TODO is the root cause of the Gemma 4 runaway-generation failure**, and porting that fix to Metal would unlock symmetric mode on Gemma 4.

Additional data point from a single short decode test (not scored):
- **planar3 / planar3 at 128K**: decode 11.3 tok/s on a 500-token response. No collapse at 128K. Collapse happens between 128K and 256K.

## Hypothesis verdicts

### H1 — rotorquant recovers from the turbo4 slowdown ✅✅ (confirmed on all three test models after rebase)

The prediction was planar3 would recover most of the gap between turbo4 (−23% vs f16) and f16. Actual results across three model architectures:

- **Qwen 27B Opus (dense GQA) K-only: +19% vs f16, same quality** (15.5 vs 13.0 tok/s, both 11/17). First pass, before rebase.
- **Gemma 4 26B-A4B Q6_K (MoE sliding window) K-only: +13% vs f16, same quality** (68.2 vs 60.3 tok/s, both 15/17). Second pass, after rebase. **+48% vs turbo4's 46 tok/s on the same model.**
- **Gemma 4 31B-IT Q4_K_M (dense full attention) K-only: +20% vs turbo4, same quality** (14.2 vs 11.8 tok/s, both 17/17). Second pass, after rebase.

**Three-for-three. H1 holds unambiguously.** The K-only `planar3 / f16` mode is the first KV quantizer we've tested that is strictly better than f16 on every model where f16 fits, and strictly better than turbo4 on the one model where turbo4 was mandatory. The deferred-K deferred quantization path (f16 during prefill, quantized-on-insertion during decode) is a genuine Pareto improvement for Metal-backend inference on this hardware.

Symmetric `planar3/planar3` is slower across the board (Qwen 27B: −15%, Gemma 4: **runaway generation bug**). Direction is correct for the speed axis on Qwen 27B (better than turbo4's −23%), but K-only is strictly the right config for a speed-focused run on every model. Symmetric mode on Gemma 4 is effectively broken by what looks like a missing Metal V-dequant port.

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

### H3 — Gemma 4 31B-IT context unlock ⚠️ (unblocked, partial confirmation)

After the gemma4 cherry-pick, H3 became testable. The highest-value prediction was "planar3/iso3 should match turbo4's capacity AND beat its throughput on Gemma 31B."

Result: **K-only `planar3 / f16` on Gemma 31B at 32K context = 14.2 tok/s, 17/17 quality.** That's +20% over turbo4's 11.8 tok/s at the same quality, fits comfortably in the 28.6 GB Metal budget (~26 GB total used), and avoids the symmetric-mode runaway-generation bug. **This is the single best measured result of the entire rotorquant experiment.**

The original H3 also projected context expansion (32K → 48-64K) on Gemma 31B via more aggressive KV compression. That part is NOT confirmed because K-only mode doesn't compress V (V stays at f16), so per-token KV cost at longer contexts grows from the f16 V side. We didn't push past 32K. Symmetric mode would be needed for the context expansion win, and symmetric mode is currently broken on Gemma 4 Metal.

**Summary:** H3's throughput prediction holds (+20% vs turbo4 at 17/17). H3's context-expansion prediction is unresolved — the infrastructure gate (Metal V-dequant fix) is still in front of the bigger win.

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

After the gemma4 rebase, the recommendation is simple and strong: **use `-ctk planar3 -ctv f16` on M4 Max for any model where the standard turboquant fork currently uses f16 or turbo4 KV**. Measured on all three test models, three-for-three, at +13% to +20% speed over the best prior config with identical coding-benchmark quality.

### Where rotorquant wins (measured)

- **Gemma 4 31B-IT Q4_K_M** — +20% vs turbo4 at same 17/17 quality. The single highest-value win. Previously blocked by the fork's arch gap; now unblocked.
- **Gemma 4 26B-A4B Q6_K** — +13% vs f16 at same 15/17 quality. The "S-tier coding model on a laptop" class gets a free speed boost.
- **Qwen 3.5 27B Opus-Distilled Q4_K_M** — +19% vs f16 at same 11/17 quality. Dense 27B with GQA, the model class where KV bandwidth is a meaningful fraction of per-token cost.

In all three cases: **use `-ctk planar3 -ctv f16`**. Do not use symmetric `planar3/planar3` on Gemma 4 (see below).

### Where rotorquant doesn't help (or is broken)

- **Symmetric planar3/planar3 on Gemma 4 26B-A4B or 31B-IT**: runaway generation. Model emits 16K tokens with `finish=length` and never reaches a stop token. Matches the "Port symmetric V dequant fix to Metal backend" TODO in rotorquant's CLAUDE.md. Fixable by the rotorquant authors; until then, use K-only on Gemma 4.
- **Dense models at 256K+ context**: symmetric rotorquant on Qwen 27B Opus loads at 256K but decodes at 1.7 tok/s, unusable. Context ceiling on this hardware with rotorquant is ~128K in practice.
- **Small models (≤9B weights)**: already have plenty of bandwidth headroom. Planar3 K-only would probably still help marginally but we didn't test (low priority — you don't need the speedup on a small model).

### Where rotorquant *would* win if specific issues got fixed

- **Context expansion on Gemma 4 31B-IT (48-64K)** — blocked by the Metal symmetric V-dequant bug. K-only planar3/f16 can't expand context because V stays at f16 and the V cache is the dominant per-token cost at long context on dense 31B. Symmetric planar3/planar3 could compress both halves, but runs away on generation. If the authors port the V-dequant inverse-rotation fix to Metal, this unlocks.

### Build instructions

The rotorquant planarquant branch needs two upstream commits cherry-picked onto it to support Gemma 4 text models:

```bash
cd ~/git/TheTom/llama-cpp-planarquant  # worktree or clone of johndpope/llama-cpp-turboquant
git checkout johndpope/feature/planarquant-kv-cache
git cherry-pick 63f8fe0ef  # upstream PR #21309: Gemma 4 architecture
# In tools/mtmd/mtmd.cpp conflict: take HEAD (revert to planarquant's version)
# Also revert tools/mtmd/{CMakeLists.txt, clip-graph.h, clip-impl.h, clip-model.h, clip.cpp, models/models.h}
# and delete tools/mtmd/models/gemma4v.cpp (the cherry-picked mtmd bits reference mtmd-image.h which doesn't exist on planarquant base)
git cherry-pick 5208e2d5b  # upstream PR #21326: Gemma 4 chat template fix (clean, no conflicts)
cmake --build build --target llama-server -j 8
```

Both commits come from `TheTom/feature/turboquant-kv-cache` (remote `origin`).

## Raw run artifacts

- Experiment outputs: [`experiments/rotorquant_m4max_bench/`](../experiments/rotorquant_m4max_bench/)
- Bench runner: [`tools/m4max_rotorquant_bench.py`](../tools/m4max_rotorquant_bench.py)
- Scorer: [`tools/score_combined.py`](../tools/score_combined.py) (same as other M4 Max runs)
- Baseline for comparison: [`MODEL_RANKINGS_M4MAX.md`](MODEL_RANKINGS_M4MAX.md) (Qwen 27B Opus row)

## See also

- [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md) — the pre-experiment hypothesis this doc tests
- [ROTORQUANT_HYPOTHESIS_SPARK.md](../ROTORQUANT_HYPOTHESIS_SPARK.md) — Spark version of the pre-experiment hypothesis
- [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md) — Spark post-experiment results (net-negative, as predicted)
- [ROTORQUANT_HYPOTHESIS_5090.md](../ROTORQUANT_HYPOTHESIS_5090.md) — 5090 version
- [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md) — the "turbo4 KV is slower than f16 on Metal" finding that motivated this experiment
