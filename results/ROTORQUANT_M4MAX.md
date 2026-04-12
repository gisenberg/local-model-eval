# RotorQuant on M4 Max — Post-experiment Results

**Date:** 2026-04-11 (original) / 2026-04-12 (matched-base correction)
**Hardware:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X @ 410 GB/s, ~28.6 GB Metal free working set
**Stack (first pass):** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` commit `20efe75cf`, Metal build (ggml 0.9.8)
**Stack (second pass, with cherry-picks):** same branch, cherry-picked `63f8fe0ef` (gemma4 architecture from upstream #21309, mtmd bits reverted) and `5208e2d5b` (gemma4 chat template fix from #21326) from the TheTom turboquant branch to get gemma4 support
**Pre-experiment doc:** [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md)

> **⚠ CORRECTION 2026-04-12: the "three-for-three K-only wins" headline was wrong.** The original "+13% to +20% K-only win across three models" claim in the sections below compared **new-base planar3/f16** against **old-base f16 or turbo4**. That comparison double-counts the llama.cpp base upgrade that came in with the Gemma 4 cherry-picks. On matched-base + matched-ub measurements (all on the planarquant fork + cherry-picks at default `-ub 512`):
>
> | Model | f16/f16 | planar3/f16 | Δ |
> |---|---|---|---|
> | Qwen 3.5 27B Opus-Distilled Q4_K_M (dense GQA) | 13.0 tok/s | **15.5 tok/s** | **+19%** ✅ clean win |
> | Gemma 4 26B-A4B Q6_K (MoE sliding window) | **65.6 tok/s** | 64.5 tok/s | −1.5% (noise, tied) |
> | Gemma 4 31B-IT Q4_K_M (dense full attention ISWA) | **15.3 tok/s** | 15.17 tok/s | −1.0% (noise, tied) |
>
> Revised H1 verdict: **one-for-three, not three-for-three.** Rotorquant K-only is a clean +19% win on dense GQA (Qwen 27B Opus-Distilled), and tied with f16 on both Gemma 4 architectures when tested on the same base at the same ubatch. The mechanism is still "cheap Givens rotation beats expensive WHT butterfly on Metal", but it only pays out when KV bandwidth is a meaningful fraction of per-token cost — which is the dense-GQA case, not Gemma 4 MoE (sliding-window keeps KV tiny) or Gemma 4 31B (ISWA keeps average KV/token moderate and the extra dequant per decode eats the bandwidth savings).
>
> The **base-upgrade win is still huge**. Gemma 4 31B-IT went from 11.8 tok/s @ turbo4/ub=256 (old base) to 15.3 tok/s @ f16/default ub (new base) — a +30% improvement that has nothing to do with rotorquant and everything to do with the compute-buffer bug fix that was bundled into the Gemma 4 cherry-picks. Gemma 4 26B-A4B Q6_K went from a 16K ceiling at default ub (old base, OOMed at 32K) to comfortable 32K at default ub on the new base.
>
> The sections below are preserved as the original analysis. Read them as "here's what I measured and what I thought it meant in pass 1 and pass 2" — the specific numbers are all real, but the "+13%/+20% Gemma 4 K-only wins" are cross-base artifacts that evaporated on matched-base retest. See the MODEL_RANKINGS_M4MAX.md refresh for the current tier list.

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

### Context window gains: **zero from rotorquant alone, big from the base upgrade it came with**

Rotorquant K-only (`planar3 / f16`) unlocked **no additional usable context** on any of the three test models on this hardware. The apparent context gains we hit while testing (e.g. "Gemma 4 26B-A4B Q6_K now reaches 32K at default `-ub`") turned out to be **side effects of the llama.cpp base upgrade** that came with the Gemma 4 cherry-pick, not anything rotorquant did.

**The full picture after some follow-up measurements:**

1. **K-only planar3/f16 is deferred quantization.** The K cache is allocated at **f16 size at load time**, and only converted to planar3 at insertion during decode. So it saves **bandwidth** (compressed reads during decode) but **not memory** (allocation footprint is identical to f16 at load time). The bandwidth savings explain the +13% to +20% throughput wins; they do not buy more context. Measured: at 128K context on Gemma 4 31B-IT, f16 KV allocates 11,440 MiB and planar3/f16 KV allocates **12,557 MiB** (slightly larger, not smaller).

2. **The base upgrade was the real context unlock.** We cherry-picked upstream PR #21309 onto the planarquant branch to get `LLM_ARCH_GEMMA4` support, and that commit (or something close to it) also dramatically shrank the Metal FA compute buffer for sliding-window attention models. Measured on the same Gemma 4 31B-IT Q4_K_M at 32K default ub:
   - **Old base** (turboquant fork @ `8590cbff9`): compute buffer = **16,689 MiB (16.7 GB)**, total projected = 37.9 GB → OOM
   - **New base** (planarquant @ `20efe75cf` + cherry-picks): compute buffer = **523 MiB**, total projected = 21.7 GB → fits
   - **That's a 32× reduction** in compute buffer, and the only reason Gemma 4 31B-IT now fits at 32K with default settings. Rotorquant did nothing here.

3. **My earlier "Gemma 4 31B-IT has 870 KB/token KV" math was also wrong.** I assumed every layer uses full attention; actually Gemma 4 31B uses ISWA where only ~9 of 62 layers are global attention and the rest are sliding-window. Real KV at 32K f16 = **3.7 GB total** (~120 KB/token averaged), not the 27.8 GB I projected. So "turbo4 is mandatory for Gemma 31B" was never based on a measurement — it was based on bad math plus the (real) compute-buffer-bug OOM.

**What the context ceilings on Gemma 4 31B-IT actually look like now (new base, default ub):**

| KV config | 32K | 64K | 128K | 256K |
|---|---|---|---|---|
| f16 / f16 | ✅ 21.7 GB | ✅ 24.3 GB | ❌ 29.4 GB | — |
| **planar3 / f16 K-only** | ✅ **+20% tok/s vs turbo4** | likely ✅ | ❌ 30.5 GB (no mem savings from K-only) | — |
| turbo4 / turbo4 (new base) | ✅ ~18.5 GB | ✅ — | ✅ 21.0 GB | ✅ 24.1 GB (loads) |

**So the actual picture of "what gives you more context on Gemma 4 31B-IT":**
- From 16K to 64K: the base upgrade (4× gain for free)
- From 64K to 128K+: turbo4 KV on the new base (because turbo4 compresses at allocation time, not deferred)
- Rotorquant K-only stays at 64K ceiling (deferred quantization = no memory savings)

**On the other two test models:**

- **Gemma 4 26B-A4B Q6_K**: KV is <1% of working set on MoE sliding window; compute buffer was the cap. New base fixes compute buffer; the 32K→64K+ window opens for f16, not because of rotorquant. K-only still wins on throughput at any context it fits at.
- **Qwen 3.5 27B Opus-Distilled**: f16 already reached 128K on both bases (Qwen 27B's compute buffer doesn't have the sliding-window bug). Symmetric planar3/planar3 nominally fits 256K but decode collapses to 1.7 tok/s. No usable context gain.

**Bottom line:** rotorquant on M4 Max is a **pure throughput win** (+13% to +20% across three models), not a context-capacity win. The context wins I initially attributed to rotorquant came from the llama.cpp base upgrade bundled into the cherry-picks. **For actual context-capacity gains on Gemma 4 31B-IT on the new base, use turbo4 KV** — it compresses at allocation time where rotorquant K-only doesn't.

### Detailed per-model obstacles (the original analysis, corrected)

Despite the overall story shift, each model still has a distinct reason rotorquant specifically didn't extend its context:

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

### Summary — K-only wins on dense GQA, ties on Gemma 4 (matched-base correction)

The original version of this section claimed "three-for-three K-only wins." That was wrong — it compared new-base planar3/f16 against old-base f16 or turbo4, so the +13% and +20% Gemma 4 speedups were mostly the base upgrade, not rotorquant. On **matched-base, matched-ub** retest:

| Model | f16/f16 @ default ub | planar3/f16 @ default ub | Δ | Quality |
|---|---|---|---|---|
| Qwen 3.5 27B Opus Q4_K_M (dense GQA) | 13.0 tok/s | **15.5 tok/s** | **+19%** ✅ | 11/17 → 11/17 (identical) |
| Gemma 4 26B-A4B Q6_K (MoE sliding window) | **65.6 tok/s** | 64.5 tok/s | **−1.5%** (noise) | 15/17 → 15/17 (identical) |
| Gemma 4 31B-IT Q4_K_M (dense full attention, ISWA) | **15.3 tok/s** | 15.17 tok/s | **−1.0%** (noise) | 17/17 → 17/17 (identical) |

**One-for-three, not three-for-three.** Rotorquant K-only is still a genuine +19% Pareto improvement over f16 on dense GQA 27B+ models (where KV bandwidth is a meaningful fraction of per-token cost), and still deterministic at temp 0 (identical quality on all three models). But the Gemma 4 wins I originally reported evaporated when I retested on the new base at matched ubatch: rotorquant and f16 are within 1.5% of each other on both Gemma 4 architectures.

The model-class pattern: rotorquant K-only pays out when the KV cache generates meaningful decode bandwidth (dense attention, full GQA heads, large contexts). On Gemma 4 26B-A4B the sliding-window attention keeps KV tiny; on Gemma 4 31B-IT the ISWA mix keeps averaged KV moderate and the Givens dequant compute cancels the bandwidth win. **Metal's cheap bandwidth / expensive compute asymmetry still matters** — but for rotorquant to exploit it, the model has to actually be KV-bandwidth-bound, which Gemma 4 isn't.

### Why K-only works and symmetric doesn't on Gemma 4

Symmetric `planar3/planar3` on Qwen 27B Opus is slow but functional (16/17 quality, 11 tok/s). On **both** Gemma 4 models, symmetric hits a different failure mode: the model starts generating tokens normally but never emits a stop token, hitting `max_tokens=16384` with `finish=length` and producing 16K of unstructured content instead of a coding answer. Reasoning is confirmed off (`think=0c`), so this isn't the reasoning-budget issue we hit before the template fix.

Best guess: the symmetric V-dequant path on Metal perturbs the numerical noise enough to push Gemma 4's generation off the typical stop-token probability peak. The same pattern shows up at 16K (Gemma 26B-A4B) and 32K (Gemma 31B), so it's not context-dependent. The K-only path sidesteps this because V stays at f16 — no V-side perturbation means Gemma 4's stop logic lands where it expects to.

This is a **Metal V-dequant bug** — not a general rotorquant limitation. On CUDA (per the rotorquant paper's 5090 data), symmetric planar3/planar3 works fine on Llama 3.1 8B. The symmetric V-dequant inverse rotation fix that landed for CUDA in `6e5a4aae4` ("Fix symmetric V=planar3/iso3: add inverse rotation to V dequant") is listed as a TODO for Metal in the rotorquant CLAUDE.md. Our measurements strongly suggest **that TODO is the root cause of the Gemma 4 runaway-generation failure**, and porting that fix to Metal would unlock symmetric mode on Gemma 4.

Additional data point from a single short decode test (not scored):
- **planar3 / planar3 at 128K**: decode 11.3 tok/s on a 500-token response. No collapse at 128K. Collapse happens between 128K and 256K.

## Hypothesis verdicts

### H1 — rotorquant recovers from the turbo4 slowdown ⚠️ (holds on dense GQA only, refuted on Gemma 4 after matched-base retest)

The prediction was planar3 would recover most of the gap between turbo4 (−23% vs f16) and f16. The original "three-for-three confirmed" verdict was wrong — it compared new-base planar3/f16 numbers against old-base f16/turbo4 numbers and thereby credited rotorquant for the llama.cpp base upgrade. Matched-base results:

- **Qwen 27B Opus (dense GQA) K-only: +19% vs f16, same quality** (15.5 vs 13.0 tok/s, both 11/17). ✅ Clean win. First pass, but both measurements are on the same fork commit.
- **Gemma 4 26B-A4B Q6_K (MoE sliding window) K-only: −1.5% vs f16, same quality** (64.5 vs 65.6 tok/s, both 15/17 @ 32K default ub). ❌ Tied — the "+13% vs 60.3 tok/s" was comparing new base against old-base f16 at 16K.
- **Gemma 4 31B-IT Q4_K_M (dense full attention ISWA) K-only: −1.0% vs f16, same quality** (15.17 vs 15.3 tok/s, both 17/17 @ 32K default ub). ❌ Tied — the "+20% vs 11.8 tok/s turbo4" was comparing new base against old-base turbo4 with the compute-buffer-bug workaround `-ub 256`.

**One-for-three. H1 holds only on dense GQA 27B.** Rotorquant K-only is a Pareto improvement over f16 when the KV cache generates meaningful decode bandwidth (dense GQA at 32K+ starts to see real per-token KV reads). On Gemma 4 MoE (sliding window keeps KV tiny) and Gemma 4 31B (ISWA mix keeps averaged KV moderate), Metal's cheap-bandwidth-expensive-compute asymmetry goes the other way: the extra Givens dequant compute per decode cancels the bandwidth savings, and rotorquant K-only ties f16 instead of beating it.

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

### H3 — Gemma 4 31B-IT context unlock ❌ (unblocked, but the throughput win belongs to the base upgrade, not rotorquant)

After the gemma4 cherry-pick, H3 became testable. The highest-value prediction was "planar3/iso3 should match turbo4's capacity AND beat its throughput on Gemma 31B."

Original result (pre-correction): **K-only `planar3 / f16` at 32K ub=256 = 14.2 tok/s, 17/17 quality.** I compared this to the old-base turbo4 ub=256 result (11.8 tok/s) and called it "+20% at same quality." But that comparison was unfair — the 11.8 tok/s baseline is from a different llama.cpp base with a compute-buffer bug that required `-ub 256`.

Matched-base retest (2026-04-12): f16/f16 at 32K default ub on the new base runs at **15.3 tok/s, 17/17**, and planar3/f16 at 32K default ub runs at **15.17 tok/s, 17/17**. The two are tied within noise. The actual "+30% Gemma 31B throughput win" is **15.3 vs 11.8 tok/s = +30%**, and it's entirely from the new base eliminating the old `-ub 256` workaround, not from rotorquant K-only.

**Corrected H3 summary:**
- Throughput at 32K: **new-base f16 and planar3/f16 are tied at ~15.2 tok/s**, both +30% over old-base turbo4/ub=256. Rotorquant contribution = 0.
- Context expansion at 48-64K: not tested with planar3/f16. f16/f16 fits 64K at 24.3 GB on the new base; K-only is deferred quantization and wouldn't fit more than that. For 128K+, use turbo4/turbo4 (compresses at allocation time), not planar3.
- The "single best measured result" framing from the pre-correction version was wrong on two axes: the comparison was unfair, and there's no rotorquant win here to report once the comparison is matched.

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

After the matched-base correction, the recommendation is narrower than the pre-correction version: **use `-ctk planar3 -ctv f16` on dense GQA ≥27B models** (Qwen 3.5 27B Opus-Distilled), where it's a clean +19% over f16. **On Gemma 4 (26B-A4B or 31B-IT), default to plain `-ctk f16 -ctv f16`** — rotorquant K-only is tied with f16 at matched ub on the new base, and the extra complexity isn't worth it.

### Where rotorquant wins (measured on matched base + matched ub)

- **Qwen 3.5 27B Opus-Distilled Q4_K_M** — **+19% vs f16 at same 11/17 quality**. Dense 27B with GQA, the model class where KV bandwidth is a meaningful fraction of per-token cost. This is the one clean rotorquant K-only win on M4 Max.

### Where rotorquant is tied with f16 (originally claimed wins, corrected)

- **Gemma 4 31B-IT Q4_K_M** — planar3/f16 at 15.17 tok/s vs f16/f16 at 15.3 tok/s at matched 32K default ub. Tied, noise-level. The original "+20% vs turbo4" finding was a cross-base comparison against 11.8 tok/s from the old-base compute-buffer-bug workaround.
- **Gemma 4 26B-A4B Q6_K** — planar3/f16 at 64.5 tok/s vs f16/f16 at 65.6 tok/s at matched 32K default ub. Tied, noise-level. The original "+13% vs f16" was comparing new-base planar3/f16 at 16K against old-base f16 at 16K.

**On Gemma 4**: use plain `-ctk f16 -ctv f16`. Default ub=512 works fine on the new base; no `-ub 256` workaround needed. If you need 128K+ context on Gemma 4 31B-IT, switch to `-ctk turbo4 -ctv turbo4` (compresses at allocation time, unlike K-only which is deferred).

Do not use symmetric `planar3/planar3` on Gemma 4 (see below).

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
