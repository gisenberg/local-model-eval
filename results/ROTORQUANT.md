# RotorQuant KV cache compression — cross-platform

RotorQuant ships four symmetric KV cache quantizers (`iso3`, `planar3`, `planar4`, `iso4`) and an asymmetric K-only `planar3 / f16` mode with deferred K quantization (K stays f16 during prefill and compresses at insertion). The technique is Givens 2D / quaternion 4D block-diagonal rotations with ~64× fewer FMAs per dequant element than TurboQuant's Walsh-Hadamard butterfly — [the scrya.com/rotorquant paper](https://www.scrya.com/rotorquant.pdf) claims "better PPL, 28% faster decode, 5× faster prefill" than TurboQuant at the same 10.3× compression, measured on Llama 3.1 8B Q4_K_M on an RTX 5090.

We tested it on three platforms (M4 Max, DGX Spark, RTX 5090) across twenty-one (model, config) pairs. The fork used was `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache @ 20efe75cf`, with local rebase + cherry-picks where needed for Gemma 4 support (see [common blocker](#common-blocker-gemma-4-support-in-the-johndpope-fork)).

For the TurboQuant story this is measured against, see [TURBOQUANT.md](TURBOQUANT.md).

## Cross-platform verdict

| Platform | Best rotorquant config | Δ vs platform default | Quality | Recommend switch? |
|---|---|---:|---|---|
| **M4 Max** (Metal) | `planar3 / f16` (K-only) **on dense GQA only** | **+19%** on Qwen 27B Opus-Distilled; **tied** on Gemma 4 (~−1% at matched ub) | Identical | **Yes on Qwen 27B Opus.** **No on Gemma 4** — plain f16 is equivalent and simpler. |
| **Spark** (aarch64 CUDA 13) | `planar3 K / f16 V` on Qwen3.5-122B | ~−1% vs f16, +5/17 quality | Better in one measurement | **Yes** for Qwen3.5-122B. **No** for GLM-4.5-Air (broken output). |
| **5090** (Blackwell CUDA 13.2) | `planar3/f16` (lowest risk); maybe iso3 for Harmonic | Qwen/Qwopus: planar3 −4 to −6%. Harmonic iso3: +0.3% (parity). Gemma 31B iso3: −21%, **no context gain** (turbo4 already fits 262K) | Unchanged on Qwen-family; 22/22 on Gemma 31B | **No** — turbo4 remains default for all 5090 models. Harmonic iso3 parity pending replication. |

> **Correction 2026-04-12**: The original "three-for-three K-only wins on M4 Max" framing was wrong. Matched-base retest shows rotorquant K-only is a clean +19% win on **dense GQA** (Qwen 27B Opus-Distilled) only. On Gemma 4 26B-A4B Q6_K and Gemma 4 31B-IT Q4_K_M, planar3/f16 is within 1.5% of plain f16 at matched `-ub` — tied, not +13-20% ahead. The original Gemma 4 "wins" were cross-base comparisons (new-base planar3/f16 vs old-base f16/turbo4) that double-counted the llama.cpp base upgrade. See the M4 Max section below for matched-base numbers.

## Key cross-platform findings

**M4 Max**. RotorQuant K-only is the first KV quantizer we have tested that is *faster than f16* on Metal — **on dense GQA models**. We previously measured TurboQuant turbo4 as 23% slower than f16 on this hardware; rotorquant's Givens rotation is ~64× less dequant compute per element than turbo4's WHT butterfly, and that gap pays out where KV bandwidth is a meaningful fraction of per-token cost. Matched-base, matched-ub (planarquant fork + Gemma 4 cherry-picks, default `-ub 512`):

- **Qwen 3.5 27B Opus-Distilled Q4_K_M** (dense GQA): 15.5 tok/s vs 13.0 f16 = **+19%**, 11/17 → 11/17 identical. ✅ Clean win.
- **Gemma 4 26B-A4B Q6_K** (MoE sliding window): 64.5 tok/s vs 65.6 f16 = **−1.5%**, 15/17 → 15/17 identical. Tied. Sliding window keeps KV tiny — no bandwidth to save.
- **Gemma 4 31B-IT Q4_K_M** (dense full attention ISWA): 15.17 tok/s vs 15.3 f16 = **−1.0%**, 17/17 → 17/17 identical. Tied. ISWA keeps averaged KV moderate; extra dequant compute cancels bandwidth win.

One-for-three, not three-for-three. Context-capacity gains on M4 Max came from the llama.cpp base upgrade bundled into the Gemma 4 cherry-picks, **not** rotorquant: Gemma 4 31B went from 11.8 tok/s @ turbo4/`-ub 256` (old base) to 15.3 tok/s @ f16/default-ub (new base), a +30% throughput improvement entirely from a compute-buffer bug fix. Symmetric `planar3/planar3` on Gemma 4 hits a runaway-generation bug (16K tokens without a stop) that matches the Metal V-dequant TODO in rotorquant's CLAUDE.md.

**Spark**. Rotorquant throughput cost on Spark is negligible (~1% for K-only, 3-8% for symmetric on MoE), but quality behavior is **bimodal**. On Qwen3.5-122B (DeltaNet hybrid: only 12 of 48 layers use full attention), `planar3K/f16V` produced **18/17 tests** — a +5 point jump over the f16 baseline's 13/17. On GLM-4.5-Air (standard attention in every one of 46 layers), all three rotorquant configs produced literal `Hello???????...` garbage. **Deep full-attention stacks break rotorquant**; hybrid-attention tolerates it fine. The Qwen quality win is almost certainly a numerical-noise path flip (the model steers onto a cleaner generation path), not a general quality claim — this experiment also revealed the single-shot temp-0 noise band on Spark is **±5 points, not ±1**, which retroactively weakens some earlier claims.

**5090**. Rotorquant is mostly a throughput tax where TurboQuant is already well-tuned. On Qwen 27B Opus-Distilled and Qwopus 27B (two Qwen 3.5 27B dense variants), iso3/iso3 ran ~10.5% slower than turbo4 — matching the pre-experiment prediction — and planar3/f16 ran 3.5-5.9% slower with better quality variance. The surprise was **Harmonic 27B** (Qwen 2.5 base with thinking on): iso3/iso3 ran at 61.5 tok/s vs the 61.3 tok/s turbo4 baseline — **+0.3%**, the first datapoint where rotorquant's 10.3× compression cost zero throughput. Needs a 5-run replication before acting on it.

The hypothesis that rotorquant would unlock Gemma 4 31B-IT long context was **invalidated**: turbo4 already fits the full 262K native window at 28 GB VRAM and 46.4 tok/s. The "58K turbo4 ceiling" projection was based on an ~10× overestimate of per-token KV cost (870 KB/tok f16 assumed all 60 layers are full-attention; in reality Gemma 4 31B has only 10 non-SWA layers with ~4 shared-KV heads each = ~22 KB/tok turbo4). iso3 on Gemma 31B is strictly slower than turbo4 with zero context advantage.

## Common blocker: Gemma 4 support in the johndpope fork

The planarquant fork was based on upstream llama.cpp from 2026-03-25 (ggml 0.9.8), which predates `LLM_ARCH_GEMMA4`. Every Gemma 4 test case initially crashed with `unknown model architecture: 'gemma4'`.

**Resolved on 5090 and M4 Max via local cherry-picks; still open on Spark.**

### M4 Max path: two upstream cherry-picks (simpler)

1. **`63f8fe0ef`** (upstream PR #21309) — adds `LLM_ARCH_GEMMA4`, `src/models/gemma4-iswa.cpp`, and Gemma 4 text-model infrastructure. One conflict in `tools/mtmd/mtmd.cpp` (multimodal image code, unrelated) — reverted to planarquant's version. Also reverted the other `tools/mtmd/*` bits from the same commit because they reference `mtmd-image.h` which doesn't exist on the planarquant base.
2. **`5208e2d5b`** (upstream PR #21326) — Gemma 4 chat-template fix. Needed because without it `--reasoning-budget 0` leaks ~43 KB of reasoning content into output. Cherry-picks cleanly.

No kernel patch needed on M4 Max — the Metal FA kernels already handle the head dimensions Gemma 4 uses. Worktree: `~/git/TheTom/llama-cpp-planarquant`, branch `local/planarquant-gemma4-cherrypick`.

### 5090 path: full rebase + D=512 FA vec kernel patch (more work, pays out bigger)

1. **3-way merge** of upstream master (241 commits ahead) into the feature branch. Produced 5 conflicted files / 8 hunks, all resolvable in one pass:

   | File | Conflict | Resolution |
   |---|---|---|
   | `ggml/include/ggml.h` | Upstream added `GGML_TYPE_Q1_0 = 41`; rotorquant claimed 41-47 | Renumbered rotorquant types to 42-48 (safe, KV type IDs are runtime-only) |
   | `ggml/src/ggml-cuda/fattn.cu` (×2) | Different head-dim exclusion lists | Union: exclude 40, 72, 256, 512, 576, 640 |
   | `ggml/src/ggml-cuda/vendors/hip.h` | Upstream added `NCCL_CHECK`; rotorquant had variadic shuffle macros | Kept both |
   | `src/llama-graph.cpp` (×2) | TurboQuant V-padding reshape vs upstream's `v_rot` mul_mat | Kept both — orthogonal operations |
   | `src/llama-kv-cache.cpp` (×2) | MSVC InnerQ stubs + `convert_deferred_keys()` vs upstream's `ggml_gen_hadamard` / `set_input_{k,v}_rot` delegators | Kept all of the above |

2. **D=512 FA vec kernel instantiations.** Gemma 4 uses head_dim 512 for its 10 full-attention (non-SWA) layers; rotorquant only had vec kernel instances for D ∈ {64, 128, 256}. Upstream routes f16 D=512 through the MMA F16 kernel; rotorquant types only have vec-kernel implementations. Added D=512 template instantiations for `f16-f16`, `iso3_0-iso3_0`, `planar3_0-planar3_0`, `planar3_0-f16`, and the **asymmetric** `f16-iso3_0` + `f16-planar3_0` (required because Gemma 4's pre-attention K-rotation materializes K as f16 before FA, so the dispatcher sees `(K=F16, V=ISO3_0)` even for "symmetric" configs). The D=512 patch is small, additive, and should be safe to upstream to johndpope.

Branch: `local/rebase-attempt` (not pushed).

### Spark is still open

Same two cherry-picks should work on Spark, plus possibly a D=512 CUDA kernel patch similar to the 5090's for symmetric iso3/iso3. Not attempted.

## Composite hypothesis outcome table

| Hypothesis (across docs) | M4 Max | Spark | 5090 |
|---|---|---|---|
| iso3/iso3 is 10-25% slower than platform default | ❌ Gemma 4: runaway-generation bug (Metal V-dequant TODO). ⚠️ Qwen 27B Opus: symmetric planar3 −15% vs f16 | ❌ Qwen: −3%. GLM: −25% | ✅ Qwen/Qwopus (~−10%). ❌ Harmonic (+0.3%). ✅ Gemma 4 31B (−21% coding workload, no context gain) |
| planar3/f16 is the sweet spot | ⚠️ Split: **+19% on dense GQA (Qwen 27B Opus)** at same quality, **tied** on Gemma 4 MoE and Gemma 4 31B dense at matched base/ub. Original "+13-20% on all three" claim was a cross-base artifact. | ✅ on Qwen. ❌ output broken on GLM | ✅ mostly (−3.5% to −5.9%, quality fine) |
| Quality is noise-dominated | ✅ K-only is clean. ⚠️ symmetric on Qwen 27B showed +5-point path flip | ✅ same finding (+5 on Qwen, −15 on GLM) | ✅ held on Qwen family (max 1-test delta). ✅ held on Gemma 31B: 22/22 best at both 32K and 128K |
| TurboQuant remains the default | ⚠️ Split: plain f16 is new default on Gemma 4 (base upgrade eliminated turbo4 requirement). planar3/f16 beats f16 only on dense GQA 27B. | n/a (f16 was Spark baseline, not turbo4) | ✅ for Qwen/Qwopus/Harmonic. ❌ **but turbo4 wins on Gemma 4 31B too** — iso3 gives no context advantage |
| Gemma 4 31B-IT throughput unlock | ❌ No rotorquant contribution — matched-base f16 (15.3) ties planar3/f16 (15.17). Real +30% unlock came from the base upgrade. | not tested | ❌ iso3 is −21% slower than turbo4 with no context gain |
| Gemma 4 31B-IT long-context unlock | ⚠️ blocked on symmetric V-dequant bug; K-only can't extend context because V stays at f16 | not tested | ❌ **invalidated** — turbo4 fits full 262K at 28 GB / 46.4 tok/s |

---

# M4 Max (Metal)

Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X @ 410 GB/s, ~28.6 GB Metal free working set.

## Mechanism: why the M4 Max was genuinely uncertain before running

Unlike the Spark (where we had strong priors that KV quantization is net-negative on MoE) and the 5090 (where TurboQuant is well-tuned), M4 Max had a surprising prior: **TurboQuant turbo4 was measured 23% slower than f16** on Gemma 4 26B-A4B Q6_K. Working explanation: Metal is compute-bound on dequant per element, not bandwidth-bound; weights bandwidth dominates the per-token budget, so compression compute overhead exceeds bandwidth savings. Rotorquant's ~64× lighter dequant kernel could plausibly invert the sign.

Direction was uncertain. Pessimistic: Apple Silicon's compute headroom is even tighter than the 5090's. Optimistic: rotorquant's proportional relief should recover much of the turbo4 gap, possibly closing it.

## Matched-base measurements (three models, planarquant fork + Gemma 4 cherry-picks, default `-ub 512`)

| Model | f16/f16 | planar3/f16 | Δ | Quality |
|---|---|---|---:|---|
| Qwen 3.5 27B Opus Q4_K_M (dense GQA) | 13.0 tok/s | **15.5 tok/s** | **+19%** ✅ | 11/17 → 11/17 (identical) |
| Gemma 4 26B-A4B Q6_K (MoE sliding window) | **65.6 tok/s** | 64.5 tok/s | −1.5% (noise) | 15/17 → 15/17 (identical) |
| Gemma 4 31B-IT Q4_K_M (dense full attention ISWA) | **15.3 tok/s** | 15.17 tok/s | −1.0% (noise) | 17/17 → 17/17 (identical) |

**Model-class pattern**: rotorquant K-only pays out when the KV cache generates meaningful decode bandwidth (dense attention, full GQA heads, large contexts). On Gemma 4 26B-A4B the sliding-window attention keeps KV tiny; on Gemma 4 31B-IT the ISWA mix keeps averaged KV moderate and the Givens dequant compute cancels the bandwidth win. Metal's cheap-bandwidth-expensive-compute asymmetry matters — but for rotorquant to exploit it, the model has to actually be KV-bandwidth-bound, which Gemma 4 isn't.

## Full measurement matrix

### Qwen 3.5 27B Opus-Distilled Q4_K_M (dense GQA, `qwen35` arch)

Baseline: 13 tok/s @ 11/17 at 32K f16 with default `-ub 512`.

| Config | Ctx | `-ub` | Avg tok/s | Score | Δ vs baseline |
|---|---|---|---|---|---|
| f16 / f16 (baseline) | 32K | 512 | 13.0 | 11/17 | — |
| **planar3 / f16** (K-only) | 32K | 512 | **15.5** | **11/17** | **+19% speed, same quality** |
| planar3 / planar3 (symmetric) | 32K | 512 | 11.0 | 16/17 | −15% speed, +5 quality (path flip) |
| f16 / f16 at 65K | 65K | 256 | 15.4 | 11/17 | +18% (mostly from `-ub 256`, not context) |
| planar3 / planar3 at 65K | 65K | 256 | 10.8 | 16/17 | −17% speed, same 16/17 |
| planar3 / planar3 at 256K | 256K | 256 | **1.7** (2 benchmarks timed out) | 6/6 (LRU only) | **Context fits but decode collapses** |

### Gemma 4 26B-A4B Q6_K (MoE, sliding window + global, `gemma4` arch)

Baseline: 60.3 tok/s @ 15/17 at 16K f16 (old base); 46.0 tok/s @ 16/17 at 16K turbo4 (old base).

| Config | Ctx | `-ub` | Avg tok/s | Score | Δ (matched base/ub) |
|---|---|---|---|---|---|
| f16 / f16 (new base) | 32K | 512 | **65.6** | 15/17 | — |
| **planar3 / f16** (K-only) | 32K | 512 | 64.5 | 15/17 | −1.5% (tied) |
| planar3 / planar3 (symmetric) | 16K | 512 | 12.3 | 0/5 ExprEval DNF | **Runaway generation bug** |

### Gemma 4 31B-IT Q4_K_M (dense full attention ISWA, `gemma4` arch)

Baseline: 11.8 tok/s @ 17/17 at 32K turbo4 (old base, required `-ub 256` workaround).

| Config | Ctx | `-ub` | Avg tok/s | Score | Δ (matched base/ub) |
|---|---|---|---|---|---|
| turbo4 / turbo4 (old base, `-ub 256`) | 32K | 256 | 11.8 | 17/17 | — (pre-correction baseline) |
| f16 / f16 (new base) | 32K | 512 | **15.3** | 17/17 | +30% from base upgrade alone |
| **planar3 / f16** (K-only, new base) | 32K | 512 | 15.17 | 17/17 | −1.0% vs f16 on new base |
| planar3 / planar3 (new base) | 32K | — | (aborted) | — | Same runaway bug as Gemma 26B-A4B |

### Context ceilings on Gemma 4 31B-IT (new base, default ub)

| KV config | 32K | 64K | 128K | 256K |
|---|---|---|---|---|
| f16 / f16 | ✅ 21.7 GB | ✅ 24.3 GB | ❌ 29.4 GB | — |
| planar3 / f16 K-only | ✅ +0% vs f16 | likely ✅ | ❌ 30.5 GB (no mem savings) | — |
| turbo4 / turbo4 (new base) | ✅ ~18.5 GB | ✅ — | ✅ 21.0 GB | ✅ 24.1 GB |

**What gives you more context on Gemma 4 31B-IT:**
- From 16K to 64K: the base upgrade (4× free)
- From 64K to 128K+: **turbo4 on the new base** — compresses at allocation time
- Rotorquant K-only stays at 64K ceiling (deferred quantization → no allocation memory savings)

## Why K-only works and symmetric doesn't on Gemma 4

Symmetric `planar3/planar3` on Qwen 27B Opus is slow but functional (16/17 quality, 11 tok/s). On **both** Gemma 4 models symmetric hits a different failure: the model generates tokens normally but never emits a stop token, hitting `max_tokens=16384` with `finish=length` — 16K of unstructured content instead of a coding answer. Reasoning is confirmed off (`think=0c`), so this isn't the reasoning-budget issue from before the template fix.

Best guess: the symmetric V-dequant path on Metal perturbs numerical noise enough to push Gemma 4's generation off the typical stop-token probability peak. K-only sidesteps this because V stays at f16 — no V-side perturbation, Gemma 4's stop logic lands where it expects.

This is a **Metal V-dequant bug**, not a general rotorquant limitation. On CUDA, symmetric planar3/planar3 works fine on Llama 3.1 8B. The symmetric V-dequant inverse-rotation fix that landed for CUDA in `6e5a4aae4` is listed as a TODO for Metal in rotorquant's CLAUDE.md. Our measurements strongly suggest that TODO is the root cause; porting the fix would unlock symmetric on Gemma 4.

## M4 Max recommendations

- **Dense GQA ≥27B**: use `-ctk planar3 -ctv f16`. Measured +19% vs f16 at identical quality on Qwen 3.5 27B Opus-Distilled.
- **Gemma 4 (26B-A4B or 31B-IT)**: use plain `-ctk f16 -ctv f16` on the new base. Rotorquant K-only is tied at matched ub — no reason to opt into the complexity. Default ub=512 works fine on the new base; no `-ub 256` workaround needed.
- **Need 128K+ context on Gemma 4 31B-IT**: switch to `-ctk turbo4 -ctv turbo4` (compresses at allocation time, unlike K-only which is deferred).
- **Never** use symmetric `planar3/planar3` on Gemma 4 until the Metal V-dequant fix lands — runaway-generation bug.

## Build instructions (M4 Max)

```bash
cd ~/git/TheTom/llama-cpp-planarquant  # worktree or clone of johndpope/llama-cpp-turboquant
git checkout johndpope/feature/planarquant-kv-cache
git cherry-pick 63f8fe0ef  # upstream PR #21309: Gemma 4 architecture
# In tools/mtmd/mtmd.cpp conflict: take HEAD (revert to planarquant's version)
# Also revert tools/mtmd/{CMakeLists.txt, clip-graph.h, clip-impl.h, clip-model.h, clip.cpp, models/models.h}
# and delete tools/mtmd/models/gemma4v.cpp (mtmd bits reference mtmd-image.h not on planarquant base)
git cherry-pick 5208e2d5b  # upstream PR #21326: Gemma 4 chat template fix (clean)
cmake --build build --target llama-server -j 8
```

---

# DGX Spark (aarch64 CUDA 13)

NVIDIA DGX Spark (GB10 Blackwell, compute cap 12.1, 128 GB LPDDR5X @ ~273 GB/s), CUDA 13.0, sm_121a.

## Results

| Model | KV config | Tok/s | Δ vs f16 | Score | Δ | Verdict |
|---|---|---:|---:|---:|---:|---|
| Qwen3.5-122B-A10B Q4_K_M (bartowski) | `f16 / f16` baseline | 25.8 | — | 13/17 | — | — |
| Qwen3.5-122B-A10B Q4_K_M | `iso3 / iso3` | 23.7–25.0 | −3% to −8% | 16/17 | +3 | Net win |
| Qwen3.5-122B-A10B Q4_K_M | `planar3 K / f16 V` | 25.2–25.6 | ~−1% | **18/17** | **+5** | Clear win |
| GLM-4.5-Air Q4_K_M (bartowski) | `f16 / f16` baseline | 21.7 | — | 15/17 | — | — |
| GLM-4.5-Air Q4_K_M | `iso3 / iso3` | 16.2 | **−25%** | 0/17 (broken) | −15 | **Broken output** |
| GLM-4.5-Air Q4_K_M | `planar3 K / f16 V` | 20.1 | −7% | 0/17 (broken) | −15 | **Broken output** |

Per-benchmark for Qwen3.5-122B:

| Benchmark | f16 | iso3/iso3 | planar3K/f16V |
|---|---|---|---|
| Expression Evaluator | **0/5** (self.tokens accumulator bug) | **5/5** | **5/5** |
| A* Pathfinding | 7/6 | 7/6 | 7/6 |
| LRU Cache with TTL | 6/6 | 4/6 | 6/6 |
| **Total** | **13/17 (76%)** | **16/17 (94%)** | **18/17 (106%)** |

## What happened on Qwen3.5-122B

**Throughput was essentially free.** iso3/iso3 at 23.7–25.0 tok/s vs 25.8 f16 is −3% to −8%, vs the predicted −20% to −50%. Planar3-K-only was within 1% of baseline across all three benchmarks. Dramatically better than the `f16K/q8V` asymmetric from the earlier KV experiment (−46%).

Why the prediction missed: extrapolated from the q8_0 V-cache result assuming dequant compute on Spark CUDA 13 was roughly constant per element. That was wrong. RotorQuant's fused rotate+dequant kernels in the Flash Attention path beat the generic q8_0 V-dequant path on bandwidth-bound MoE decode.

**Quality went up.** Headline result. The bartowski + mainline baseline scores 13/17 because of a specific implementation bug on Expression Evaluator — the model chooses a `self.tokens` list accumulator pattern and fails to reset state between calls, making test 2+ fail. Every rotorquant config landed the model on a *different* generation path that produces correct code.

**This is not evidence that rotorquant is inherently higher quality.** It's evidence that at temp=0, Qwen3.5-122B on this hardware is **pathologically sensitive to numerical noise**: the bartowski + mainline kernels happen to sit in a local pocket where the model makes a specific coding mistake. Perturb the KV numerics in almost any direction (ik-llama's different kernels, rotorquant's rotation-quant, even the q8_0 asym result which scored 14/17) and the model snaps into a different generation path — sometimes better, sometimes worse. RotorQuant got lucky. It does not improve the model; it shifts noise enough to jump out of the bad pocket.

The earlier KV q8_0 asymmetric experiment hinted at this (14/17 vs 13/17) and we'd written it off as within-noise. This experiment **confirms the noise band is ±5 points, not ±1** — a huge finding for how we interpret single-shot temp=0 results on this platform going forward.

## What happened on GLM-4.5-Air

All three code benchmarks timed out at 600s, generating >9,700 tokens without a natural stop. Throughput test hit `max_tokens=2048` (vs 875 on the f16 baseline — 2.3× output inflation, still never stopping). Every code benchmark errored with zero saved output.

Manual confirmation with the smallest prompt:

```
>>> Say the two words hello world. Then stop.
```

Result from GLM-4.5-Air + `planar3 K / f16 V` at temp 0:

```
FINISH: length
CONTENT: 'Hello???????????????????????????????????????????????????????????????????????????????'
USAGE: {'completion_tokens': 80, 'prompt_tokens': 19}
```

Eighty tokens of `?` after "Hello". Finish reason `length` = hit max_tokens, never stopped. **The model does not recognize its own stop token when its K cache is rotated-and-quantized, even in the lightest K-only mode** that the rotorquant paper claims is "zero PPL loss."

iso3/iso3 is the same but worse (throughput −25%, all code benchmarks timed out identically).

### Why GLM breaks but Qwen doesn't

Both models are ~120B total / ~10-12B active MoE with GQA. The difference is attention structure:

- **Qwen3.5-122B-A10B** is a DeltaNet hybrid: only **12 of 48 layers** do full attention with a KV cache. The other 36 are linear-attention layers whose recurrent state is outside the rotorquant quantization path. Only ~25% of the model's attention sees the KV quantization noise.
- **GLM-4.5-Air** uses standard attention in **every one of its 46 layers**. Every attention op reads a rotated-quantized K cache, every op accumulates the perturbation, and error compounds across the full depth.

At layer-level math this is 4× difference in how much of the forward pass is affected. The rotorquant paper was validated on Llama 3.1 8B (32 standard-attention layers, closer to GLM's shape than Qwen's). But Llama 3.1 8B has 8× fewer total layers; the decomposition claim rests on the KV vector's directional structure surviving small rotations, and on a 46-layer dense attention stack the error evidently stops being small.

The failure mode is striking: not subtly-wrong code, literal `?` tokens. Characteristic of logit corruption so severe that sampling finds no coherent continuation and keeps emitting the same high-probability filler. Does **not** match the "graceful PPL degradation" story the paper tells for Llama 3.1 8B (PPL 6.63 → 6.91 = 4.2% for iso3/iso3).

## Spark recommendations

- **Qwen3.5-122B-A10B (DeltaNet hybrid)**: use `-ctk planar3 -ctv f16` as a drop-in replacement for `-ctk f16 -ctv f16`. 18/17 at 25.2–25.6 tok/s is materially better than the 13/17 mainline baseline on both axes, and ties the S-tier ik-llama result (17/17 @ 26 tok/s) with a simpler engine stack. **Caveat**: almost certainly a numerical-noise coincidence; one single-shot temp=0 result isn't a general quality claim. If you already run single-shot temp=0 as standard, this is the best single-shot config we've measured.
- **GLM-4.5-Air**: do not use rotorquant in any config. Not iso3, not planar3, not K-only. Output is destroyed.
- **Follow-up**: re-run Qwen3.5-122B rotorquant at temp=0.3 best-of-3 to distinguish "rotorquant is actually a win here" from "we got lucky with noise." If mean over 3 runs holds above 14/17 for planar3K, it's a genuine improvement on the bartowski + mainline baseline.

---

# RTX 5090 (Windows + WSL2 client, CUDA 13.2)

NVIDIA RTX 5090 32 GB GDDR7, Blackwell sm_120a. `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` rebased onto current upstream for Gemma 4 support + D=512 FA vec kernel patch.

## Summary

| Model | Config | Best-of-3 | Avg | Decode | Baseline (turbo4) | Δ decode |
|---|---|---:|---:|---:|---:|---:|
| Qwen 3.5 27B Opus-Distilled Q4_K_M | iso3/iso3 | 21/22 | 17.0/22 | 53.6 t/s | 60.0 t/s | **−10.6%** |
| Qwen 3.5 27B Opus-Distilled Q4_K_M | planar3/f16 | 22/22 | 19.0/22 | 57.4 t/s | 60.0 t/s | **−4.3%** |
| Qwopus 3.5 27B-v3 Q6_K | iso3/iso3 | 22/22 | 19.3/22 | 44.5 t/s | 49.6 t/s | **−10.4%** |
| Qwopus 3.5 27B-v3 Q6_K | planar3/f16 | 22/22 | 18.3/22 | 46.6 t/s | 49.6 t/s | **−5.9%** |
| Harmonic 27B Q4_K_M (think on) | iso3/iso3 | 22/22 | 19.7/22 | **61.5 t/s** | 61.3 t/s | **+0.3%** |
| Harmonic 27B Q4_K_M (think on) | planar3/f16 | 22/22 | 20.3/22 | 59.1 t/s | 61.3 t/s | −3.5% |
| Gemma 4 31B-IT Q4_K_M (rebased, D=512 patch) | iso3/iso3 | 22/22 | 20.4/22 | 39.7 t/s | 50.3 t/s (turbo4 32K) | **−21.1%** |

## Hypothesis verdicts

**H1 (iso3/iso3 is 10-25% slower than turbo4)**: ✅ held for Qwen 27B Opus-Distilled and Qwopus (−10.6%, −10.4%), **violated by Harmonic 27B (+0.3%)**. Genuine surprise — the first datapoint where rotorquant's 10.3× compression doesn't cost throughput vs turbo4's 3.8×. Possibilities: (a) Harmonic's thinking-heavy generation changes the K/V read/write ratio favorably, (b) session variance (only 3 runs), (c) architecture-specific alignment of Harmonic's head dim with iso3's 4D quaternion blocks.

**H2 (planar3/f16 within 5% of turbo4)**: ✅ mostly held (−3.5%, −4.3%, −5.9%). Qwopus just outside 5% but inside the 10% secondary prediction. Tight band across all 3 models matches the paper's −4% measurement on 8B — most reproducible result in the experiment. **Lowest-risk rotorquant config.**

**H3 (quality within noise)**: ✅ held decisively. Every (model, config) hit 22/22 best-of-3 **except** Qwen 27B Opus-Distilled iso3/iso3 (21/22, dropped one test in Expression Evaluator). Averages lower (17-20/22) because Qwen and Qwopus have high per-run variance at temp 0.3 regardless of KV backend — that's a model property, not rotorquant property.

**H4 (turbo4 remains default)**: mixed.
- **Qwen 27B Opus-Distilled, Qwopus 27B**: turbo4 still wins. iso3/iso3 costs ~10% for zero context gain (both already hit 262K native). Keep turbo4.
- **Harmonic 27B Q4_K_M with thinking on**: iso3/iso3 ties turbo4 on throughput (+0.3%) AND matches on quality (22/22). iso3 compresses 2.7× more aggressively at no throughput cost — **could become the recommended Harmonic config if a 5-run replication confirms the tie.** Not acting on it yet.

**H5 (Gemma 4 31B-IT context unlock)**: ❌ **invalidated**. Turbo4 fits the full 262K native window at 28 GB VRAM with 4 GB headroom, 46.4 tok/s, no spill cliff. The "~58K turbo4 ceiling" was based on an ~10× overestimate of per-token KV cost (870 KB/tok f16 assumed all 60 layers are full-attention; reality is 10 non-SWA layers with ~4 shared-KV heads each → ~22 KB/tok turbo4). Rotorquant iso3 provides zero context advantage and is strictly slower (39.7 vs 46.4 tok/s at 262K). The rebase + D=512 kernel work proved the fork *can* support Gemma 4 with rotorquant, but the use case it was built for doesn't exist on the 5090.

## The Harmonic anomaly

Harmonic 27B Q4_K_M with `-rea on` is the only model where iso3/iso3 doesn't cost throughput vs turbo4:

| Benchmark | iso3/iso3 | planar3/f16 |
|---|---:|---:|
| Expression Evaluator | 56.4 | 62.1 |
| A* Pathfinding | 62.9 | 58.4 |
| LRU Cache with TTL | 62.9 | 58.0 |
| String Processor | 63.8 | 58.1 |
| **mean** | **61.5** | **59.1** |

This ordering (iso3 > planar3) is opposite of the other two models and contradicts the paper's Llama 3.1 8B ordering (planar3 > iso3). Expression Evaluator on iso3 ran at 56.4 t/s, the others at 62.9-63.8 — an 8 t/s swing within the same server session. If Expression Evaluator sampled a cold GPU and the others sampled warm, the iso3 mean is an underestimate.

**Action**: before declaring iso3/iso3 the new Harmonic default, re-run Harmonic × iso3/iso3 × 5 runs with a clean server restart between each, vs 5 turbo4 runs measured the same way. +0.3% is too small to act on from 3 runs.

## Gemma 4 31B-IT context sweep (rebased fork)

Context sweep with iso3/iso3 after the rebase + D=512 kernel patch:

| Context `-c` | VRAM (MB) | Smoke-test decode | Status |
|---:|---:|---:|---|
| 32,768 (32K) | 23,835 | 44.0 tok/s | works, no spill |
| 131,072 (128K) | 29,127 | 43.9 tok/s | works, no spill |
| 163,840 (160K) | 30,284 | 43.3 tok/s | works, no spill |
| 196,608 (192K) | 31,723 | 9.1 tok/s | **KV spills to RAM, cliff** |
| 262,144 (262K native) | 31,524 | 7.4 tok/s | spills, worse cliff |

Cliff between 160K and 192K — pre-allocated KV exceeds the 32 GB budget alongside ~21 GB weights + compute buffer. Decode collapses 5× because every token traverses PCIe for the spilled portion.

Then we went back and measured **turbo4 at the same contexts** on the same rebased binary:

| Context `-c` | turbo4 VRAM (MB) | Decode | Status |
|---:|---:|---:|---|
| 98,304 (96K) | 24,323 | 46.3 tok/s | works |
| 163,840 (160K) | 25,724 | 46.9 tok/s | works |
| 262,144 (262K native) | **28,025** | **46.4** | **works, no spill, 4 GB headroom** |

**Turbo4 fits the full 262K window with headroom.** The "58K ceiling" didn't exist. H5 invalidated.

Coding-suite quality at 32K and 128K (Expression Evaluator only for sanity check):

| Benchmark (32K) | Best | Avg | Decode |
|---|---:|---:|---:|
| Expression Evaluator | 5/5 | 4.7/5 | 39.4 |
| A* Pathfinding | 6/6 | 6.0/6 | 39.3 |
| LRU Cache with TTL | 6/6 | 6.0/6 | 39.5 |
| String Processor | 5/5 | 3.7/5 | 40.5 |
| **Total** | **22/22** | **20.4/22** | **39.7** |

At 128K: Expression Evaluator 5/5 best, 4.7/5 avg, 39.6 tok/s. **Quality holds at both context sizes.** Throughput is stable (39.7 vs 39.6) — long context doesn't add an extra decode tax on top of rotorquant's base cost.

The single-prompt smoke test at 44 tok/s vs coding-suite 39.7 tok/s reconciles as warm-cache short-decode vs sustained-decode. Realistic number for Gemma 31B + iso3 on coding workloads is **~39.7 tok/s**, a −21.1% delta vs turbo4's 50.3 baseline — inside H1's 10-25% band, at the upper end.

## Correction: Gemma 4 26B-A4B doesn't need rotorquant for context

The original hypothesis projected Gemma 4 26B-A4B Q6_K as "turbo4 max ~230K, iso3 would unlock full 262K, +32K gain." That projection was based on `CONTEXT_CAPACITY_5090.md`'s architecture summary listing 15 global-attention layers at head_dim 256. The live gemma4 loader reports 5 global + 25 SWA, head_dim 512, 8 KV heads. Recomputed:

- Per-token global-attn KV at f16: ~80 KB (80 × 5 layers)
- At turbo4 (3.8× vs f16): ~21 KB/tok → 262K × 21 KB = 5.5 GB → **fits full 262K** in ~26.5 GB total
- At iso3 (10.3×): ~7.8 KB/tok → 262K × 7.8 KB = 2.0 GB → fits in ~23 GB total

**Turbo4 already saturates 262K on Gemma 4 26B-A4B Q6_K with 5.5 GB headroom.** Rotorquant gives more headroom but zero additional usable context. The original "+32K gain" projection was wrong because the architecture table it was based on was wrong.

## 5090 recommendations

Keep TurboQuant turbo4 as the default for every model. Rotorquant provides no context or throughput advantage:

- **Gemma 4 31B-IT Q4_K_M**: turbo4 fits full 262K at 28 GB / 46.4 tok/s. Iso3 is strictly slower (39.7) with zero context gain.
- **Gemma 4 26B-A4B Q6_K/Q4_K_M**: turbo4 fits full 262K with ~5 GB headroom.
- **Qwen 27B Opus-Distilled, Qwopus 27B**: iso3 ~10% slower than turbo4 with zero context gain. Planar3/f16 ~5% slower with less compression.
- **Harmonic 27B**: iso3/iso3 ties turbo4 (+0.3%) but the gap is too small without replication. Keep turbo4 unless 5-run replication confirms.

---

# Practical recommendations summary

| Platform | Model class | Recommended KV config |
|---|---|---|
| M4 Max | Dense GQA ≥27B (Qwen 27B Opus) | **`-ctk planar3 -ctv f16`** (+19%) |
| M4 Max | Gemma 4 (any size) | `-ctk f16 -ctv f16` (plain f16, new base) |
| M4 Max | Gemma 4 31B-IT at 128K+ ctx | `-ctk turbo4 -ctv turbo4` (not rotorquant) |
| Spark | Qwen3.5-122B DeltaNet hybrid | **`-ctk planar3 -ctv f16`** (ties S-tier ik-llama) |
| Spark | GLM-4.5-Air or any deep dense-attention MoE | **Never rotorquant** — broken output |
| 5090 | All models | `-ctk turbo4 -ctv turbo4` (rotorquant no-op or regression) |

---

# Open questions / follow-ups

1. **Rebase planarquant onto Gemma 4-capable llama.cpp on M4 Max and Spark.** Done on 5090 via rebase + D=512 kernel patch; done on M4 Max via two-commit cherry-pick. Spark still open. The cherry-pick path is much lighter-weight than the full rebase and should transplant to Spark easily (possibly with a D=512 CUDA kernel patch similar to the 5090's for symmetric iso3/iso3).

2. **Why does Harmonic 27B escape the throughput tax on the 5090?** A 5-run replication of Harmonic × iso3/iso3 vs Harmonic × turbo4 with matched server restarts would resolve whether it's a real architecture/thinking-mode effect or session variance. If real, the mechanism could be Harmonic's attention head dimensions aligning with iso3's 4D quaternion blocks — worth a follow-up and potentially a published counter-point to the "rotorquant is always a tax on CUDA" prior.

3. **Why does GLM-4.5-Air break catastrophically under any rotorquant config?** The failure mode (literal `?` spam, no stop tokens) is much more severe than the paper's 4.2% PPL degradation on Llama 3.1 8B. Two plausible causes: (a) error accumulation across 46 dense-attention layers (vs Llama 3.1 8B's 32), if per-layer rotation error is larger than the paper suggested; (b) a Metal/CUDA kernel bug in the rotation step specific to GLM's head dimensions or group size. A single PPL measurement on wikitext-2 for GLM + planar3K would separate the "accumulation" hypothesis from the "kernel bug" hypothesis. Worth filing upstream.

4. **Rotorquant + thinking-mode models.** Harmonic 27B is the only thinking-mode model tested. Its anomalous 5090 result (iso3 beats turbo4) plus GLM's specific failure mode (stop-token recognition breaks) both suggest rotorquant's quality and throughput behavior could differ for thinking-mode workloads in ways the paper's single-dense-8B baseline cannot predict. A broader thinking-mode evaluation (Harmonic, GPT-OSS 20B, Qwen 32B thinking) would be valuable before adopting rotorquant as a default for any thinking-heavy pipeline.

5. **The noise band is bigger than assumed.** Spark's experiment exposed a ±5 point noise band at temp=0 single-shot, retroactively weakening any claim based on a single 17-test single-shot run — including parts of [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md). Future KV quantizer evaluations should default to best-of-3 at temp 0.3 or explicitly report multi-seed variance.

6. **Port Metal V-dequant symmetric fix.** The `6e5a4aae4` CUDA fix that made symmetric planar3/iso3 work on Llama 3.1 8B is listed as a TODO for Metal. Porting it would unlock symmetric configs on Gemma 4 on the M4 Max and close the single biggest rotorquant coverage gap on that platform.

---

# Appendix: M4 Max pre-experiment projections

The [ROTORQUANT_HYPOTHESIS_M4MAX.md document](https://github.com/gisenberg/local-model-eval/commits/main/results/ROTORQUANT_HYPOTHESIS_M4MAX.md) (no longer stored as a separate file) contained the pre-experiment projection table showing where rotorquant's context benefit was expected to concentrate on M4 Max:

| Model | Current max ctx | With planar3 (projected) | Practical win? |
|---|---|---|---|
| Nemotron 4B | 32K | 32K | No (already huge) |
| Qwen 3.5 9B | 32K | 32K | No (already huge) |
| Gemma 4 26B-A4B Q6_K | 32K | 32K | **No** (compute buffer, not KV) |
| Gemma 4 26B-A4B Q4_K_M | 32K | 32K-48K | Minor |
| Qwen 3.5 27B Opus Q4 | 32K | **128K-256K** | **Yes, biggest win** |
| Gemma 4 31B-IT Q4 | 32K (turbo4) | **48-64K** | **Yes, with quality trade** |

Actual outcomes after the experiment:

- **Qwen 27B Opus**: f16 already reached 128K on this model (the compute-buffer scaling formula derived from Gemma 4's sliding-window pattern doesn't generalize — Qwen 27B Opus has a near-constant 250-420 MiB compute buffer). Symmetric planar3 at 256K loads but decode collapses to 1.7 tok/s. Practical ceiling ~128K either way — **no usable context gain**.
- **Gemma 4 31B-IT**: the real ceiling improvement came from the base upgrade, not rotorquant. With the new base, f16 fits 64K comfortably; turbo4 reaches 128K+. Rotorquant K-only has deferred quantization, so it doesn't reduce allocation memory, capping it at roughly the same 64K ceiling as f16.

The hypothesis doc's main intellectual contribution was framing the "compute buffer dominates the working set on Metal" prior that made the outcome uncertain in the right direction: rotorquant's 64× lighter dequant kernel was specifically a better match for Metal's compute-bound-on-dequant bottleneck than turbo4's WHT butterfly. The matched-base result on Qwen 27B Opus (+19% vs f16) validates that mechanism in the specific model class where it matters — dense GQA with enough KV cache to generate meaningful per-token bandwidth.

## See also

- [TURBOQUANT.md](TURBOQUANT.md) — the KV quantizer this is measured against
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md), [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md), [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — per-platform tier lists with the baselines rotorquant is measured against
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — the compute-buffer bottleneck that rotorquant cannot solve on Metal
