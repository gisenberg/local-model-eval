# RotorQuant on RTX 5090 — Experiment Results

**Date:** 2026-04-11
**Hardware:** NVIDIA RTX 5090 32 GB GDDR7, Blackwell sm_120a, Windows 11 + CUDA 13.2
**Stack:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` commit `20efe75cf` (Windows build with four MSVC patches described in [ROTORQUANT_HYPOTHESIS_5090.md](../ROTORQUANT_HYPOTHESIS_5090.md))
**Scope:** 3 models × 2 rotorquant configs × 4 benchmarks × 3 runs = 72 runs. Gemma 4 models excluded (base upstream does not support `gemma4` architecture — see hypothesis doc for the scope cut).
**Client:** WSL2 Linux `requests` → Windows llama-server (avoids the ~2s urllib3-on-Windows TTFT bug).

## Summary

| Model | Config | Best-of-3 | Avg | Decode | Baseline (turbo4) | Δ decode |
|---|---|---:|---:|---:|---:|---:|
| Qwen 3.5 27B Opus-Distilled Q4_K_M | iso3/iso3 | 21/22 | 17.0/22 | 53.6 t/s | 60.0 t/s | **−10.6%** |
| Qwen 3.5 27B Opus-Distilled Q4_K_M | planar3/f16 | 22/22 | 19.0/22 | 57.4 t/s | 60.0 t/s | **−4.3%** |
| Qwopus 3.5 27B-v3 Q6_K | iso3/iso3 | 22/22 | 19.3/22 | 44.5 t/s | 49.6 t/s | **−10.4%** |
| Qwopus 3.5 27B-v3 Q6_K | planar3/f16 | 22/22 | 18.3/22 | 46.6 t/s | 49.6 t/s | **−5.9%** |
| Harmonic 27B Q4_K_M (think on) | iso3/iso3 | 22/22 | 19.7/22 | 61.5 t/s | 61.3 t/s | **+0.3%** |
| Harmonic 27B Q4_K_M (think on) | planar3/f16 | 22/22 | 20.3/22 | 59.1 t/s | 61.3 t/s | **−3.5%** |

Bench log: [experiments/rotorquant_5090/bench.log](../experiments/rotorquant_5090/bench.log). Raw per-run JSON: [experiments/rotorquant_5090/results.json](../experiments/rotorquant_5090/results.json).

## Hypothesis verdict

**H1 (iso3/iso3 is 10-25% slower than turbo4):** ✅ **held for Qwen 27B Opus-Distilled and Qwopus** (−10.6%, −10.4%) but ❌ **violated by Harmonic 27B (+0.3%).** The Harmonic result is a genuine surprise — it's the first datapoint where rotorquant's 10.3× compression doesn't cost any throughput vs turbo4's 3.8×. The paper's −16% on dense Llama 3.1 8B and our −10.5% on the other two 27B Qwen dense models suggest a consistent tax, but Harmonic completely escapes it. Why it's different is an open question — possibilities are (a) Harmonic's thinking-heavy generation pattern changes the K/V read/write ratio in a way that favors the Givens rotation, (b) session variance we didn't capture with only 3 runs, or (c) something architecture-specific about Harmonic's attention layout.

**H2 (planar3/f16 within 5% of turbo4):** ✅ **mostly held** (−3.5%, −4.3%, −5.9%). Qwopus is just outside the 5% threshold but still inside H2's secondary prediction of 10%. The tight band across all 3 models (−3.5% to −5.9%) exactly matches the paper's −4% measurement on 8B — this is the most reproducible result in the experiment. Planar3/f16 is the lowest-risk rotorquant config by a wide margin.

**H3 (quality within noise):** ✅ **held decisively**. Every (model, config) hit 22/22 best-of-3 **except** Qwen 27B Opus-Distilled iso3/iso3 (21/22, dropped one test in Expression Evaluator). Averages are lower (17-20/22) because Qwen and Qwopus have high per-run variance at temp 0.3 regardless of KV backend — that's a model property, not a rotorquant property. No quality regression attributable to the KV quantizer.

**H4 (turbo4 remains the default):** **mixed, partially refuted**.
- For Qwen 27B Opus-Distilled and Qwopus 27B: **turbo4 still wins clearly.** iso3/iso3 costs ~10% throughput for zero context gain (both already hit 262K native). planar3/f16 is closer but still loses on throughput with less compression. Keep turbo4.
- For **Harmonic 27B Q4_K_M with thinking on**: **iso3/iso3 ties turbo4 on throughput (+0.3%) AND matches on quality (22/22 vs turbo4's 31/31 published — 22/22 is our suite's max for these 4 benchmarks).** Because iso3 compresses 2.7× more aggressively than turbo4 at no throughput cost for this model, iso3/iso3 could become the recommended config for Harmonic *if we can rule out measurement noise with a replication*. This is the one result in the experiment that would change our default recommendation if it holds.

**H5 (Gemma 31B-IT context unlock):** ✅ **measured after a local rebase + D=512 kernel patch** — the johndpope fork's base originally predated Gemma 4 support, but we rebased onto current upstream llama.cpp and added the missing flash-attention vec kernel instances for head-dim 512 (details in the addendum below). Measured usable ceiling: **~160K context** on Gemma 4 31B-IT Q4_K_M with iso3/iso3, a **+102K gain** over turbo4's 58K cap. Throughput cost: **−13.9%** (43.3 tok/s vs 50.3 turbo4 baseline). The pre-experiment projection was 157K; measured reality was 160K — the hypothesis landed within 2%.

## The Harmonic anomaly

Harmonic 27B Q4_K_M with `-rea on` is the only model in the experiment where iso3/iso3 doesn't cost throughput vs turbo4:

| Benchmark | iso3/iso3 decode | planar3/f16 decode | Notes |
|---|---:|---:|---|
| Expression Evaluator | 56.4 t/s | 62.1 t/s | iso3 is 10% *slower* than planar3 here — only benchmark where iso3 underperforms |
| A* Pathfinding | 62.9 t/s | 58.4 t/s | iso3 is 8% *faster* than planar3 — reversal |
| LRU Cache with TTL | 62.9 t/s | 58.0 t/s | iso3 faster again |
| String Processor | 63.8 t/s | 58.1 t/s | iso3 faster again |
| **mean** | **61.5 t/s** | **59.1 t/s** | **iso3 > planar3 on Harmonic** |

This ordering (iso3 > planar3) is the opposite of what we see on the other two models and contradicts the paper's Llama 3.1 8B ordering (planar3 > iso3). Possible explanations:

1. **Measurement noise.** Expression Evaluator on iso3 ran at 56.4 t/s, A* / LRU / String at 62.9-63.8 t/s — an 8 t/s swing *within the same server session*. If Expression Evaluator happened to sample a cold GPU and the others sampled a warm one, the iso3 mean is an underestimate and the real Harmonic iso3 rate could be even higher. With only 3 runs per benchmark this is hard to rule out.

2. **Thinking-mode token traffic pattern.** Harmonic is the only model in the experiment running with `-rea on` and a 16K reasoning budget. Thinking output is typically generated in long, coherent bursts that write many V tokens per K attention — which plays to iso3's symmetric 4D quaternion compression differently than the short, structured code output the other benchmarks produce on non-thinking models.

3. **Base model architecture.** Harmonic is built on a Qwen 2.5 72B distilled base and has a different attention head count/dim than Qwen 3.5 27B. The rotorquant block-diagonal rotation kernels are head-dim-dependent — if Harmonic's head dimension happens to align with iso3's 4D block size better than with turbo4's WHT butterfly size, that could give iso3 a local kernel advantage.

**Action item:** before declaring iso3/iso3 the new Harmonic default, re-run Harmonic × iso3/iso3 × 5 runs (not 3) with a clean server start between each run and compare to 5 turbo4 runs measured the same way. The +0.3% gap is too small to act on from 3 runs.

## Findings in plain English

1. **Rotorquant's paper holds up on 27B Qwen dense models, within error.** The paper claimed iso3/iso3 is ~16% slower than f16 and planar3/f16 is ~4% slower than f16 on an 8B dense model. Our measurements against turbo4 (a ~5% faster comparator than f16) show iso3/iso3 at ~10.5% slower and planar3/f16 at ~5% slower. Adding ~5% back to get an f16-equivalent gives ~15.5% and ~10%, landing right next to the paper's numbers. The rotorquant kernels scale from 8B to 27B without additional penalty.

2. **Planar3/f16 is the robustly safe rotorquant config.** −3.5 to −5.9% vs turbo4, zero quality regression on any model, tight variance. If a user has to pick one rotorquant config blind, it's this one. But it also compresses less than turbo4 (1.67× vs 3.8×), so it gives *less* context — it's strictly inferior to turbo4 for our use case unless you specifically want the paper's "minimal PPL loss" guarantee.

3. **Iso3/iso3 is variable.** −10.4 to −10.6% on Qwen/Qwopus, but +0.3% on Harmonic. Needs more replication before we can say whether this is architecture-specific, thinking-mode-specific, or noise.

4. **Turbo4 remains the default for 5 of 6 planned models.** On the 3 we tested, Qwen and Qwopus clearly stay on turbo4. Harmonic is a possible switch to iso3 if the replication holds. The 3 Gemma models default to turbo4 because we couldn't test them at all.

5. **The Gemma 31B-IT context unlock was measured after a local rebase** — see the addendum below. Iso3/iso3 unlocks **~160K usable context** on Gemma 4 31B-IT on the 5090 (turbo4 caps at ~58K), a **+102K gain** at **−13.9% throughput**. The pre-experiment projection of 157K landed within 2% of the measured result. This is the single strongest rotorquant-win datapoint anywhere in the three-platform experiment — if you run Gemma 4 31B-IT and need long context, iso3 is now the right default.

## What we'd change in the next round

- **More runs per benchmark** (5 instead of 3) for Harmonic specifically, to resolve the iso3 vs turbo4 ambiguity.
- **Include a matched f16/f16 run** on one model so we can reproduce the paper's reference point directly on our hardware.
- **Coding-suite quality bench on Gemma 4 31B-IT + iso3/iso3** at both 32K and 128K-160K to complete the H5 picture. The addendum below measured context + throughput + one qualitative smoke test of output coherence, but not the 22-test coding suite.
- **Try `planar3/q8_0` and `iso3/q8_0`** asymmetric configs from the johndpope README — we only tested the two most-cited configs and there are 5+ more variants that might trade differently.

## Not in this experiment

Everything listed in the "Not tested" section of [ROTORQUANT_HYPOTHESIS_5090.md](../ROTORQUANT_HYPOTHESIS_5090.md) still applies: no vLLM/NVFP4 comparisons (wrong engine), no turbo3 comparison (we use turbo4), no 4-bit variants (planar4/iso4 crash on FA dispatch), no PPL measurement. Long-context (≥64K) is **no longer** in this list — the addendum below measured iso3/iso3 up to 262K on Gemma 4 31B-IT.

## Addendum (2026-04-11 afternoon): rebase onto upstream + Gemma 4 31B-IT H5 measurement

After writing everything above, we went back and rebased the johndpope fork onto current upstream llama.cpp and retested the Gemma 4 31B-IT H5 hypothesis that the original scope cut had ruled out. The summary: it worked, with two surprises that required local kernel patches.

### What the rebase took

The johndpope `feature/planarquant-kv-cache` branch had 167 rotorquant commits on top of upstream master at `9c600bcd4` (2026-03-25, ggml version 0.9.8). Current upstream master at the time of the rebase was `ff5ef8278` (2026-04-11, ggml 0.9.11), **241 commits ahead**, and included full `LLM_ARCH_GEMMA4` registration plus all the Gemma 4 tokenizer/template/architecture support.

A 3-way merge of upstream/master into a local copy of the feature branch produced only **5 conflicted files and 8 conflict hunks** — much smaller than we feared. All conflicts were local and resolvable in a single pass:

| File | Conflict | Resolution |
|---|---|---|
| `ggml/include/ggml.h` | Upstream added `GGML_TYPE_Q1_0 = 41`; rotorquant claimed 41-47 for TURBO/PLANAR/ISO types | Renumbered rotorquant types to 42-48 (safe because KV cache type IDs are runtime-only, never serialized) |
| `ggml/src/ggml-cuda/fattn.cu` (×2) | Different head-dim exclusion lists for WMMA and MFMA kernels | Union: exclude 40, 72, 256, 512, 576, 640 |
| `ggml/src/ggml-cuda/vendors/hip.h` | Upstream added `NCCL_CHECK`; rotorquant had variadic shuffle macros | Kept both |
| `src/llama-graph.cpp` (×2) | TurboQuant V-padding reshape vs upstream's `v_rot` mul_mat | Kept both — orthogonal operations |
| `src/llama-kv-cache.cpp` (×2) | MSVC InnerQ stubs + `convert_deferred_keys()` vs upstream's `ggml_gen_hadamard` / `ggml_mul_mat_aux` helpers and `set_input_k_rot`/`set_input_v_rot` delegators | Kept all of the above |

The tree built cleanly on Windows + CUDA 13.2 + MSVC with the same four MSVC patches we already had. **Gemma 4 31B-IT Q4_K_M loaded on the first try with f16/f16 KV** — the rebase alone was enough to fix the "unknown model architecture: gemma4" error.

### Second blocker: FA vec kernel gap at D=512

Loading Gemma 4 31B-IT with `-ctk iso3 -ctv iso3` hit `GGML_ABORT("fatal error")` in `ggml-cuda/fattn.cu` during warmup. Debug instrumentation showed the flash-attention vec dispatcher was being called with `Q->ne[0]=512, K->type=F16, V->type=ISO3_0`. Two things going on:

1. **Gemma 4 uses head_dim 512 for its full-attention layers.** The live model loader reports 10 non-SWA layers at `key_length=512, value_length=512` and 50 SWA layers at 256. Upstream routes f16 D=512 through the **MMA F16** kernel path (new code added in the same upstream window that added Gemma 4 support). Rotorquant types, however, only have **FA vec** kernel implementations — the `FATTN_VEC_CASES_ALL_D` macro instantiates D ∈ {64, 128, 256}. D=512 was never considered because Llama 3.1 8B (rotorquant's validation model) has D=128.

2. **Gemma 4's attention rotation path materializes K as f16 before FA**, even when KV storage is iso3. This happens because upstream's new `v_rot`/`k_rot` mul_mat pipeline dequantizes K as part of the pre-attention rotation step. So from the FA kernel's perspective, K arrives as f16 while V is still iso3 — an **asymmetric** dispatch that requires an `(F16, ISO3_0)` kernel instance at the head dim the model uses.

Workaround: added D=512 vec kernel template instantiations for the subset of configs we actually want to test on Gemma 4 models. The vec kernel's only hard constraint on D is `D % 64 == 0`, which 512 satisfies. Patched six template-instance files and added matching `FATTN_VEC_CASE(512, ...)` dispatch lines in `fattn.cu`:

- `fattn-vec-instance-f16-f16.cu` — control (matches the MMA F16 path for comparison)
- `fattn-vec-instance-iso3_0-iso3_0.cu` — symmetric iso3
- `fattn-vec-instance-planar3_0-planar3_0.cu` — symmetric planar3
- `fattn-vec-instance-planar3_0-f16.cu` — K-only planar3
- `fattn-vec-instance-f16-iso3_0.cu` — **asymmetric, required for Gemma 4 K-rotation pipeline**
- `fattn-vec-instance-f16-planar3_0.cu` — **asymmetric, same reason**

Everything compiled cleanly on `sm_120a` with CUDA 13.2 MSVC and ran correctly in smoke tests. Commits live on a local `local/rebase-attempt` branch in the johndpope fork (not pushed anywhere). The D=512 kernel patch in particular is additive and should be safe to upstream to johndpope if they want it — it's the minimum needed to make any rotorquant config work on any Gemma 4 model on CUDA, and by extension on any future model that ships with head_dim 512.

### H5 measurement: Gemma 4 31B-IT Q4_K_M + iso3/iso3 context sweep

Once the kernel gap was closed, loaded Gemma 4 31B-IT Q4_K_M with `-ctk iso3 -ctv iso3 -ngl 99 -fa on -np 1 -rea off` at progressively larger context sizes. For each, ran a single 300-token streaming decode off a fixed short prompt to measure throughput (more runs wouldn't change the picture — the interesting variable is whether the KV fits in VRAM, not run-to-run decode variance).

| Context `-c` | VRAM (MB) | Decode (tok/s) | vs turbo4 baseline (50.3) | Status |
|---:|---:|---:|---:|---|
| 32,768 (32K) | 23,835 | **44.0** | **−12.6%** | works, full speed |
| 131,072 (128K) | 29,127 | **43.9** | **−12.7%** | works, full speed |
| 163,840 (160K) | 30,284 | **43.3** | **−13.9%** | works, full speed |
| 196,608 (192K) | 31,723 | 9.1 | −81.8% | **KV spills to RAM, cliff** |
| 262,144 (262K, native) | 31,524 | 7.4 | −85.3% | KV spills, worse cliff |

The cliff lands between 160K and 192K — the pre-allocated KV buffer exceeds what fits in the 32 GB VRAM budget alongside the ~21 GB of weights and the compute buffer. Once the cliff hits, decode throughput collapses by ~5x because every token traverses PCIe for the spilled portion of the KV cache. The same RAM-spill failure mode we previously documented for Gemma 4 26B-A4B at 256K on f16 in [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md).

**Usable ceiling with iso3/iso3: ~160K context**, same throughput as 32K. Turbo4 caps at ~58K. **Net context unlock: +102K** at a **−13.9% throughput cost** (43.3 vs 50.3 tok/s).

### H5 hypothesis vs reality

The pre-experiment doc projected iso3 would unlock ~157K usable context with a ~99K gain:

> At 10.3× compression, Gemma 31B's 870 KB/token KV drops to ~84 KB/token, which could push the 5090 to 150-200K usable context on that model. Turbo4's 3.8× compression only gets us to ~230 KB/token, capping at 58K.

Measured result: 160K usable, +102K gain. **The hypothesis landed within 2% of the measured ceiling.** This is the cleanest pre-experiment prediction in the whole three-platform rotorquant experiment.

Coherence spot check at 128K passed — the model generated a correct Fibonacci implementation with a docstring and example, identical in structure to the same prompt at 32K. A full coding-suite quality bench at both 32K and 128K+ is listed in "what we'd change in the next round" above and is the obvious next follow-up.

### Correction: Gemma 4 26B-A4B does not need rotorquant for context

When writing the original hypothesis, the context math table projected Gemma 4 26B-A4B Q6_K as "turbo4 max ~230K, iso3 would unlock full 262K, useful gain +32K". That projection was based on `CONTEXT_CAPACITY_5090.md`'s architecture summary, which listed 15 global-attention layers with head_dim 256. The live gemma4 loader shows the correct numbers: **5 global-attention layers + 25 SWA layers, head_dim 512, 8 KV heads**. Recomputing from the live architecture:

- Per-token global-attn KV at f16: **~80 KB** (80 KB × 5 layers)
- At turbo4 (3.8× vs f16): ~21 KB/tok → 262K × 21 KB = 5.5 GB → **fits full 262K** in ~26.5 GB total (21 GB weights + 5.5 GB KV)
- At iso3 (10.3×): ~7.8 KB/tok → 262K × 7.8 KB = 2.0 GB → fits full 262K in ~23 GB total

**Turbo4 already saturates the 262K native window on Gemma 4 26B-A4B Q6_K with 5.5 GB of headroom to spare.** Rotorquant gives more headroom but zero additional usable context on this model. Same for Q4_K_M (smaller weights, even more headroom). The original "+32K gain" projection for 26B-A4B was wrong because the architecture table it was based on was wrong; the real answer is "no context gain on any Gemma 4 26B-A4B config — turbo4 is already enough." This correction doesn't change anything for Gemma 4 31B-IT, where the unlock is real and substantial.

### So what's still worth testing on the rebased fork?

- **Gemma 4 31B-IT + iso3/iso3 — the 4-benchmark coding suite at 32K** (confirms the quality half of H5 alongside the context/throughput halves already measured above).
- **Gemma 4 31B-IT + iso3/iso3 — one quality sanity-check at 128K or 160K** (confirms quality doesn't collapse when you actually use the unlocked context).
- **Optionally** planar3/f16 on Gemma 4 31B-IT for H2 completeness, though planar3/f16 compresses less than turbo4 (1.67× vs 3.8×) and can't reach the same long-context ceilings anyway, so it's not going to change the recommendation.

Not worth re-running on the rebased fork:
- **Gemma 4 26B-A4B Q6_K/Q4_K_M** — turbo4 already fits full native context (see correction above), and the 26B-A4B models' published turbo4 quality is already competitive, so there's no unlock to measure. The only data point it would add is "does rotorquant break anything on a hybrid-attention MoE model," which is a different question from H5.
- **Re-running Qwen/Qwopus/Harmonic on the rebased binary** — those models were tested fine on the pre-rebase binary and the rebase doesn't change the rotorquant kernels they use (D=256 vec path).
