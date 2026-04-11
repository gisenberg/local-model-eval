# RotorQuant TL;DR — M4 Max, Spark, 5090

**Date:** 2026-04-11
**Scope:** three platforms, one KV quantizer (`johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache @ 20efe75cf`), twenty-one total (model, config) pairs across three result docs.

This is a consolidated digest of [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md), [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md), and [ROTORQUANT_5090.md](ROTORQUANT_5090.md). Read those for the full per-platform story; this file is the one-screen version for deciding whether to switch away from TurboQuant for a given machine.

## The one-line verdict per platform

| Platform | Best rotorquant config | Δ vs existing default | Quality | Recommend switch? |
|---|---|---:|---|---|
| **M4 Max** (Metal) | `planar3 / f16` (K-only) | **+19% vs f16** | identical | **Yes** — first net-positive KV quantizer on Metal |
| **Spark** (aarch64 CUDA 13) | `planar3 K / f16 V` on Qwen3.5-122B | ~−1% vs f16, +5/17 quality | better in one measurement | **Yes** for Qwen3.5-122B. **No** for GLM-4.5-Air (broken output). |
| **5090** (Blackwell CUDA 13.2) | `iso3/iso3` on **Gemma 4 31B-IT** for long context; planar3/f16 on most; maybe iso3 for Harmonic | Gemma 31B: **+102K context** at −13.9% decode. Qwen/Qwopus: planar3 −4 to −6%. Harmonic iso3: +0.3% (parity) | unchanged on Qwen-family; Gemma 31B quality TBD | **Yes** for Gemma 4 31B-IT long context; **no** for Qwen/Qwopus at short context; Harmonic pending replication |

## The key finding per platform

**M4 Max.** Rotorquant is the first KV quantizer we have tested that is *faster than f16* on Metal. We had previously measured TurboQuant turbo4 as 23% *slower* than f16 on this hardware (the WHT dequant cost beats the bandwidth savings when the GPU has cheap LPDDR5X and expensive compute). Rotorquant's Givens rotation is ~64× less dequant compute per element than turbo4's WHT butterfly, and that gap was exactly what Metal needed. K-only planar3/f16 on Qwen 27B Opus-Distilled ran at 15.5 tok/s vs 13.0 tok/s f16 baseline — **+19% speedup at the same quality**. This is by far the strongest pro-rotorquant finding across all three platforms.

**Spark.** Rotorquant's throughput cost on Spark is negligible (~1% for K-only, 3-8% for symmetric on MoE) but its **quality behavior is bimodal**. On Qwen3.5-122B (DeltaNet hybrid: only 12 of 48 layers use full attention, the rest are linear attention outside the KV quantization path), planar3K/f16V produced **18/17 tests** — a +5 point jump over the f16 baseline's 13/17. On GLM-4.5-Air (standard attention in every one of 46 layers), all three rotorquant configs produced **literal `Hello?????????...` garbage** — the model stopped emitting coherent tokens or stop tokens at all. The rotorquant paper's "zero PPL loss" guarantee validated on Llama 3.1 8B did **not** transfer to GLM-4.5-Air's much deeper dense-attention stack. **Deep full-attention stacks break rotorquant**; hybrid-attention or sparse-attention architectures tolerate it fine.

**5090.** Rotorquant is mostly a throughput tax on hardware where TurboQuant is already well-tuned — *except* on Gemma 4 31B-IT, where rotorquant unlocks real additional context that turbo4 can't reach.

On Qwen 27B Opus-Distilled and Qwopus 27B (two Qwen 3.5 27B dense variants), iso3/iso3 ran ~10.5% slower than turbo4 — right in the middle of our pre-experiment prediction — and planar3/f16 ran 3.5-5.9% slower with better quality variance. Rotorquant never won on either model on pure throughput, and both models already hit the native 262K context window with turbo4 so there was no capacity to unlock. **The surprise on the Qwen family was Harmonic 27B** (Qwen 2.5 base with thinking-on mode): iso3/iso3 ran at 61.5 tok/s vs the 61.3 tok/s turbo4 baseline, **+0.3%** — the first datapoint where rotorquant's 10.3× compression cost zero throughput. Whether that's architecture-specific, thinking-mode-specific, or session-variance is still open; 3 runs aren't enough to be sure.

**On Gemma 4 31B-IT Q4_K_M** — tested after a local rebase of the johndpope fork onto current upstream llama.cpp plus a D=512 flash-attention vec kernel patch (details in [ROTORQUANT_5090.md](ROTORQUANT_5090.md)) — iso3/iso3 unlocks a measured **+102K usable context** (160K with iso3 vs 58K with turbo4) at a **−13.9% throughput cost** (43.3 vs 50.3 tok/s). The pre-experiment projection was 157K; measured reality was 160K — the hypothesis landed within 2%. For anyone running Gemma 4 31B-IT on a 5090 who needs long context, iso3/iso3 is the right default, and this is the single strongest pro-rotorquant datapoint anywhere in the three-platform experiment.

## The common blocker: Gemma 4 support in the johndpope fork

**Status as of 2026-04-11 afternoon: resolved on the 5090 via a local rebase + kernel patch, still open on M4 Max and Spark.**

The M4 Max and Spark rotorquant experiments ran against the johndpope fork at commit `20efe75cf`, which was based on upstream llama.cpp from 2026-03-25 (ggml 0.9.8). That base predates the upstream addition of `LLM_ARCH_GEMMA4`, so every Gemma 4 test case on every platform crashed with `unknown model architecture: 'gemma4'` before generating a single token. Both per-platform writeups ([ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md) and [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md)) record this as the biggest scope cut.

On the 5090, we went back and fixed it. The steps:

1. **Rebase the feature branch onto current upstream master** (241 commits ahead, includes full `LLM_ARCH_GEMMA4` plus the Gemma 4 tokenizer, template, graph, and attention-rotation machinery). The merge produced only 5 conflicted files / 8 hunks, all resolvable in one pass — rotorquant type IDs 41-47 had to be renumbered to 42-48 to make room for upstream's new `GGML_TYPE_Q1_0 = 41`, and a few attention-graph and InnerQ-stub conflicts had to be unioned.

2. **Patch a second blocker: the FA vec kernel doesn't support D=512.** Gemma 4 uses head_dim 512 for its full-attention layers (SWA layers stay at 256). Upstream f16 routes D=512 through the MMA F16 kernel; rotorquant only has vec kernel instances for D ∈ {64, 128, 256}. Loading iso3 on Gemma 4 hit `GGML_ABORT` until we added D=512 template instantiations. A subtlety: Gemma 4's pre-attention K rotation materializes K as f16 before FA, so the dispatcher actually sees `(K=F16, V=ISO3_0)` — an asymmetric combination that requires a separate kernel instance at D=512 even for "symmetric" iso3/iso3 configs.

The 5090 rebase branch is local (`local/rebase-attempt` in the johndpope fork, not pushed). The D=512 kernel patch in particular is small, additive, and should be safe to upstream if anyone wants to.

**The M4 Max and Spark experiments would still need the same rebase applied on their respective build setups** (Metal and aarch64 CUDA). The rebase steps are the same; only the kernel patch differs — M4 Max needs the D=512 Metal FA instances instead of CUDA, and Spark needs the CUDA instances for sm_121a. Neither has been done; both would unlock Gemma 4 models on those platforms and, in particular, would let us test whether the M4 Max K-only-planar3 +19% win generalizes from the Qwen 27B Opus-Distilled dense model to the Gemma 4 family.

## Hypothesis outcome summary (composite)

| Hypothesis (across docs) | M4 Max | Spark | 5090 |
|---|---|---|---|
| iso3/iso3 is 10-25% slower than the platform default | n/a (Metal) | ❌ Qwen: −3%. GLM: −25% | ✅ Qwen/Qwopus. ❌ Harmonic (+0.3%). ✅ Gemma 4 31B-IT (−13.9%) |
| planar3/f16 is the sweet spot (small throughput loss, same quality) | ✅ **+19% faster**, exceeds prediction | ✅ on Qwen. ❌ output broken on GLM | ✅ mostly (−3.5% to −5.9%, quality fine) |
| Quality is noise-dominated | ⚠️ confirmed extreme (±5 points at temp=0) | ✅ same finding (+5 on Qwen, −15 on GLM) | ✅ held on Qwen family (max 1-test delta). Gemma 31B quality-bench TBD |
| TurboQuant remains the default | ❌ turbo4 loses to planar3/f16 | n/a (we use f16, not turbo4, as Spark baseline) | ✅ for Qwen/Qwopus/Harmonic. ❌ **rotorquant wins on Gemma 4 31B-IT long context** |
| Gemma 4 31B-IT long-context unlock | ❌ blocked (gemma4 arch) | not tested | ✅ **+102K usable context** (160K vs 58K), **−13.9% throughput**, projection within 2% |

## Practical recommendations

**If you have an M4 Max:** use `planar3 / f16` K-only rotorquant as a drop-in replacement for f16 on dense GQA models (measured on Qwen 3.5 27B Opus-Distilled). You'll get +19% decode throughput at the same quality and the same context limits. For MoE sliding-window models (Gemma 4 family), this fork can't load them at all — keep the turboquant fork for those.

**If you have a DGX Spark:** use `planar3 K / f16 V` on Qwen3.5-122B (the DeltaNet hybrid) — it ties the S-tier ik-llama result with a simpler engine stack. **Do not** use rotorquant on GLM-4.5-Air or any deep standard-attention MoE until upstream fixes the output collapse. The "zero PPL loss" claim from the paper demonstrably does not hold on deep dense-attention stacks, which is an important caveat for anyone picking rotorquant for a new model class.

**If you have a 5090:**
- **For Gemma 4 31B-IT Q4_K_M with long-context needs (>58K)**: use `iso3/iso3` on the locally-rebased johndpope fork (see the common-blocker section above for the rebase + D=512 kernel patch details). Measured usable ceiling is ~160K context at 43.3 tok/s, a +102K gain over turbo4's 58K ceiling at a 14% throughput cost. This is the one clear rotorquant win on the 5090.
- **For Gemma 4 31B-IT at short context (≤58K)**: turbo4 still wins on throughput (50.3 vs 43.3 tok/s). Only switch to iso3 when you actually need the extra context.
- **For Qwen 27B Opus-Distilled, Qwopus 27B, Harmonic 27B**: keep TurboQuant turbo4. Rotorquant is a ~10% throughput tax on Qwen/Qwopus for zero context gain (both already hit 262K native with turbo4), and Harmonic's apparent +0.3% iso3 win is too small to act on without replication.
- **For Gemma 4 26B-A4B Q6_K/Q4_K_M**: keep turbo4 — the live architecture has only 5 global-attention layers, and turbo4 already fits the full 262K native window with 5+ GB of headroom. Rotorquant has no context to unlock here (the pre-experiment projection of "+32K with iso3" was based on a wrong architecture table — see the correction in [ROTORQUANT_5090.md](ROTORQUANT_5090.md)).

## Open questions / follow-ups

1. **Rebase the planarquant fork onto gemma4-capable llama.cpp — done on 5090, still open on M4 Max and Spark.** This was the single highest-value action when the three experiments were first run, and doing it on the 5090 cleanly unlocked the H5 long-context win on Gemma 4 31B-IT (+102K usable context, within 2% of the hypothesis projection). The M4 Max and Spark versions of the same rebase have not been done; they'd unlock the same class of test on those platforms. The 5090 rebase steps — 3-way merge of upstream/master into the feature branch (5 conflicts), plus D=512 FA vec kernel instances for the Gemma 4 head dimension — should be transplantable to Metal and aarch64 CUDA with backend-specific kernel changes.

1b. **A coding-suite quality bench on Gemma 4 31B-IT + iso3/iso3** — the 5090 H5 measurement confirmed the context-unlock and throughput story but only did a single-prompt smoke test for quality. The full 4-benchmark 22-test suite at 32K and one sanity-check at the unlocked 128K-160K range would finish the H5 picture.

2. **Why does Harmonic 27B escape the throughput tax on the 5090?** A 5-run replication of Harmonic × iso3/iso3 vs Harmonic × turbo4 with matched server restarts would resolve whether it's a real architecture/thinking-mode effect or session variance. If real, the mechanism (possibly Harmonic's attention head dimensions aligning with iso3's 4D quaternion blocks) would be worth a follow-up investigation — and potentially a published counter-point to the "rotorquant is always a tax on CUDA" prior we started with.

3. **Why does GLM-4.5-Air break catastrophically under any rotorquant config?** The failure mode (literal `?` spam, no stop tokens) is much more severe than the paper's 4.2% PPL degradation on Llama 3.1 8B. Two plausible causes:
   - Error accumulation across 46 dense-attention layers (vs Llama 3.1 8B's 32), if the per-layer rotation error is larger than the 8B paper suggested.
   - A Metal/CUDA kernel bug in the rotation step specific to GLM's head dimensions or group size.

   A single PPL measurement on wikitext-2 for GLM + planar3K would separate the "accumulation" hypothesis from the "kernel bug" hypothesis. This is worth filing upstream.

4. **Rotorquant + thinking-mode models.** Harmonic 27B is the only thinking-mode model we tested anywhere in this experiment. Its anomalous 5090 result (iso3/iso3 beats turbo4) plus the specific failure mode on GLM (stop-token recognition breaks) both suggest that rotorquant's quality and throughput behavior could differ for thinking-mode workloads in ways the paper's single-dense-8B baseline cannot predict. A broader thinking-mode evaluation (Harmonic, GPT-OSS 20B, Qwen 32B thinking) would be valuable before adopting rotorquant as a default for any thinking-heavy pipeline.

5. **The noise band is bigger than we thought.** Spark's experiment exposed a ±5 point noise band at temp=0 single-shot, which retroactively weakens any claim based on a single 17-test single-shot run — including parts of MODEL_RANKINGS_SPARK.md. Future KV quantizer evaluations should default to best-of-3 at temp 0.3 or explicitly report multi-seed variance.

## See also

- [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md) — full M4 Max results (the "+19% on Metal" story)
- [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md) — full Spark results (the "Qwen win, GLM broken" story)
- [ROTORQUANT_5090.md](ROTORQUANT_5090.md) — full 5090 results (the "turbo4 mostly wins, Harmonic ties" story)
- [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md) — pre-experiment M4 Max predictions
- [../ROTORQUANT_HYPOTHESIS_SPARK.md](../ROTORQUANT_HYPOTHESIS_SPARK.md) — pre-experiment Spark predictions
- [../ROTORQUANT_HYPOTHESIS_5090.md](../ROTORQUANT_HYPOTHESIS_5090.md) — pre-experiment 5090 predictions
