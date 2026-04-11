# RotorQuant TL;DR — M4 Max, Spark, 5090

**Date:** 2026-04-11
**Scope:** three platforms, one KV quantizer (`johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache @ 20efe75cf`), twenty-one total (model, config) pairs across three result docs.

This is a consolidated digest of [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md), [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md), and [ROTORQUANT_5090.md](ROTORQUANT_5090.md). Read those for the full per-platform story; this file is the one-screen version for deciding whether to switch away from TurboQuant for a given machine.

## The one-line verdict per platform

| Platform | Best rotorquant config | Δ vs existing default | Quality | Recommend switch? |
|---|---|---:|---|---|
| **M4 Max** (Metal) | `planar3 / f16` (K-only) | **+19% vs f16** | identical | **Yes** — first net-positive KV quantizer on Metal |
| **Spark** (aarch64 CUDA 13) | `planar3 K / f16 V` on Qwen3.5-122B | ~−1% vs f16, +5/17 quality | better in one measurement | **Yes** for Qwen3.5-122B. **No** for GLM-4.5-Air (broken output). |
| **5090** (Blackwell CUDA 13.2) | `planar3 / f16` for most, maybe `iso3/iso3` for Harmonic | −3.5 to −5.9% vs turbo4 (planar3), +0.3% for Harmonic iso3 | unchanged | **No** for 5 of 6 planned models; maybe for Harmonic pending replication. |

## The key finding per platform

**M4 Max.** Rotorquant is the first KV quantizer we have tested that is *faster than f16* on Metal. We had previously measured TurboQuant turbo4 as 23% *slower* than f16 on this hardware (the WHT dequant cost beats the bandwidth savings when the GPU has cheap LPDDR5X and expensive compute). Rotorquant's Givens rotation is ~64× less dequant compute per element than turbo4's WHT butterfly, and that gap was exactly what Metal needed. K-only planar3/f16 on Qwen 27B Opus-Distilled ran at 15.5 tok/s vs 13.0 tok/s f16 baseline — **+19% speedup at the same quality**. This is by far the strongest pro-rotorquant finding across all three platforms.

**Spark.** Rotorquant's throughput cost on Spark is negligible (~1% for K-only, 3-8% for symmetric on MoE) but its **quality behavior is bimodal**. On Qwen3.5-122B (DeltaNet hybrid: only 12 of 48 layers use full attention, the rest are linear attention outside the KV quantization path), planar3K/f16V produced **18/17 tests** — a +5 point jump over the f16 baseline's 13/17. On GLM-4.5-Air (standard attention in every one of 46 layers), all three rotorquant configs produced **literal `Hello?????????...` garbage** — the model stopped emitting coherent tokens or stop tokens at all. The rotorquant paper's "zero PPL loss" guarantee validated on Llama 3.1 8B did **not** transfer to GLM-4.5-Air's much deeper dense-attention stack. **Deep full-attention stacks break rotorquant**; hybrid-attention or sparse-attention architectures tolerate it fine.

**5090.** Rotorquant is mostly a throughput tax on hardware where TurboQuant is already well-tuned. On Qwen 27B Opus-Distilled and Qwopus 27B (two Qwen 3.5 27B dense variants), iso3/iso3 ran ~10.5% slower than turbo4 — right in the middle of our pre-experiment prediction — and planar3/f16 ran 3.5-5.9% slower with better quality variance. Rotorquant never won on either model. **The surprise was Harmonic 27B** (Qwen 2.5 base with thinking-on mode): iso3/iso3 ran at 61.5 tok/s vs the 61.3 tok/s turbo4 baseline, **+0.3%** — the first (and only) 5090 datapoint where rotorquant's 10.3× compression cost zero throughput. Whether that's architecture-specific, thinking-mode-specific, or session-variance is still open; 3 runs aren't enough to be sure.

## The common blocker: Gemma 4 unsupported everywhere

Across all three platforms, the highest-value rotorquant hypothesis was the same: use iso3/iso3's 10.3× compression to extend Gemma 4 31B-IT's dense-KV context far beyond what turbo4 allows. On the 5090, we projected turbo4's 58K ceiling → iso3's 157K (+99K gain). On the M4 Max, we projected similar numbers at smaller absolute context. **None of this was testable.** The johndpope fork is based on an older llama.cpp commit (ggml 0.9.8) that does not know the `gemma4` architecture. Every Gemma 4 test case — the most interesting one on every platform — crashed with `unknown model architecture: 'gemma4'` before generating a single token.

This is the single most important action item across all three experiments: **a rebase of the planarquant fork onto a gemma4-capable llama.cpp base** would unblock the only experiment configuration where rotorquant's compression ratio has a clear, measurable win on our model lineup. Without it, rotorquant's 10.3× compression story has nowhere to land.

## Hypothesis outcome summary (composite)

| Hypothesis (across docs) | M4 Max | Spark | 5090 |
|---|---|---|---|
| iso3/iso3 is 10-25% slower than the platform default | n/a (Metal) | ❌ Qwen: −3%. GLM: −25% | ✅ Qwen/Qwopus. ❌ Harmonic (+0.3%) |
| planar3/f16 is the sweet spot (small throughput loss, same quality) | ✅ **+19% faster**, exceeds prediction | ✅ on Qwen. ❌ output broken on GLM | ✅ mostly (−3.5% to −5.9%, quality fine) |
| Quality is noise-dominated | ⚠️ confirmed extreme (±5 points at temp=0) | ✅ same finding (+5 on Qwen, −15 on GLM) | ✅ held (max 1-test delta on 22-test suites) |
| TurboQuant remains the default | ❌ turbo4 loses to planar3/f16 | n/a (we use f16, not turbo4, as Spark baseline) | ✅ for 5 of 6 planned models |
| Gemma 4 31B-IT long-context unlock | ❌ blocked (gemma4 arch) | not tested | ❌ blocked (gemma4 arch) |

## Practical recommendations

**If you have an M4 Max:** use `planar3 / f16` K-only rotorquant as a drop-in replacement for f16 on dense GQA models (measured on Qwen 3.5 27B Opus-Distilled). You'll get +19% decode throughput at the same quality and the same context limits. For MoE sliding-window models (Gemma 4 family), this fork can't load them at all — keep the turboquant fork for those.

**If you have a DGX Spark:** use `planar3 K / f16 V` on Qwen3.5-122B (the DeltaNet hybrid) — it ties the S-tier ik-llama result with a simpler engine stack. **Do not** use rotorquant on GLM-4.5-Air or any deep standard-attention MoE until upstream fixes the output collapse. The "zero PPL loss" claim from the paper demonstrably does not hold on deep dense-attention stacks, which is an important caveat for anyone picking rotorquant for a new model class.

**If you have a 5090:** keep TurboQuant turbo4 as the default. Rotorquant loses to turbo4 on every model we tested except Harmonic 27B, where it ties. The one experiment where rotorquant would have genuinely helped (Gemma 31B-IT context unlock) was blocked by the same gemma4-architecture gap that broke the M4 Max experiment. If you need >58K context on a dense model and can use a Qwen family instead, Qwen 27B Opus-Distilled already runs at 262K on turbo4 for less quality than Gemma 31B-IT.

## Open questions / follow-ups

1. **Rebase the planarquant fork onto gemma4-capable llama.cpp.** This is the single highest-value action for anyone invested in rotorquant on our model lineup. Unblocks Gemma 4 26B-A4B and Gemma 4 31B-IT on all three platforms and makes the H5 long-context story testable.

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
