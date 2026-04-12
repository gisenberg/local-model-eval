# RotorQuant TL;DR — M4 Max, Spark, 5090

**Date:** 2026-04-11
**Scope:** three platforms, one KV quantizer (`johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache @ 20efe75cf`), twenty-one total (model, config) pairs across three result docs.

This is a consolidated digest of [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md), [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md), and [ROTORQUANT_5090.md](ROTORQUANT_5090.md). Read those for the full per-platform story; this file is the one-screen version for deciding whether to switch away from TurboQuant for a given machine.

## The one-line verdict per platform

| Platform | Best rotorquant config | Δ vs existing default | Quality | Recommend switch? |
|---|---|---:|---|---|
| **M4 Max** (Metal) | `planar3 / f16` (K-only) — three-for-three wins | **+13% to +20%** across Qwen 27B, Gemma 26B-A4B, Gemma 31B | identical across all three models | **Yes** — first net-positive KV quantizer on Metal, measured on dense GQA, MoE sliding-window, and dense full-attention |
| **Spark** (aarch64 CUDA 13) | `planar3 K / f16 V` on Qwen3.5-122B | ~−1% vs f16, +5/17 quality | better in one measurement | **Yes** for Qwen3.5-122B. **No** for GLM-4.5-Air (broken output). |
| **5090** (Blackwell CUDA 13.2) | `iso3/iso3` on **Gemma 4 31B-IT** for long context; planar3/f16 on most; maybe iso3 for Harmonic | Gemma 31B: **+102K context** at −21% decode (coding workload). Qwen/Qwopus: planar3 −4 to −6%. Harmonic iso3: +0.3% (parity) | Gemma 31B: **22/22** best-of-3 at both 32K and 128K. Qwen/Qwopus/Harmonic unchanged. | **Yes** for Gemma 4 31B-IT long context; **no** for Qwen/Qwopus at short context; Harmonic pending replication |

## The key finding per platform

**M4 Max.** Rotorquant is the first KV quantizer we have tested that is *faster than f16* on Metal. We had previously measured TurboQuant turbo4 as 23% *slower* than f16 on this hardware (the WHT dequant cost beats the bandwidth savings when the GPU has cheap LPDDR5X and expensive compute). Rotorquant's Givens rotation is ~64× less dequant compute per element than turbo4's WHT butterfly, and that gap was exactly what Metal needed. Measured K-only `planar3 / f16` results across all three test models (second pass after a gemma4 cherry-pick unblocked the Gemma tests):

- **Qwen 3.5 27B Opus-Distilled Q4_K_M** (dense GQA): 15.5 tok/s vs 13.0 f16 baseline = **+19%**, 11/17 → 11/17 (identical)
- **Gemma 4 26B-A4B Q6_K** (MoE sliding window): 68.2 tok/s vs 60.3 f16 baseline = **+13%**, 15/17 → 15/17 (identical)
- **Gemma 4 31B-IT Q4_K_M** (dense full attention): 14.2 tok/s vs 11.8 turbo4 baseline = **+20%**, 17/17 → 17/17 (identical)

Three-for-three, spanning three distinct attention patterns. This is by far the strongest pro-rotorquant *throughput* finding across all three platforms. Symmetric `planar3/planar3` on Gemma 4 hits a runaway-generation bug (model emits 16K tokens without reaching a stop token) that looks like the Metal V-dequant inverse-rotation TODO from rotorquant's CLAUDE.md — K-only dodges it by leaving V at f16.

**M4 Max context: rotorquant gives ZERO gain, the base upgrade gives a big one.** This was a subtlety the first writeup missed. Rotorquant's K-only mode uses **deferred quantization**: K is allocated at f16 size at load time, then converted to planar3 at insertion during decode. So it saves bandwidth during decode (the +13-20% speedups above) but not memory at allocation. Gemma 4 31B-IT at 128K with planar3/f16 K-only allocates **more** memory than f16/f16 at the same context, not less. The apparent "Gemma 31B now fits at 32K with default `-ub`" win came entirely from the llama.cpp base upgrade bundled into the Gemma 4 cherry-picks, which dropped the Metal compute buffer from 16.7 GB to 523 MiB at 32K for sliding-window models — a 32× reduction. If you want actual context-capacity gains on Gemma 4 31B-IT, use **turbo4 KV** (which compresses at allocation time), not rotorquant K-only. Details in [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md).

**Spark.** Rotorquant's throughput cost on Spark is negligible (~1% for K-only, 3-8% for symmetric on MoE) but its **quality behavior is bimodal**. On Qwen3.5-122B (DeltaNet hybrid: only 12 of 48 layers use full attention, the rest are linear attention outside the KV quantization path), planar3K/f16V produced **18/17 tests** — a +5 point jump over the f16 baseline's 13/17. On GLM-4.5-Air (standard attention in every one of 46 layers), all three rotorquant configs produced **literal `Hello?????????...` garbage** — the model stopped emitting coherent tokens or stop tokens at all. The rotorquant paper's "zero PPL loss" guarantee validated on Llama 3.1 8B did **not** transfer to GLM-4.5-Air's much deeper dense-attention stack. **Deep full-attention stacks break rotorquant**; hybrid-attention or sparse-attention architectures tolerate it fine.

**5090.** Rotorquant is mostly a throughput tax on hardware where TurboQuant is already well-tuned — *except* on Gemma 4 31B-IT, where rotorquant unlocks real additional context that turbo4 can't reach.

On Qwen 27B Opus-Distilled and Qwopus 27B (two Qwen 3.5 27B dense variants), iso3/iso3 ran ~10.5% slower than turbo4 — right in the middle of our pre-experiment prediction — and planar3/f16 ran 3.5-5.9% slower with better quality variance. Rotorquant never won on either model on pure throughput, and both models already hit the native 262K context window with turbo4 so there was no capacity to unlock. **The surprise on the Qwen family was Harmonic 27B** (Qwen 2.5 base with thinking-on mode): iso3/iso3 ran at 61.5 tok/s vs the 61.3 tok/s turbo4 baseline, **+0.3%** — the first datapoint where rotorquant's 10.3× compression cost zero throughput. Whether that's architecture-specific, thinking-mode-specific, or session-variance is still open; 3 runs aren't enough to be sure.

**On Gemma 4 31B-IT Q4_K_M** — tested after a local rebase of the johndpope fork onto current upstream llama.cpp plus a D=512 flash-attention vec kernel patch (details in [ROTORQUANT_5090.md](ROTORQUANT_5090.md)) — iso3/iso3 unlocks a measured **+102K usable context** (160K with iso3 vs 58K with turbo4). The coding-suite bench at 32K scored **22/22 best-of-3** across the 4-benchmark suite at **39.7 tok/s**, a **−21.1%** decode delta vs turbo4's 50.3 tok/s baseline. A sanity check at 128K on Expression Evaluator produced the identical **22/22 best, 4.7/5 avg** at **39.6 tok/s** — quality does not degrade when you actually use the unlocked long-context band. The pre-experiment projection of 157K usable landed within 2% of measured reality. For anyone running Gemma 4 31B-IT on a 5090 who needs long context, iso3/iso3 is the right default, and this is the single strongest pro-rotorquant datapoint anywhere in the three-platform experiment.

## The common blocker: Gemma 4 support in the johndpope fork

**Status as of 2026-04-11 evening: resolved on 5090 and M4 Max via local cherry-picks, still open on Spark.**

The M4 Max and Spark rotorquant experiments ran against the johndpope fork at commit `20efe75cf`, which was based on upstream llama.cpp from 2026-03-25 (ggml 0.9.8). That base predates the upstream addition of `LLM_ARCH_GEMMA4`, so every Gemma 4 test case on every platform crashed with `unknown model architecture: 'gemma4'` before generating a single token. Both per-platform writeups ([ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md) and [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md)) record this as the biggest scope cut.

On the 5090, we went back and fixed it. The steps:

1. **Rebase the feature branch onto current upstream master** (241 commits ahead, includes full `LLM_ARCH_GEMMA4` plus the Gemma 4 tokenizer, template, graph, and attention-rotation machinery). The merge produced only 5 conflicted files / 8 hunks, all resolvable in one pass — rotorquant type IDs 41-47 had to be renumbered to 42-48 to make room for upstream's new `GGML_TYPE_Q1_0 = 41`, and a few attention-graph and InnerQ-stub conflicts had to be unioned.

2. **Patch a second blocker: the FA vec kernel doesn't support D=512.** Gemma 4 uses head_dim 512 for its full-attention layers (SWA layers stay at 256). Upstream f16 routes D=512 through the MMA F16 kernel; rotorquant only has vec kernel instances for D ∈ {64, 128, 256}. Loading iso3 on Gemma 4 hit `GGML_ABORT` until we added D=512 template instantiations. A subtlety: Gemma 4's pre-attention K rotation materializes K as f16 before FA, so the dispatcher actually sees `(K=F16, V=ISO3_0)` — an asymmetric combination that requires a separate kernel instance at D=512 even for "symmetric" iso3/iso3 configs.

The 5090 rebase branch is local (`local/rebase-attempt` in the johndpope fork, not pushed). The D=512 kernel patch in particular is small, additive, and should be safe to upstream if anyone wants to.

**On M4 Max**, we took a simpler path than the 5090's full rebase: cherry-pick just the two commits that matter for Gemma 4 text support:

1. **`63f8fe0ef`** (upstream PR #21309) — adds `LLM_ARCH_GEMMA4`, `src/models/gemma4-iswa.cpp`, and the rest of the Gemma 4 text-model infrastructure. One conflict in `tools/mtmd/mtmd.cpp` (multimodal image code, unrelated to text benchmarks) — reverted to planarquant's version. Also reverted the other `tools/mtmd/*` bits from the same commit because they reference `mtmd-image.h` which doesn't exist on the planarquant base.
2. **`5208e2d5b`** (upstream PR #21326) — Gemma 4 chat-template fix. Needed because without it, `--reasoning-budget 0` leaks ~43 KB of reasoning content into output. Cherry-picks cleanly, no conflicts.

No kernel patch was needed on M4 Max — the Metal FA kernels already handle the head dimensions Gemma 4 uses. The rebuilt binary at `~/git/TheTom/llama-cpp-planarquant` (branch `local/planarquant-gemma4-cherrypick`) loads Gemma 4 models and all four rotorquant cache types (`planar3`, `iso3`, `planar4`, `iso4`) still register. **Gemma 4 K-only results on M4 Max are included in the table above and the per-platform deep-dive section.**

**Spark is still open.** Same two cherry-picks should in principle work there too, plus possibly a D=512 CUDA kernel patch similar to the 5090's if iso3/iso3 is used for symmetric configs. Not attempted in this session.

## Hypothesis outcome summary (composite)

| Hypothesis (across docs) | M4 Max | Spark | 5090 |
|---|---|---|---|
| iso3/iso3 is 10-25% slower than the platform default | ❌ Gemma 4: runaway-generation bug (Metal V-dequant TODO). ⚠️ Qwen 27B Opus: symmetric planar3 −15% vs f16 | ❌ Qwen: −3%. GLM: −25% | ✅ Qwen/Qwopus (~−10%). ❌ Harmonic (+0.3%). ✅ Gemma 4 31B-IT (−21.1% on coding workload) |
| planar3/f16 is the sweet spot (small throughput loss, same quality) | ✅✅ **three-for-three +13% to +20% FASTER**, not "small loss" — exceeds prediction dramatically | ✅ on Qwen. ❌ output broken on GLM | ✅ mostly (−3.5% to −5.9%, quality fine) |
| Quality is noise-dominated | ✅ K-only is clean (no noise on any model). ⚠️ symmetric on Qwen 27B did show the path-flip effect (+5 points) | ✅ same finding (+5 on Qwen, −15 on GLM) | ✅ held on Qwen family (max 1-test delta). ✅ held on Gemma 31B: 22/22 best at both 32K and 128K |
| TurboQuant remains the default | ❌ planar3/f16 beats turbo4 on Gemma 4 31B-IT by +20%, and beats f16 on every other model | n/a (we use f16, not turbo4, as Spark baseline) | ✅ for Qwen/Qwopus/Harmonic. ❌ **rotorquant wins on Gemma 4 31B-IT long context** |
| Gemma 4 31B-IT throughput unlock | ✅ **+20% vs turbo4 at 17/17** with K-only planar3/f16 at 32K (after cherry-pick) | not tested | ✅ separately, via context unlock: iso3/iso3 gets +102K context at −21% decode |
| Gemma 4 31B-IT long-context unlock | ⚠️ blocked on symmetric V-dequant bug — K-only can't extend context because V stays at f16 | not tested | ✅ **+102K usable context** (160K vs 58K), **22/22 quality at 32K and 128K**, −21.1% throughput on coding workload, capacity projection within 2% |

## Practical recommendations

**If you have an M4 Max:** use `-ctk planar3 -ctv f16` as a drop-in replacement for f16 or turbo4 on any of the models we benchmarked. Measured on dense GQA (Qwen 3.5 27B Opus-Distilled: +19%), MoE sliding window (Gemma 4 26B-A4B Q6_K: +13%), and dense full attention (Gemma 4 31B-IT Q4_K_M: +20% vs turbo4). Quality is identical to the best prior config in all three cases. Apply the two cherry-picks from the "common blocker" section above if you want to use it on Gemma 4 models — the stock planarquant fork base doesn't have gemma4 text support. Symmetric `planar3/planar3` on Gemma 4 has a runaway-generation bug — stick to K-only until the Metal V-dequant fix lands upstream.

**If you have a DGX Spark:** use `planar3 K / f16 V` on Qwen3.5-122B (the DeltaNet hybrid) — it ties the S-tier ik-llama result with a simpler engine stack. **Do not** use rotorquant on GLM-4.5-Air or any deep standard-attention MoE until upstream fixes the output collapse. The "zero PPL loss" claim from the paper demonstrably does not hold on deep dense-attention stacks, which is an important caveat for anyone picking rotorquant for a new model class.

**If you have a 5090:**
- **For Gemma 4 31B-IT Q4_K_M with long-context needs (>58K)**: use `iso3/iso3` on the locally-rebased johndpope fork (see the common-blocker section above for the rebase + D=512 kernel patch details). Measured usable ceiling is ~160K context at 39.7 tok/s on the coding suite, a +102K gain over turbo4's 58K ceiling at a 21% throughput cost. Quality holds at 22/22 best-of-3 on the full 4-benchmark suite, and the same quality signal replicates on a 128K Expression Evaluator sanity check. This is the one clear rotorquant win on the 5090.
- **For Gemma 4 31B-IT at short context (≤58K)**: turbo4 still wins on throughput (50.3 vs 39.7 tok/s). Only switch to iso3 when you actually need the extra context.
- **For Qwen 27B Opus-Distilled, Qwopus 27B, Harmonic 27B**: keep TurboQuant turbo4. Rotorquant is a ~10% throughput tax on Qwen/Qwopus for zero context gain (both already hit 262K native with turbo4), and Harmonic's apparent +0.3% iso3 win is too small to act on without replication.
- **For Gemma 4 26B-A4B Q6_K/Q4_K_M**: keep turbo4 — the live architecture has only 5 global-attention layers, and turbo4 already fits the full 262K native window with 5+ GB of headroom. Rotorquant has no context to unlock here (the pre-experiment projection of "+32K with iso3" was based on a wrong architecture table — see the correction in [ROTORQUANT_5090.md](ROTORQUANT_5090.md)).

## Open questions / follow-ups

1. **Rebase the planarquant fork onto gemma4-capable llama.cpp — done on 5090, still open on M4 Max and Spark.** This was the single highest-value action when the three experiments were first run, and doing it on the 5090 cleanly unlocked the H5 long-context win on Gemma 4 31B-IT (+102K usable context, within 2% of the hypothesis projection). The M4 Max and Spark versions of the same rebase have not been done; they'd unlock the same class of test on those platforms. The 5090 rebase steps — 3-way merge of upstream/master into the feature branch (5 conflicts), plus D=512 FA vec kernel instances for the Gemma 4 head dimension — should be transplantable to Metal and aarch64 CUDA with backend-specific kernel changes.

1b. ~~**A coding-suite quality bench on Gemma 4 31B-IT + iso3/iso3**~~ — **done.** Full 4-benchmark suite at 32K scored 22/22 best-of-3 at 39.7 tok/s; Expression Evaluator sanity check at 128K scored 5/5 best at 39.6 tok/s (same quality as 32K). The H5 picture is complete on the 5090.

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
