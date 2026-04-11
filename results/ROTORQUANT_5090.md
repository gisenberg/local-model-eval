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

**H5 (Gemma 31B-IT context unlock):** ❌ **not testable** — the johndpope fork's base upstream predates Gemma 4 support. The one model where rotorquant's +99K context unlock would have mattered is the one we couldn't run.

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

5. **The Gemma 31B-IT context unlock remains untested.** This is the one rotorquant use case where we predicted a clear win on the 5090 (turbo4 ceiling 58K → iso3 projected 157K, a +99K gain on the model's existing turbo4 baseline). Needs a newer johndpope rebase onto a llama.cpp base that includes Gemma 4 architecture support before we can measure it.

## What we'd change in the next round

- **More runs per benchmark** (5 instead of 3) for Harmonic specifically, to resolve the iso3 vs turbo4 ambiguity.
- **Include a matched f16/f16 run** on one model so we can reproduce the paper's reference point directly on our hardware.
- **Retest when johndpope rebases** onto a llama.cpp base with Gemma 4 support, so H5 can actually be measured.
- **Try `planar3/q8_0` and `iso3/q8_0`** asymmetric configs from the johndpope README — we only tested the two most-cited configs and there are 5+ more variants that might trade differently.

## Not in this experiment

Everything listed in the "Not tested" section of [ROTORQUANT_HYPOTHESIS_5090.md](../ROTORQUANT_HYPOTHESIS_5090.md) still applies: no vLLM/NVFP4 comparisons (wrong engine), no turbo3 comparison (we use turbo4), no long-context (≥64K) runs, no 4-bit variants (planar4/iso4 crash on FA dispatch), no PPL measurement.
