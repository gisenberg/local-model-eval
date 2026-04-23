# TurboQuant KV cache compression — cross-platform

TurboQuant is a KV cache compression fork of llama.cpp (`feature/turboquant-kv-cache` branch). It offers `turbo2`/`turbo3`/`turbo4` quant types for the K and V cache tensors, targeting 3.8× compression vs f16 at +0.23% perplexity (turbo4) up to 5× at +1.06% (turbo3).

**The core cross-platform finding**: TurboQuant's value depends on what's bottlenecking the working set on your hardware. **On CUDA (RTX 5090) it's usually a free context-unlocker; on Metal (M4 Max) it's a pure speed loss except for one dense model where it's the only thing that makes the model load at all.**

| | RTX 5090 (32 GB, 1792 GB/s) | M4 Max (36 GB, 410 GB/s) |
|---|---|---|
| Default recommendation | **Use `-ctk turbo4 -ctv turbo4 -rea off`** | **Use `-ctk f16 -ctv f16`; first lever for OOMs is `-ub 256`, not turbo4** |
| Typical quality delta | 0 to +1 test (turbo4 occasionally helps reasoning models) | ~0 (within single-shot noise) |
| Typical speed delta | −5 to −10% at 32K context; faster than f16 at 128K+ (f16 spills to RAM) | **−23% to −31% at matched context** |
| "Unlocks" | Gemma 4 31B dense (16K → 58K ctx), Gemma 4 26B-A4B (85K → 230K full) | Only: Gemma 4 31B dense (would OOM at f16 even at 16K on old base; useful optional context-extender on new base) |
| Why the asymmetry | 5090 is bandwidth-bound; KV reads dominate per-token budget | M4 Max is weights-bound; KV is <1% of working set. Compute buffer is the real memory sink. |

For RotorQuant (the newer K-only Givens-rotation variant), see [ROTORQUANT.md](ROTORQUANT.md).

## Mechanism asymmetry

Two platform differences combine to invert TurboQuant's value proposition on Apple Silicon.

**Bandwidth bottleneck (speed).** The 5090 runs at 1,792 GB/s with 32 GB VRAM — KV cache reads are a meaningful fraction of per-token bandwidth for dense/sliding-window models. Compressing the KV saves bandwidth → faster decode. The M4 Max runs at 410 GB/s with ~30 GB usable. Weights bandwidth (16-22 GB read per token) already dominates the decode budget; compressing the KV cache doesn't free enough bandwidth to matter, and the dequantization compute overhead actively hurts.

**Compute buffer dominates the working set (capacity).** On Metal the *compute buffer* is the dominant working-set cost for context, not the KV cache. From an actual Gemma 4 26B-A4B Q6_K turbo4 run at 32K on M4 Max:

```
weights:           21,574 MiB
KV cache (turbo4):    250 MiB    ← 0.8% of the budget
compute buffer:     8,402 MiB    ← 28% of the budget
total:             30,244 MiB    → exceeds 28,753 MiB free, OOMs
```

The KV cache is less than 1% of the working set. Compressing it further saves almost nothing — there's nothing left to compress. The 5090 uses 25,636 MiB total VRAM for the same model at 32K; Metal's compute buffer is roughly 4-5 GB bigger than CUDA's for this arch. That's why the 5090 hits 230K context on this model and the M4 Max can't even hit 24K — nothing to do with the KV cache.

**The result**: turbo4 helps the M4 Max in exactly one scenario — the small fraction of cases where the KV cache itself is a meaningful share of memory (dense full-attention models like Gemma 4 31B-IT). Everywhere else it's a pure speed loss.

---

# RTX 5090 (32 GB)

Tested on RTX 5090 32 GB, TurboQuant fork of llama.cpp (`feature/turboquant-kv-cache`), CUDA 13.2, `-fa on`, temperature 0, `max_tokens=16384`.

## The three categories

### Must-have: models that don't work without TurboQuant

**Gemma 4 31B-IT Q4_K_M** is the extreme case. Dense architecture burns ~870 KB per token on KV cache (7× more than the MoE variant). With f16 KV, after 18 GB of weights, only ~12 GB remains — that's ~16K tokens of context. Turbo4 stretches the same 12 GB budget to ~58K tokens, and the model scores **17/17 (100%)** on our coding benchmarks. Without TurboQuant it's a paperweight on 32 GB; with it, it's S-tier.

### Game-changer: from limited to full context

**Gemma 4 26B-A4B** (sliding-window + global attention, ~120 KB/token) maxes at ~85K context with f16 KV and drops 150 → 55 tok/s at 256K due to RAM spill. With turbo4:

- Q6_K reaches ~230K context entirely in VRAM, no spill
- Q4_K_M reaches the full 262K native context window
- Both score 16-17/17

**Qwopus 27B Q6_K** goes from ~160K to full 262K. Not dramatic, but the difference between "most of the context window" and "all of it" matters for long-document tasks.

### Minimal impact: models that already fit

**Qwen 3.5 35B-A3B** uses a DeltaNet hybrid with only 10 attention layers and ~20 KB/token KV. Full 262K fits in 32 GB even with f16. TurboQuant saves negligible VRAM here (it also scores poorly on llama-server for unrelated reasons — see [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md)).

### Context-unlock summary

| Model | Weights | KV/token (f16) | Max ctx f16 | Max ctx turbo4 | Gained | Score |
|---|---|---|---|---|---|---|
| Gemma 4 31B-IT Q4_K_M | 18 GB | ~870 KB | ~16K | ~58K | +42K | 17/17 S |
| Gemma 4 26B-A4B Q6_K | 22 GB | ~120 KB | ~85K | ~230K | +145K | 17/17 S |
| Gemma 4 26B-A4B Q4_K_M | 16 GB | ~120 KB | ~136K | 262K full | +126K | 16/17 A |
| Qwopus 27B Q6_K | 22 GB | ~64 KB | ~160K | 262K full | +102K | 16/17 A |
| Qwen3.5-27B-opus Q4_K_M | 16 GB | ~64 KB | ~256K | 262K full | ~6K | 17/17 S |
| Qwen3.5-35B-A3B Q4_K_M | 20 GB | ~20 KB | 262K | 262K full | 0 | 11/17 C |

## Optimal config: turbo4 + thinking off

Three models hit 100% (17/17) — including Gemma 31B which previously maxed out at 10/17.

| Model | Quant | KV | Thinking | VRAM (32K) | Tok/s | Total | Notes |
|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | Q6_K | turbo4/turbo4 | off | 25,636 MB | 142 | **17/17** | Fastest S-tier at 121-124 tok/s generation |
| Gemma 4 31B-IT | Q4_K_M | turbo4/turbo4 | off | 22,293 MB | 53 | **17/17** | Transformed from D-tier |
| Gemma 4 26B-A4B | Q4_K_M | turbo4/turbo4 | off | 20,046 MB | 156 | 16/17 | Lowest VRAM. One LRU edge case |
| Qwopus 27B | Q6_K | turbo4/turbo4 | off | 24,549 MB | 52 | 16/17 | Reliable. Embeds reasoning in content even with -rea off |
| Qwen3.5-35B-A3B | Q4_K_M | turbo4/turbo4 | off | 23,910 MB | 188 | 11/17 | Fast but LRU fails. Q4_K_M sensitivity |

**Thinking off is equal or better for every model**. The Gemma 31B delta is transformative:

| Model | turbo4 + thinking on | turbo4 + thinking off |
|---|---|---|
| Gemma 26B Q6_K | 17/17 | 17/17 |
| Gemma 31B Q4_K_M | 4/17 | **17/17 (+13)** |
| Gemma 26B Q4_K_M | 13/17 | 16/17 (+3) |
| Qwopus 27B Q6_K | 16/17 | 16/17 |
| Qwen 35B Q4_K_M | 6/17 | 11/17 (+5) |

## Throughput cost

TurboQuant isn't free — dequant overhead is ~5-10% throughput at 32K on the 5090:

| Model | f16 KV | turbo4 KV | Delta |
|---|---|---|---|
| Gemma 26B Q6_K | 152 | 142 | −7% |
| Gemma 26B Q4_K_M | 164 | 156 | −5% |
| Gemma 31B Q4_K_M | N/A† | 53 | N/A |
| Qwopus 27B Q6_K | 52 | 52 | ~0% |

† Gemma 31B can't run at useful context with f16 KV on 32 GB — no baseline exists.

At short context this is modest. At long context (128K+) turbo4 is actually **faster** than f16 because f16 would spill to system RAM while turbo4 stays in VRAM.

## Model-size effect on compression quality

Quality cost scales inversely with model size — smaller models pay more.

| Model | turbo4 vs f16 KV |
|---|---|
| Gemma 4 31B-IT (31B dense) | 17/17 → 17/17 |
| Gemma 4 26B-A4B (4B active MoE) | 17/17 → 17/17 |
| Qwen3-8B (8B dense) | 12/17 → 8/17 (−24%) |
| Gemma 4 E4B (2.3B active) | 5/22 → 5/22 (both at floor) |

Models with ~20B+ active params absorb the quantization noise. 8B models lose meaningful quality. Below ~5B, the model is already at our benchmark suite's capability floor. Practical implication: use turbo4 freely on 26B+ models; on 8B models prefer TriAttention (token eviction) instead.

## Per-model KV configs

Architecture determines the right lever:

| Model | Arch | KV/tok (f16) | Recommendation |
|---|---|---|---|
| Qwen3.5-35B-A3B | DeltaNet hybrid (10 attn layers) | ~20 KB | **q8_0** — savings negligible; turbo causes thinking loops |
| Qwopus 27B | DeltaNet hybrid (16 attn layers) | ~64 KB | **turbo3** — moderate savings, handles quantization well |
| Gemma 4 26B-A4B | Sliding window + global (15+15) | ~120 KB | **turbo4** — significant savings, turbo3 too aggressive |
| Gemma 4 31B-IT | Dense (50 sliding + 10 global) | ~870 KB | **turbo3** — must compress; dense arch is VRAM-hungry |

**K precision controls quality** (research consensus confirmed in our runs): keys drive softmax attention routing; values are comparatively cheap quality-wise. This is why turbo4 (+0.23% PPL) is dramatically safer than turbo3 (+1.06% PPL) for thinking models — the extra bit per key preserves the attention patterns that gate "stop thinking, start coding" transitions.

## Thinking-budget exhaustion with turbo3

The most significant finding across all 5090 runs: **turbo3 KV causes some models to enter infinite thinking loops**, burning their entire token budget on chain-of-thought and producing no code.

| Run | Config | Models affected | Cause |
|---|---|---|---|
| turbo3 everywhere, 32K ctx | turbo3/turbo3 | Qwen 35B (0/17), Gemma Q6_K partial | KV quant noise disrupts "stop thinking" signal |
| turbo3, parallel=4 | turbo3/turbo3 | Same + Qwopus degraded | FP accumulation order compounds |
| turbo3, 96K ctx, 64K max | turbo3 single-slot | Gemma Q6_K (65K tokens of thinking!) | More budget = more wasted thinking |
| **Optimal** | **per-model KV + reasoning budget** | **None** | Solved via turbo4 + budget cap |

The fix was not more tokens — it was better settings. turbo4 instead of turbo3 for Gemma, q8_0 for Qwen, reasoning budget caps as safety net.

## Full context at optimal KV

| Model | Weights | KV config | KV at 262K | Total | Fits 32 GB? | Max usable ctx |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B Q4_K_M | ~21 GB | q8_0 | ~5.2 GB | ~26 GB | Yes | 262K full |
| Qwopus 27B Q6_K | ~23 GB | turbo3 | ~4.2 GB | ~27 GB | Yes | 262K full |
| Gemma 4 26B-A4B Q4_K_M | ~17 GB | turbo4 | ~9.4 GB | ~26 GB | Yes | 262K full |
| Gemma 4 26B-A4B Q6_K | ~23 GB | turbo4 | ~9.4 GB | ~32 GB | Barely | ~230K |
| Gemma 4 31B-IT Q4_K_M | ~20 GB | turbo3 | ~57 GB | ~77 GB | No | ~58K |

Gemma Q6_K with f16 KV at 256K required ~41 GB and dropped from 150 to 55 tok/s due to RAM spill. With turbo4 KV, it fits to ~230K entirely in VRAM.

## Asymmetric K/V limitation

turboquant_plus recommends asymmetric K/V (e.g. `-ctk q8_0 -ctv turbo3`) as the safest config since K precision controls quality. However, [ggml-org/llama.cpp#20866](https://github.com/ggml-org/llama.cpp/issues/20866) documents that **asymmetric K/V types cannot be GPU-offloaded** — they force CPU processing, dropping throughput to ~30 tok/s. Until fixed, symmetric turbo4/turbo4 is the practical optimum.

---

# MacBook Pro M4 Max (36 GB)

> **⚠ Update 2026-04-11**: Most "turbo4 is mandatory" claims in the original M4 Max write-up were base-version artifacts. When the planarquant rotorquant fork rebased onto a newer llama.cpp base (via cherry-picking upstream PR #21309), we discovered:
>
> 1. **The "Gemma 4 31B-IT Q4_K_M f16 OOMs at 16K, requires turbo4" claim was wrong math.** Original calc assumed full attention on every layer; Gemma 4 31B uses ISWA with ~9 of 62 layers as global. Real KV is ~120 KB/token averaged, ~3.7 GB at 32K f16 — not the 27.8 GB originally projected. On a current base, **Gemma 4 31B-IT f16 fits at 64K context with default `-ub 512`** — no turbo4 needed.
> 2. **The "turbo4 −23% speed penalty" numbers on Gemma 4 26B-A4B were measured on an old base with a buggy compute buffer.** On a newer base both f16 and turbo4 get a 16× smaller compute buffer, changing the bandwidth math underneath both.
> 3. **Turbo4 is still useful for long context on Gemma 4 31B-IT.** New-base measurements: turbo4 fits Gemma 31B at 128K (21 GB) and 256K (24 GB) with default ub, where f16 OOMs at 128K. Turbo4 becomes an optional context-extender rather than a mandatory workaround.
> 4. **The "−23% always" number should be read as a data point from a specific pinned commit**, not a permanent Apple-Silicon property. The mechanism — turbo4's dequant compute dominating on Metal because KV bandwidth is not the bottleneck — is still broadly correct.
>
> The sections below preserve the historical analysis. Take specific numbers with this caveat in mind.

## TL;DR on M4 Max

| Model | Without turbo4 (f16 KV) | With turbo4 KV | Net |
|---|---|---|---|
| Gemma 4 31B-IT Q4_K_M @ 16K (old base) | OOM | ✅ 11.5 tok/s, 17/17 | Required on old base; optional on new |
| Gemma 4 26B-A4B Q6_K @ 16K | ✅ **60 tok/s**, 15/17 | 46 tok/s, 16/17 | **−23% speed**, tied quality |
| Gemma 4 26B-A4B Q6_K @ 32K | OOM | OOM | No help (weights are the bottleneck) |
| Anything else tested | f16 fits at full context | turbo4 hurts speed | **Don't use turbo4** |

**Decision rule**: use turbo4 KV if and only if the model literally won't load with f16. There's no other use case where it's the right choice on this hardware.

## Why Gemma 4 31B-IT is (was) the one case

Original calc: `KV bytes/token = 2 × 62 layers × 16 KV heads × 256 head_dim × 2 (FP16) ≈ 870 KB` — at 16K, 13.6 GB KV + 18 GB weights = 32 GB, beyond the ~30 GB Metal working set ceiling. Turbo4 drops KV to 3.5 GB, total 21.5 GB — fits.

**With the corrected ISWA math**, Gemma 4 31B KV is actually ~120 KB/token average (only ~9 of 62 layers are global), and the 16K f16 scenario fits on a current base. But at 128K+, turbo4 is still the only way to fit on 36 GB.

The 5090 ranking has this same observation with different framing: turbo4 extends Gemma 31B from 16K → 58K on the 5090 (VRAM-only). Same root cause (per-token KV cost on the dense global-attention layers), different solution profile.

## What turbo4 doesn't save you from on M4 Max

For most M4 Max OOMs, the **compute buffer** is the bottleneck, not the KV cache.

- **Gemma 4 26B-A4B Q6_K @ 24K+**: 22 GB weights + 6+ GB compute buffer is already at the limit. KV (with or without turbo4) adds <1 GB at 32K. **Both f16 and turbo4 OOM at 24K+.**
- **Anything Q6_K above ~22 GB at non-trivial context**: Q8_0 of these models is ~30 GB by itself, leaving zero room for compute or KV.
- **Long contexts (>32K) on any ≥26B model**: compute buffer scales linearly at ~260 MiB per 1024 tokens for sliding-window architectures. By 64K it's 16 GB on top of weights. Turbo4 doesn't help.

The "just compress the KV more" instinct is a CUDA habit that doesn't transfer. On the M4 Max, more aggressive **weight** quantization (Q4 instead of Q6) is usually a better lever — shrinks the dominant cost (weights), which makes room for the second-largest cost (compute buffer). KV compression fights over the smallest line item.

## Speed cost (pinned-commit data)

Gemma 4 26B-A4B Q6_K at 16K context, both KV configs, same prompts:

| Config | Weight read/tok | KV read/tok (rough) | Tok/s | % of 410 GB/s peak |
|---|---|---|---|---|
| f16 KV | ~4 GB (MoE 4B active) | ~3 GB | **60.3** | ~88% util |
| turbo4 KV | ~4 GB | ~0.8 GB | 46.0 | ~54% util |

f16 is 31% faster and hits 88% bandwidth utilization. turbo4 drops to 54% because dequant compute eats more cycles than the smaller KV reads save in bandwidth. Quality: f16 = 15/17, turbo4 = 16/17 (single-shot variance, not a real difference).

## M4 Max decision tree

1. **Default to `-ctk f16 -ctv f16`.**
2. **First lever for OOMs is `-ub 256`, not turbo4 KV.** The compute buffer is usually the actual bottleneck. Halving `-ub` (physical batch size) cuts the compute buffer in half with zero decode-throughput cost. See [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md) for the full sweep.
3. **Use turbo4 KV only when the KV cache itself is the bottleneck.** Basically Gemma 4 31B-IT, where dense/global KV puts the cache at a meaningful GB count. For most other models KV is <1% of the working set on Metal.
4. **Don't expect a speedup from turbo4 on Apple Silicon.** It's purely a capacity workaround.
5. **If `-ub 256` + turbo4 still OOM, the bottleneck is the weights themselves.** Drop to a smaller quant (Q6 → Q4) or switch to a smaller model class.

## Future experiments (M4 Max)

- **turbo3 vs turbo4** — only turbo4 was tested on M4 Max. The 5090 ranking found turbo3 faster but riskier; whether the tradeoff exists on Metal is unknown.
- **Asymmetric `-ctk f16 -ctv q8_0`** — standard llama.cpp flag (only quantize V; keep K for tool-calling safety). Might offer a milder speed-vs-capacity tradeoff.
- **`iogpu.wired_limit` override** — `sudo sysctl iogpu.wired_limit_mb=33024` bumps macOS's GPU memory ceiling from 75% to ~92%; would let Gemma 31B run with f16 KV at 16K (32 GB total fits in 33 GB ceiling). Untested.

---

# Cross-engine notes

## vs NVFP4-turbo (vLLM) on Gemma 31B — RTX 5090

We tested LilaRest's `gemma-4-31B-it-NVFP4-turbo` against our standard llama.cpp + turbo4 pipeline.

| Config | Score | Tok/s | VRAM | Max ctx | Setup |
|---|---|---|---|---|---|
| llama.cpp Q4_K_M + turbo4 | **17/17 (100%)** | **53** | **22.3 GB** | ~58K | Medium |
| vLLM NVFP4-turbo + FP8 KV | 16/17 (94%) | 41 | 29.6 GB | ~16K | High (CUDA 13, cu130 wheel, JIT compile) |

For single-stream coding workloads **llama.cpp + turbo4 wins on every metric we measure**. NVFP4-turbo's strength is concurrent batched throughput (1,244 tok/s in published benchmarks) — relevant for production serving, not our workload.

## TriAttention vs TurboQuant on Qwen3-8B — RTX 5090

Same model, three compression strategies:

| Approach | Engine | Compression | Best-of-3 | Avg |
|---|---|---|---|---|
| TriAttention (token eviction, 4096 budget) | vLLM | Keep subset at full precision | **14/17 (82%)** | 8.3/17 |
| f16 baseline | llama.cpp | None | 12/17 (71%) | 9.1/17 |
| turbo4 | llama.cpp | KV quantization (3.8×) | 8/17 (47%) | 5.7/17 |

TriAttention wins on 8B models. Token eviction (keeping 4096 of 16K+ tokens at full precision) preserves more quality than quantizing all tokens to 4 bits. turbo4 hurts 8B models more than 27B+ models: on Gemma 26B/31B it scores 17/17, on 8B it drops to 8/17. Small models have less redundancy to absorb quantization noise. **Compression strategy depends on model size.**

---

# Settings reference

## Build (RTX 5090)

```bash
cd T:/git/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

## Per-model server invocations (5090, optimal configs)

```bash
# Gemma 26B Q6_K (S-tier)
llama-server -m gemma-4-26B-A4B-it-Q6_K.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo4 -ctv turbo4 --reasoning-budget 12288

# Qwopus 27B Q6_K (A-tier)
llama-server -m Qwopus3.5-27B-v3-Q6_K.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo3 -ctv turbo3

# Qwen 35B Q4_K_M (fast, less reliable)
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk q8_0 -ctv q8_0
```

## Key flags

| Flag | Purpose | Notes |
|---|---|---|
| `-ctk TYPE` | KV cache K quantization | turbo2/turbo3/turbo4/q8_0/q4_0/f16 |
| `-ctv TYPE` | KV cache V quantization | Same options as K |
| `--reasoning-budget N` | Cap thinking tokens | -1=unlimited, 0=no thinking, N=max |
| `-fa on` | Flash attention | **Required** for turbo KV — without it, turbo is slower than f16 |
| `-np N` | Parallel slots | Divides context per slot. Use 1 for max context per request |

## See also

- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — full per-model tier list with all configs
- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — M4 Max tier list
- [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md) — what fits at what context on Metal
- [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md) — KV cache spill cliff on the 5090
- [ROTORQUANT.md](ROTORQUANT.md) — newer K-only Givens-rotation KV quantizer
