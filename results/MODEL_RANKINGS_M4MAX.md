# Local Model Tier List — MacBook Pro M4 Max 36 GB

**Platform:** Apple M4 Max (14C CPU / 32C GPU binning), 36 GB unified LPDDR5X, 410 GB/s
**Inference:** llama.cpp (planarquant fork with Gemma 4 cherry-picks, Metal backend) — see [ROTORQUANT.md](ROTORQUANT.md) for build instructions
**Standard config:** flash attention (`-fa on`), `max_tokens=16384`, `-ctk f16 -ctv f16` (default; rotorquant K-only on dense GQA models — see KV format section), thinking off (`--reasoning-budget 0`), `-np 1 --jinja`, default `-ub 512`

Tested April 2026. Last refreshed 2026-04-12 after the rotorquant + llama.cpp base upgrade.

> **What changed in this refresh.** An earlier version of this document was built on the `turboquant` fork at commit `8590cbff9`, which had a compute-buffer allocation bug for Gemma 4 models (~16× oversized) that forced us into `-ub 256` workarounds and into turbo4 KV as "mandatory" for Gemma 31B. Neither is true on a current llama.cpp base. After rebasing onto the `planarquant` fork and cherry-picking Gemma 4 support (upstream PR #21309 + #21326):
>
> - **`-ub 256` is no longer required.** Default `-ub 512` works at 32K on all models below. The old "compute buffer scales with n_ubatch × n_ctx" claim was the bug, not the architecture.
> - **Turbo4 KV is no longer mandatory for Gemma 4 31B-IT.** f16 fits at 64K on the new base; turbo4 is now an optional context extender for 128K+, not a fit requirement.
> - **Plain f16 KV is much faster on the new base than the old-base turbo4/ub=256 configs it replaces.** Gemma 4 31B-IT f16 @ 32K default ub runs at **15.3 tok/s** vs the old 11.8 tok/s turbo4 number (+30%). Gemma 4 26B-A4B Q6_K f16 @ 32K runs at **65.6 tok/s** vs the old 60.3 tok/s f16 @ 16K number. The base upgrade by itself (without rotorquant) produces most of the speedup.
> - **Rotorquant `planar3/f16` K-only is a +19% win on dense GQA (Qwen 27B), but a wash on Gemma 4.** Originally I measured +13-20% across three models, but retesting on the new base at matched ub shows Gemma 4 26B-A4B Q6_K is within 2% between f16 and planar3/f16. The original wins on Gemma 4 were cross-base comparisons (old-base f16 vs new-base planar3/f16) that double-counted the base upgrade. See [ROTORQUANT.md](ROTORQUANT.md) for the detail.
>
> The tier ordering below hasn't changed, but all S/A-tier throughput numbers, context ceilings, and config examples are new measurements.

## The M4 Max's defining constraint: the Metal working set is ~30 GB

Three things still shape what runs on this machine, but the old story ("compute buffer dominates and scales with ubatch") was base-version-specific and no longer applies. The real constraints:

1. **Metal working set is ~30 GB**, not the full 36 GB. `recommendedMaxWorkingSetSize ≈ 30,150 MB`. macOS reserves the rest for the system. Going over produces `kIOGPUCommandBufferCallbackErrorOutOfMemory` immediately, no graceful spillover.

2. **KV cache math is model-specific.** The old rule of thumb was "KV is <1% of the working set, don't bother compressing." That's right for Gemma 4 MoE (sliding-window attention keeps KV tiny) and wrong for Gemma 4 31B-IT (dense with ISWA still leaves ~120 KB/token averaged KV — 3.7 GB at 32K). Before assuming KV doesn't matter, compute it: `2 × n_layers × n_kv_heads × head_dim × 2 (FP16) × ctx`, adjusted down by whatever fraction of layers use sliding-window.

3. **No PCIe spill.** When the 5090 hits its 32 GB limit, it spills to system RAM via PCIe (slow but functional — see [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md) for the cliff at 256K). The Mac has no PCIe to spill across; either it fits or it OOMs.

On the current base, the compute buffer is **~523 MiB regardless of context** for Gemma 4 sliding-window architectures (down from 8.4-16.7 GB on the old buggy base — a ~32× reduction). This is what opened up 32K+ context on all the Gemma 4 models.

### Practical context limits, by model

These are on the planarquant fork with Gemma 4 cherry-picks, default `-ub 512`. All measurements below use `-ctk f16 -ctv f16` unless noted.

| Model | Quant | Max ctx (default ub) | Working set | How to go further |
|---|---|---|---|---|
| Gemma 4 26B-A4B | Q6_K | **32K** | ~23 GB | Compute buf is small, so `-ub 256` + longer ctx works |
| Gemma 4 26B-A4B | Q4_K_M | **64K+** | ~17 GB | Lots of headroom, weights are small |
| Gemma 4 31B-IT | Q4_K_M | **64K** (f16 KV) | 24.3 GB | Switch to `-ctv turbo4` → 128K at 21 GB, 256K at 24.1 GB |
| Qwen 3.5 27B Opus-Distilled | Q4_K_M | **128K** (f16 or planar3/f16) | ~24 GB | 256K loads but decode collapses to 1.7 tok/s; not usable |
| Qwen 3.5 9B / Nemotron 4B | Q4_K_M | 32K+ | ≪ 30 GB | Small footprint, plenty of headroom |

**Key asymmetry from the 5090**: the 5090 ranking shows Gemma 4 26B-A4B Q6_K at ~230K context. We haven't pushed the M4 Max that far, but the math says it's in reach with `-ub 256` or smaller. The earlier claim that "compute buffer grows linearly with ubatch × ctx" was a base-specific bug, not a Metal-intrinsic property.

Two follow-on findings:

- **Rotorquant `planar3/f16` K-only helps on dense GQA, not on Gemma 4.** Measured cleanly on the new base at matched ub: Qwen 27B Opus-Distilled gets +19% (15.5 vs 13.0 tok/s) at identical quality — a real win. Gemma 4 26B-A4B Q6_K at 32K default ub is a **wash** (64.5 vs 65.6 tok/s, noise-level). The original +13-20% Gemma 4 numbers I reported were cross-base comparisons that double-counted the base upgrade. Use planar3/f16 on dense GQA models; default to f16/f16 on Gemma 4. See [ROTORQUANT.md](ROTORQUANT.md).
- **TurboQuant KV is now an optional context extender, not a speed or fit requirement.** On Apple Silicon, turbo4's dequant compute still costs decode throughput vs f16 on every model where both fit (measured: -20% to -30% on Gemma 4 31B-IT). But unlike planar3/f16 (deferred quantization, no memory savings), turbo4 compresses at allocation time — so on Gemma 4 31B-IT it's the path to 128K/256K context when you need it. Use f16 first; reach for turbo4 only to extend context beyond what f16 fits.

---

## S-Tier: The Best Quality That Fits

### Gemma 4 31B-IT Q4_K_M (f16 KV)
The highest-quality model on this machine. Dense 31B with ISWA (~9 of 62 layers global, rest sliding-window). Perfect single-shot score. On the new base, f16 KV fits at 32K and 64K at default ub without drama, and runs at 15.3 tok/s — +30% over the old turbo4/ub=256 workaround.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Throughput | **15.3 tok/s** (f16/f16, new base default ub) |
| Weight size | 18.3 GB (Q4_K_M) |
| Context | **32K** default ub (comfortable) / **64K** also fits with f16 / **128K–256K** with turbo4 KV |
| KV format | f16/f16 (fastest at ≤64K), turbo4/turbo4 (extends context to 256K, ~20% slower) |
| Config | `-c 32768 -fa on -ctk f16 -ctv f16 --reasoning-budget 0 --jinja` |

**Strengths:** Same quality as a 5090 (also 17/17 on this suite). Per-benchmark perfect: ExprEval 5/5, A* 6/6, LRU 6/6.
**Weakness:** Throughput. 15 tok/s is usable but not snappy; a 2K-token generation still takes ~2 minutes. Acceptable for "give me the right answer once" workflows; painful for chat.
**Rotorquant K-only:** `-ctk planar3 -ctv f16` was previously reported as +20% on this model, but that comparison was against the old-base turbo4/ub=256 config (11.8 tok/s). On a matched same-base comparison at default ub, rotorquant K-only's benefit on Gemma 4 disappears (see [ROTORQUANT.md](ROTORQUANT.md)). Default to f16/f16.
**Context tradeoff:** At 128K, f16 OOMs at 30.5 GB. If you need 128K+, use `-ctk turbo4 -ctv turbo4` instead: 128K fits at 21 GB and 256K at 24.1 GB, at a ~20% throughput penalty from the turbo4 dequant overhead.
**Why not Q6_K?** 31B Q6_K is ~25 GB of weights; leaves no room for KV. Untested but unlikely to load.

---

### Gemma 4 26B-A4B Q6_K (f16 KV)
The all-around best quality + speed combination on this machine. MoE with ~4B active parameters per token, so throughput is much higher than the 31B dense. Quality matches the 5090's S-tier within single-test single-shot variance (15-16/17 vs 17/17).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **15/17 (88%)** |
| Throughput | **65.6 tok/s** @ 32K default ub (new base) |
| Weight size | 22.6 GB |
| Context | **32K** at default ub on the new base (no `-ub 256` needed) |
| KV format | f16/f16 (rotorquant planar3/f16 is a wash here — 64.5 tok/s, within noise) |
| Config | `-c 32768 -fa on -ctk f16 -ctv f16 --reasoning-budget 0 --jinja` |

**Strengths:** Fastest high-quality model on this hardware. Per-benchmark: ExprEval 3/5, A* 6/6, LRU 6/6.
**Compared to other KV formats on the same model on the new base:** planar3/f16 → 64.5 tok/s @ 15/17 (1.5% slower, noise). turbo4/turbo4 → 46 tok/s @ 16/17 on the old base; we haven't re-measured turbo4 on the new base but the ~25-30% dequant penalty likely still holds. **f16 is the right default on Gemma 4 MoE on Metal.**
**The old `-ub 256` workaround is obsolete.** On the old turboquant fork base, 32K f16 OOMed because of an 8.4 GB compute buffer bug. On the planarquant fork + cherry-picks, 32K fits at default ub with room to spare (working set ~23 GB vs 30 GB ceiling).

---

## A-Tier: Best Dense Speed (with caveats)

### Qwen 3.5 27B Opus-Distilled MLX 4bit (mlx_lm)
The fastest *dense* 27B+ model on this machine — and one of only two cases where MLX outperforms llama.cpp on this hardware. **42% faster** than the same model under llama.cpp at f16 KV (18.5 vs 13.0 tok/s) at ~99% bandwidth utilization. Run via `mlx_lm.server`, not llama.cpp.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **13/17 (76%)** |
| Throughput | **18.5 tok/s** |
| Weight size | ~14 GB (MLX 4-bit) |
| Context | 32K (mlx_lm default) |
| Config | `mlx_lm.server --model mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --port 8766 --temp 0` |

**Per-benchmark:** ExprEval 5/5 (24/28 tests pass — over-engineered), A* 6/6, LRU 2/6.
**Strengths:** Best speed/quality combo for dense 27B+ on this hardware. Beats llama.cpp on the same model in both speed AND single-shot code quality.
**Weakness:** mlx_lm.server takes 5-15 min to load, vs llama-server's seconds. Slow iteration. LRU cache fell apart at 2/6.

---

### Qwen 3.5 27B Opus-Distilled Q4_K_M (llama.cpp, planar3/f16)
The same model under llama.cpp's Metal backend. Slower than MLX but easier to set up, and rotorquant K-only brings it within 17% of MLX's speed at the same capacity.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **15.5 tok/s** (planar3/f16, +19% vs f16's 13.0) |
| Weight size | 16.5 GB |
| Context | 128K at default ub |
| KV format | planar3/f16 |
| Config | `-c 131072 -fa on -ctk planar3 -ctv f16 --reasoning-budget 0 --jinja` |

**Per-benchmark:** ExprEval 5/5, A* 0/6 (model emitted `except ImportError:` outside try block — real syntax error), LRU 6/6.
**Caveat:** Single-shot temp 0 variance. The 5090 saw 100% on this exact GGUF. Use the MLX version above unless you specifically need llama.cpp.

---

### Gemma 4 26B-A4B Q4_K_M (f16 KV)
The "fits everywhere with room to spare" version. ~16 GB weights leaves plenty of headroom for long-context KV. Quality drops noticeably from Q6_K — including a literal incomplete-line truncation in the LRU cache impl (`self.tail.prev = self.` and then EOF). The Q4_K_M quality risk that the 5090 ranking warns about is real and apparently amplified by the Mac inference path.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Throughput | **58.9 tok/s** (old-base measurement; likely higher on new base, untested) |
| Weight size | 16.5 GB |
| Context | 64K+ at default ub on new base |
| KV format | f16/f16 |

**Per-benchmark:** ExprEval 5/5, A* 6/6, LRU 0/6 (truncated impl).
**Bottom line:** Take Q6_K if you don't need >32K context and you have the memory headroom. Take Q4_K_M only if you specifically need long context on this model class.

---

## B-Tier: Small + Fast

### Qwen 3.5 9B Q4_K_M (thinking off)
The fastest dense model that produces usable code on this machine. With thinking off, scores are roughly half the 5090's Q8_0 baseline (15/18 → 9/17) — most of the gap is the quant difference (Q4_K_M vs Q8_0), some is single-shot variance.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **9/17 (53%)** |
| Throughput | **35.4 tok/s** (thinking off) |
| Weight size | 5.5 GB |
| Context | 32K |
| Config | `-c 32768 -fa on -ctk f16 -ctv f16 --reasoning-budget 0` |

**Thinking on/off:** Thinking ON drops this to 3/17 (18%). Same model, same prompts, same temp — thinking burns tokens then runs out of room for content. Always disable for this model on this hardware.

---

### Nemotron 3 Nano 4B Q4_K_M (thinking on)
The fastest model that produces usable output. 4B parameters, ~58-65 tok/s. **Thinking ON is meaningfully better here**: 7/17 with thinking, 4/17 without — opposite polarity from Qwen 9B.

| Metric | Value |
|---|---|
| Single-shot (temp 0, thinking on) | **7/17 (41%)** |
| Single-shot (temp 0, thinking off) | 4/17 (24%) |
| Throughput | **65.5 tok/s** (thinking on), 58.1 tok/s (thinking off) |
| Weight size | 2.8 GB |
| Context | 32K |

**Why use it:** Lowest VRAM, fastest, runs on battery. If you need *some* model and the rest are too big or too slow, this is the floor.
**Why not:** Hard benchmarks (LRU) are mostly out of reach. Use it for short-form completions, not full implementations.

---

## Which KV Format to Use on M4 Max

The right KV format depends on both the model architecture and what you're optimizing for. On this hardware the options are:

| KV format | Speed vs f16 | Memory vs f16 | Good for |
|---|---|---|---|
| `-ctk f16 -ctv f16` | baseline | baseline | **Default for Gemma 4 (both 26B-A4B and 31B-IT).** Tied with planar3/f16 on Gemma 4 MoE and on Gemma 4 31B at default ub. |
| **`-ctk planar3 -ctv f16`** (rotorquant K-only) | **+19% on dense GQA**; ~tied on Gemma 4 | identical (deferred quant) | **Dense GQA models with meaningful KV bandwidth** (Qwen 27B Opus-Distilled). Requires the planarquant fork. |
| `-ctk turbo4 -ctv turbo4` | **−20% to −30%** | **much smaller** (~4× smaller KV) | Context extension only. Gemma 4 31B-IT 128K+. Never for speed. |
| `-ctk planar3 -ctv planar3` (symmetric) | −15% on Qwen; **broken on Gemma 4** | smaller | Avoid on Gemma 4 — runaway generation bug from missing Metal V-dequant inverse rotation |
| `-ctk f16 -ctv q8_0` | untested on M4 Max | slightly smaller V | Tool-calling safety fallback (never quantize K) when you need some V savings without rotorquant |

**Rule of thumb:**
1. **Gemma 4 (26B-A4B or 31B-IT) up to 64K → `f16/f16`.** Rotorquant doesn't help here on the new base; the old +13-20% numbers were cross-base measurement artifacts.
2. **Dense GQA 27B+ (Qwen 3.5 27B Opus-Distilled) → `planar3/f16`.** This is the clean +19% rotorquant win — KV bandwidth is a meaningful fraction of per-token cost at this model class and deferred K quantization actually saves on it.
3. **Gemma 4 31B-IT beyond 64K → `turbo4/turbo4`.** Only way to reach 128K/256K with the weights that big. ~20% throughput penalty vs f16.

See [ROTORQUANT.md](ROTORQUANT.md) for the measurement detail behind these rows and [TURBOQUANT.md](TURBOQUANT.md) for the turbo4 speed-cost explanation.

---

## MLX vs llama.cpp on Apple Silicon

The HARDWARE_SPECS.md footnote claimed "MLX sometimes outperforms llama.cpp on Apple Silicon by 10–30%." We tested this directly using `mlx_lm.server` 0.31.2 against `mlx-community/*` quants, same prompts, temp 0.

**The answer is "it depends on the model class."** MLX wins on dense ≥27B models, loses on MoE models, and ties on small dense.

| Model | llama.cpp | MLX | Winner |
|---|---|---|---|
| Qwen 3.5 9B (Q4_K_M / 4bit, dense) | 35.4 tok/s, 9/17 | 33.9 tok/s, 10/17 | tie (~4% MLX slower) |
| Gemma 4 26B-A4B (Q6_K f16 / 6bit, **MoE 4B-active**) | **65.6 tok/s**, 15/17 | 33.5–37.4 tok/s, 12/17 | **llama.cpp** (-43% on MLX) |
| Gemma 4 31B-IT (Q4_K_M f16 / 4bit, dense) | **15.3 tok/s**, 17/17 | 7.9–11.9 tok/s, 17/17 | **llama.cpp** (new base f16 wins on speed at same quality) |
| Qwen 3.5 27B Opus-Distilled (Q4_K_M / 4bit, **dense**) | 15.5 tok/s, 11/17 (planar3/f16) | **18.5 tok/s**, 13/17 | **MLX** (+19% speed, +12% quality) |
| Nemotron 3 Nano 4B (Q4_K_M / 6bit) | 65.5 tok/s, 7/17 | **N/A** | mlx_lm 0.31.2 hits transformers `KeyError: '-'` parsing nemotron_h layer pattern |

### The pattern: MoE active params hurt MLX, dense weights help it — and rotorquant narrows the gap on Qwen

Bandwidth utilization (rough) = `(weights+KV bytes per token) × tok/s ÷ peak bandwidth`. Plugging in real numbers:

- **Qwen 27B dense Q4** at 15.5 tok/s (llama.cpp planar3/f16) ≈ 83% of 410 GB/s peak. Same model at 18.5 tok/s (MLX) ≈ 99% of peak. **MLX still wins but rotorquant cuts the gap from 42% → 19%.**
- **Gemma 4 26B-A4B Q6** (only ~4 GB read per token from MoE) at 65.6 tok/s (llama.cpp f16) ≈ 94% of peak. MLX at 33 tok/s is under-performing here — the MLX kernels don't exploit the sparse MoE activation pattern as efficiently.

Plus:
- **Gemma 4 31B-IT 4bit MLX matched llama.cpp on quality (17/17)** — the same 5/5 / 6/6 / 6/6 breakdown. The 4-bit MLX quantization doesn't lose accuracy on this model.
- **Gemma 4 26B-A4B 6bit MLX had a chat-template regression**: ExprEval went runaway (16K tokens of misclassified `reasoning_content` in the output before any actual code). The mlx_lm.server handling of Gemma 4's reasoning fields is broken; this contributed to the speed loss above and the 0/5 ExprEval score.

### Recommendations

- **Dense GQA ≥27B model on Mac → consider MLX first, or rotorquant K-only on llama.cpp second.** MLX stays fastest for Qwen 27B Opus; rotorquant cuts the gap to ~19%.
- **MoE model on Mac → use llama.cpp with plain f16.** MLX kernels lose 40-45% on this class, and rotorquant is a wash on Gemma 4 MoE.
- **Gemma 4 specifically on MLX → wait for the chat-template fix.** Outputs are sometimes garbage due to reasoning misclassification.
- **Gemma 4 31B-IT → llama.cpp new-base f16 wins.** 15.3 tok/s vs MLX's 8-12 tok/s at identical 17/17 quality.

---

## Thinking ON vs OFF

| Model | Thinking ON | Thinking OFF | Winner |
|---|---|---|---|
| Qwen 3.5 9B Q4_K_M | 3/17 (18%) | **9/17 (53%)** | **OFF (+6)** — burns budget thinking |
| Nemotron 3 Nano 4B Q4_K_M | **7/17 (41%)** | 4/17 (24%) | **ON (+3)** — actually uses the reasoning |

Two models, two opposite outcomes — same lesson the 5090 ranking found: thinking on/off is **model-dependent**. Default off and flip on per model after testing.

---

## Quick Reference: Choosing a Model

| Priority | Pick | Engine | Quant | KV | Context | Why |
|---|---|---|---|---|---|---|
| **Best quality** | Gemma 4 31B-IT | llama.cpp planarquant | Q4_K_M | f16/f16 | 32K–64K | 17/17, **15.3 tok/s** (usable, +30% vs old turbo4 number) |
| **Best quality + interactive speed** | Gemma 4 26B-A4B | llama.cpp planarquant | Q6_K | f16/f16 | 32K | 15/17, **65.6 tok/s** at default ub |
| **Best dense 27B at speed** | Qwen 3.5 27B Opus-Distill | **MLX 4bit** | — | — | 32K | 13/17, 18.5 tok/s (19% > llama.cpp planar3/f16) |
| **Best 27B in llama.cpp** | Qwen 3.5 27B Opus-Distill | llama.cpp planarquant | Q4_K_M | planar3/f16 | 128K | 11/17, **15.5 tok/s** (+19% vs f16 — the one clean rotorquant win) |
| **Max context on best quality** | Gemma 4 31B-IT | llama.cpp planarquant | Q4_K_M | turbo4/turbo4 | 128K–256K | 17/17, ~11-12 tok/s (turbo4 speed penalty) |
| **Quality + 64K+ context cheap** | Gemma 4 26B-A4B | llama.cpp planarquant | Q4_K_M | f16/f16 | 64K+ | 11/17, ~60 tok/s, real Q4 quality cost |
| **Maximum throughput** | Nemotron 3 Nano 4B | llama.cpp | Q4_K_M | f16 | 32K | 65 tok/s, but only 41% quality |
| **Lowest memory** | Nemotron 3 Nano 4B | llama.cpp | Q4_K_M | f16 | 32K | 2.8 GB weights |

---

## Configuration

### llama.cpp (planarquant fork) — default

The planarquant fork + Gemma 4 cherry-picks is the recommended binary. Build instructions are in [ROTORQUANT.md](ROTORQUANT.md). It supports `f16`, `turbo3`, `turbo4`, `planar3`, `iso3`, `planar4`, and `iso4` KV cache types.

```bash
# Standard config — Gemma 26B-A4B Q6_K at 32K, default ub, plain f16
~/git/TheTom/llama-cpp-planarquant/build/bin/llama-server \
  -m gemma-4-26B-A4B-it-Q6_K.gguf --port 8765 \
  -c 32768 -ngl 999 -fa on \
  -ctk f16 -ctv f16 \
  -np 1 --jinja --reasoning-budget 0

# Gemma 31B — plain f16 at 32K or 64K; for longer context use turbo4 KV
... -c 32768 -ctk f16 -ctv f16 ...         # 32K, 15.3 tok/s
... -c 65536 -ctk f16 -ctv f16 ...         # 64K, fits at ~24 GB
... -c 131072 -ctk turbo4 -ctv turbo4 ...  # 128K; use for long context workflows
... -c 262144 -ctk turbo4 -ctv turbo4 ...  # 256K loads, decode speed unverified

# Qwen 27B Opus-Distilled — planar3/f16 gives a clean 19% speedup over f16
... -c 131072 -ctk planar3 -ctv f16 ...

# Small models (Qwen 9B, Nemotron 4B) — plain f16 is fine, no rotorquant needed
... -c 32768 -ctk f16 -ctv f16 ...
```

**What changed from the old config notes:**

- **`-ub 256` is gone.** The compute-buffer bug that forced it was specific to the old turboquant base `8590cbff9`. Default `-ub 512` works on the new base.
- **`f16/f16` is the default KV format again.** On Gemma 4 (26B-A4B and 31B-IT), rotorquant K-only is a wash at matched ub on the new base — the +13-20% I originally reported was a cross-base artifact.
- **`planar3/f16` is worth it specifically on dense GQA ≥27B** (Qwen 27B Opus-Distilled), where KV bandwidth is a meaningful fraction of per-token cost. +19% speedup at identical quality.
- **`turbo4/turbo4` becomes an opt-in context extender for Gemma 31B only.** Use it when you need 128K+ context and can accept the ~20% throughput penalty vs f16.

The Docker build at `~/.docker/bin/inference/llama-server` (build 3191462) is still too old for Gemma 4 (`unknown model architecture: 'gemma4'`). Use the planarquant fork.

### MLX (mlx_lm.server) — for dense ≥27B Qwen

```bash
# Qwen 3.5 27B Opus-Distilled MLX 4bit — fastest dense 27B+ on this machine
/tmp/mlx-venv/bin/python3 -m mlx_lm server \
  --model mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
  --port 8766 --temp 0 --max-tokens 16384 \
  --chat-template-args '{"enable_thinking": false}'
```

mlx-lm 0.31+ requires Python 3.10+ (system Python 3.9 only has wheels up to mlx 0.29.3 which lacks `qwen3_5.py` / `gemma4.py`). Use a brew Python venv:

```bash
brew install python@3.14
/opt/homebrew/bin/python3.14 -m venv /tmp/mlx-venv
/tmp/mlx-venv/bin/pip install mlx-lm
```

Note that mlx_lm.server takes 5-15 minutes to load a model on first invocation. The OpenAI-style endpoints work after that.

## See also

- [ROTORQUANT.md](ROTORQUANT.md) — the rotorquant experiment that changed the KV format defaults here, plus the base-upgrade compute-buffer finding
- [TURBOQUANT.md](TURBOQUANT.md) — why TurboQuant costs speed on Apple Silicon, and when it's still worth using
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — what fits at what context, in detail
- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — full hardware comparison with the 5090 and Spark
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — what these same models score on a 5090 with TurboQuant
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — same models on the bandwidth-bound DGX Spark
