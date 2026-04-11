# RotorQuant on DGX Spark — Results

**Date:** 2026-04-11
**Hardware:** NVIDIA DGX Spark (GB10 Blackwell, compute cap 12.1, 128 GB LPDDR5X @ ~273 GB/s)
**Engine:** `johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` (commit `20efe75cf`), CUDA 13.0, sm_121a, aarch64
**Build:** `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121`
**Pre-experiment predictions:** [ROTORQUANT_HYPOTHESIS.md](../ROTORQUANT_HYPOTHESIS.md)
**Prompts, config, test harness:** same as [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md), 32K ctx, `-np 1 --no-mmap --jinja -rea off`, temp 0, single shot

Every hypothesis from the pre-experiment doc was wrong in an interesting way. Headline: **rotorquant is a net-positive quality bump on Qwen3.5-122B at negligible throughput cost, and it is completely broken on GLM-4.5-Air** — producing literal `Hello???????????` garbage even in the "zero PPL loss" K-only mode.

## Results table

| Model | KV config | Tok/s (decode) | Δ vs f16 | Code score | Δ vs f16 | Verdict |
|---|---|---:|---:|---:|---:|---|
| Qwen3.5-122B-A10B Q4_K_M (bartowski) | `f16 / f16` (baseline) | 25.8 | — | **13/17** | — | — |
| Qwen3.5-122B-A10B Q4_K_M (bartowski) | `iso3 / iso3` | **23.7–25.0** | −3% to −8% | **16/17** | **+3** | Net win |
| Qwen3.5-122B-A10B Q4_K_M (bartowski) | `planar3 K / f16 V` | **25.2–25.6** | ~−1% | **18/17** | **+5** | Clear win |
| GLM-4.5-Air Q4_K_M (bartowski) | `f16 / f16` (baseline) | 21.7 | — | **15/17** | — | — |
| GLM-4.5-Air Q4_K_M (bartowski) | `iso3 / iso3` | **16.2** | **−25%** | **0/17** (broken) | **−15** | **Broken output** |
| GLM-4.5-Air Q4_K_M (bartowski) | `planar3 K / f16 V` | **20.1** | −7% | **0/17** (broken) | **−15** | **Broken output** |

Per-benchmark breakdown for Qwen3.5-122B:

| Benchmark | f16 (baseline) | iso3/iso3 | planar3K/f16V |
|---|---|---|---|
| Expression Evaluator | **0/5** (self.tokens accumulator bug) | **5/5** | **5/5** |
| A* Pathfinding | 7/6 | 7/6 | 7/6 |
| LRU Cache with TTL | 6/6 | 4/6 | 6/6 |
| **Total** | **13/17 (76%)** | **16/17 (94%)** | **18/17 (106%)** |

## What happened on Qwen3.5-122B

**Throughput was essentially free.** iso3/iso3 sat in the 23.7–25.0 tok/s band versus the 25.8 f16 baseline — a −3% to −8% drop on coding benchmarks, vs. my predicted −20% to −50%. Planar3-K-only was within 1% of baseline throughput across all three benchmarks. Both are dramatically better than the `f16K/q8V` asymmetric result from the earlier KV experiment (−46%).

Why the prediction missed: I extrapolated from the q8_0 V-cache result, assuming the dequant compute overhead on Spark CUDA 13 was roughly constant per element and scalar-quantization would be the cheapest possible case. That assumption was wrong. RotorQuant's fused rotate+dequant kernels in the Flash Attention path (`fattn-common.cuh`) apparently beat the generic `q8_0` V-dequant path on bandwidth-bound MoE decode. The prior was drawn from a single data point that turned out not to generalize.

**Quality went up.** This is the headline result. The bartowski + mainline baseline scores 13/17 because of a specific implementation bug on Expression Evaluator — the model chooses a `self.tokens` list accumulator pattern and fails to reset state between evaluator calls, making test 2+ fail. Every rotorquant config I tested landed the model on a *different* generation path that produces correct code:

- iso3/iso3 wrote a clean recursive-descent parser that resets `self.tokens` inside `_tokenize()` → 5/5 ExprEval (vs 0/5 baseline), and hit a new minor bug on LRU (4/6 vs 6/6 baseline). Net: +3 points.
- planar3K/f16V wrote perfect code on all three benchmarks → 18/17 (includes the bonus A* test), which matches the **S-tier ik-llama result** on the same model file.

**This is not evidence that rotorquant is inherently higher quality.** It is evidence that at temp=0, Qwen3.5-122B on this hardware is **pathologically sensitive to numerical noise**, and the bartowski + mainline kernels happen to sit in a local pocket where the model makes a specific coding mistake. Perturb the KV cache numerics in almost any direction (ik-llama's different kernels, rotorquant's rotation-quant, even the q8_0 asym result which scored 14/17) and the model snaps into a different generation path — sometimes better, sometimes worse. RotorQuant got lucky here. It does not improve the model; it just shifts the noise enough to jump out of the bad pocket.

The earlier KV q8_0 asymmetric experiment already hinted at this with its 14/17 vs 13/17 result, which we had written off as "within noise." This experiment confirms the noise band is ±5 points, not ±1 — a huge finding for how we interpret single-shot temp=0 results on this platform going forward.

## What happened on GLM-4.5-Air

**All three code benchmarks timed out at the 600s client read-timeout**, generating more than ~9,700 tokens without ever emitting a natural stop token. The throughput test also hit the `max_tokens=2048` cap, vs. 875 tokens on the f16 baseline — more than 2.3× output inflation and still never stopping. The benchmark harness reported errors for every code benchmark and zero saved output.

To confirm this was real runaway generation and not a config bug, I started a fresh server with the same flags and issued the smallest possible prompt:

```
>>> Say the two words hello world. Then stop.
```

Result from GLM-4.5-Air + `planar3 K / f16 V` at temp 0:

```
FINISH: length
CONTENT: 'Hello???????????????????????????????????????????????????????????????????????????????'
USAGE: {'completion_tokens': 80, 'prompt_tokens': 19}
```

Eighty tokens of `?` after the word "Hello". Finish reason `length` = hit max_tokens, never stopped naturally. **The model does not recognize its own stop token when its K cache is rotated-and-quantized, even in the lightest K-only mode that the rotorquant paper claims is "zero PPL loss."**

The iso3/iso3 symmetric result is the same story but worse — throughput dropped to 16.2 tok/s (−25%, consistent with H1's bracketing) and all three code benchmarks timed out identically. I did not need a second manual confirmation; the planar3K `?` output is sufficient.

### Why GLM breaks but Qwen doesn't

Both models are ~120B total / ~10–12B active MoE with GQA. The difference is attention structure:

- **Qwen3.5-122B-A10B** is a DeltaNet hybrid: only **12 of 48 layers** do full attention with a KV cache. The other 36 layers are linear-attention (DeltaNet) layers whose recurrent state lives outside the rotorquant quantization path. So only ~25% of the model's attention computation sees the KV quantization noise.
- **GLM-4.5-Air** uses standard attention in **every one of its 46 layers**. Every attention op reads a rotated-quantized K cache, every op accumulates the perturbation, and the numerical error compounds across the entire depth of the network.

At the layer-level math this is a 4× difference in how much of the forward pass is affected. The rotorquant paper was validated on Llama 3.1 8B, which has 32 standard-attention layers — more similar to GLM's shape than Qwen's. But Llama 3.1 8B also has 8× fewer total layers to accumulate error through, and the decomposition claim rests on the KV vector's directional structure surviving small rotations. On a 46-layer dense attention stack the error evidently stops being small.

The failure mode is striking: the model doesn't produce subtly-wrong code, it produces literal `?` tokens. This is characteristic of logit corruption so severe that the sampling step finds no coherent continuation and keeps emitting the same high-probability filler token. It does **not** match the "graceful PPL degradation" story the rotorquant paper tells for Llama 3.1 8B (PPL 6.63 → 6.91 for iso3/iso3 is a 4.2% degradation — on GLM, we are not in the same regime).

## Hypothesis verdict

| Hypothesis | Prediction | Result | Status |
|---|---|---|---|
| **H1** | iso3/iso3 slows decode 20–50% on both models | Qwen: −3 to −8%; GLM: −25% | **Half wrong.** Right order on GLM, badly pessimistic on Qwen. |
| **H2** | planar3K/f16V is the least-bad config | Qwen planar3K: ~−1% and +5 points quality; GLM planar3K: still broken output | **Mostly right** on throughput. **Wrong** that it would preserve quality on GLM — K-only is still enough to destroy it. |
| **H3** | Quality is noise-dominated, direction unknowable | Qwen +3/+5 points; GLM −15 points (total collapse) | **Right on noise magnitude, wrong on the failure mode.** The noise band is larger than I guessed on both sides. GLM didn't "flip to a different code path" — it stopped producing coherent output at all. |
| **H4** | Net-negative on Spark MoE | Net-positive on Qwen, catastrophic on GLM | **Wrong on Qwen, right on GLM — by accident.** |
| **H5** | Real use case is dense long-context (Gemma 31B) | Not tested this round | **Still open.** Now has a second motivation: GLM's failure mode suggests rotorquant needs deep-attention stacks to be validated, and Llama 3.1 8B isn't representative. |

## Implications

1. **For Qwen3.5-122B on Spark with mainline llama.cpp: use `planar3 K / f16 V` as a drop-in replacement for `f16 / f16`.** 18/17 at 25.2–25.6 tok/s is materially better than the 13/17 mainline baseline on both axes, and ties the S-tier ik-llama result (17/17 @ 26 tok/s) with a simpler engine stack (no fork, no engine-specific quirks). **Caveat:** this is almost certainly a numerical-noise coincidence, not a general win — one single-shot temp=0 result is not a general quality claim. But if we are already running single-shot temp=0 as our standard, this is the best single-shot config we have measured for bartowski Qwen3.5-122B on mainline.

2. **For GLM-4.5-Air on Spark: do not use rotorquant in any config.** Not iso3, not planar3, not K-only. The model's output is destroyed.

3. **The noise band at temp=0 on Spark is ±5 points, not ±1.** This invalidates any claim based on a single 17-test single-shot run — including, retroactively, much of MODEL_RANKINGS_SPARK.md. The existing rankings remain useful for relative ordering among models with similar scores, but we should stop interpreting a 2–3 point delta between configs as a real quality signal without at least a best-of-3 at temp 0.3 or a multi-seed run.

4. **RotorQuant's quality claim generalizes poorly.** The Llama 3.1 8B result (PPL +4.2%) does not predict GLM-4.5-Air's failure mode (complete generation collapse in 46-layer standard attention). The rotorquant paper would benefit from validation on a dense-attention stack deeper than Llama 3.1 8B before the "better quality than TurboQuant" headline is reliable.

5. **The Qwen result tells us something about DeltaNet hybrids.** Having 3/4 of your attention layers be linear-attention (outside the KV quantization path entirely) makes the model dramatically more tolerant to KV quantization errors on the remaining full-attention layers. This is an underappreciated benefit of the hybrid architecture for KV-compression experimentation. Any future KV compression work on Spark should prefer hybrids as the testbed.

## Follow-ups

- **Re-run the Qwen3.5-122B rotorquant configs at temp=0.3 best-of-3** to distinguish "rotorquant is actually a win here" from "we got lucky with noise." If the mean over 3 runs holds above 14/17 for planar3K, the config is a genuine improvement on the bartowski + mainline baseline.
- **Test rotorquant on Gemma 4 31B Q8_0 (F-tier) at long context** (64K or 128K, not currently feasible in f16 on Spark). This is the actual use case the compression ratio was designed for — and with dense attention but only 32 layers, it should be closer to the rotorquant paper's Llama 3.1 8B regime than GLM is.
- **Measure PPL on wikitext-2** for at least one config on each model. This would separate "numerical noise steers temp-0 path" from "actual quality degradation," which the current single-shot single-seed design can't distinguish.
- **File a finding upstream** (johndpope/llama-cpp-turboquant and/or scrya-com/rotorquant) about GLM-4.5-Air output collapse under planar3 — this is likely a kernel or numerics bug worth their time, not just a "won't work on MoE" disclaimer. The fact that iso3 and planar3K *both* fail in the same way on GLM suggests the issue is in the rotation step, not the quantization step.

## Raw results

Paths are relative to the repo root:

- `experiments/spark_bench/qwen122b-bartowski-iso3/` — 16/17, all benchmarks finished naturally
- `experiments/spark_bench/qwen122b-bartowski-planar3k/` — 18/17, all benchmarks finished naturally
- `experiments/spark_bench/glm-45-air-iso3/` — throughput test only (hit max_tokens=2048); code benchmarks errored with client read-timeout after generating >9700 tokens
- `experiments/spark_bench/glm-45-air-planar3k/` — same as above; throughput 20.1 tok/s; manual `Hello` prompt test captured the runaway `?` failure mode
- Baseline comparisons: `experiments/spark_bench/qwen122b-bartowski/` and `experiments/spark_bench/glm-45-air/` (unchanged from the April 2026 MODEL_RANKINGS_SPARK rebenches)
