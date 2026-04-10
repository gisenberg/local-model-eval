# Local Model Tier List — RTX 5090 32GB

**Platform:** RTX 5090 32 GB VRAM, Windows 11
**Inference:** TurboQuant llama-server fork (`feature/turboquant-kv-cache` branch), CUDA 13.2
**Standard config:** flash attention, `max_tokens=16384`, turbo4 KV (unless noted), thinking off (`-rea off`)

Tested April 2026.

Rankings combine single-shot accuracy (temp 0), multi-run consistency (temp 0.3, best-of-3 and average across 3 runs), and practical factors (speed, VRAM, max context).

> **All TTFT and throughput values below have been corrected** (April 10) after we discovered our Windows Python `requests` client was inflating TTFT by ~2.1s and slightly inflating decode tok/s due to buffering. Re-measured from a Linux client (WSL2) hitting a llama-server built from the same TurboQuant source. The corrected decode numbers are within ~5% of the originals (relative rankings hold) but TTFT is now ~25-40x lower. Benchmark scores were unaffected and have not been re-measured. See [A Note on TTFT](#a-note-on-ttft--our-numbers-were-measuring-the-wrong-thing) at the bottom for the full investigation.

## Capability Spectrum

How model size correlates with benchmark capability across our test suite:

| Active Params | Best Score | Capable Of |
|---|---|---|
| **2.3B** (Gemma E4B) | 5/22 (23%) | String manipulation only |
| **8B** (Qwen3-8B) | 14/17 (82%) | Medium coding, some hard tasks |
| **~4B active / 26B MoE** (Gemma 26B-A4B) | 17/17 (100%) | All benchmarks, fastest S-tier |
| **27B dense** (Harmonic, Qwopus, Opus-distilled) | 16-17/17 (94-100%) | All benchmarks, varies by fine-tune |
| **31B dense** (Gemma 31B) | 17/17 (100%) | All benchmarks, most consistent |
| **35B MoE** (Qwen 3.5 35B-A3B) | 11/17 (65%) | Easy/medium only — fails LRU consistently |

**Key insight:** Active parameter count matters more than total parameters. Gemma 26B-A4B (~4B active, MoE) ties Gemma 31B-IT (31B dense, every param active) on quality while being 3x faster.

## S-Tier: Reliable Excellence

### Gemma 4 26B-A4B Q6_K
The most consistent model tested. Average score nearly equals best-of-3 — produces the same quality every run.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Best-of-3 (temp 0.3) | 30/31 (97%) |
| Average (temp 0.3) | 30.0/31 (97%) |
| Throughput | **138.9 tok/s** |
| TTFT | **61 ms** |
| VRAM (32K ctx) | 26,739 MB |
| Max context (turbo4) | ~230K |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Fastest S-tier, extremely consistent, reaches near-full context window with TurboQuant.
**Weakness:** ExprEval caps at 4/5 at temp 0.3 (test wording mismatch, not implementation quality).

---

### Gemma 4 31B-IT Q4_K_M
Highest consistency of any model. Perfect single-shot, 99% average. **Cannot run without TurboQuant** — only 16K context with f16 KV.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Best-of-3 (temp 0.3) | **31/31 (100%)** |
| Average (temp 0.3) | **30.7/31 (99%)** |
| Throughput | **50.3 tok/s** |
| TTFT | **91 ms** |
| VRAM (32K ctx) | 23,620 MB |
| Max context (turbo4) | ~58K |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Most consistent across all benchmarks and runs. Dense architecture produces very stable output.
**Weakness:** 53 tok/s (dense arch penalty), limited to ~58K context even with turbo4 (870 KB/token KV cache).

---

## A-Tier: Strong With Caveats

### Qwen 3.5 27B Opus-Distilled Q4_K_M
Lowest VRAM of any high-quality model. Peak capability matches S-tier but higher run-to-run variance.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Best-of-3 (temp 0.3) | **31/31 (100%)** |
| Average (temp 0.3) | 25.3/31 (82%) |
| Throughput | **60.0 tok/s** |
| TTFT | **92 ms** |
| VRAM (32K ctx) | 20,530 MB |
| Max context (turbo4) | 262K (full) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Only 20 GB VRAM with full 262K context. A* 6-8/6 (always passes, often generates bonus tests).
**Weakness:** Impl-Only scores vary wildly (1/5, 1/5, 5/5). Best-of-3 flatters it.

---

### Gemma 4 26B-A4B Q4_K_M
Fastest model with 94%+ quality. The Q4_K_M tradeoff: 10% faster, 10% less VRAM, one persistent test failure.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Throughput | **149.5 tok/s** |
| TTFT | **58 ms** |
| VRAM (32K ctx) | 21,194 MB |
| Max context (turbo4) | 262K (full) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Fastest high-quality model. Full 262K context at low VRAM.
**Weakness:** Q4_K_M + turbo4 is the documented risky combination (turboquant_plus). One LRU lazy-cleanup edge case that Q6_K doesn't have.

---

## A-Tier (with thinking): Reasoning Fine-Tunes

### Harmonic 27B Q4_K_M (thinking ON)
With thinking enabled, Harmonic becomes the most consistent model tested. **30.7/31 average** — near-perfect on every benchmark, every run. Designed for structured reasoning (self-correction, verification, multi-path exploration).

| Metric | Value |
|---|---|
| Best-of-3 (temp 0.3, thinking on) | **31/31 (100%)** |
| Average (temp 0.3, thinking on) | **30.7/31 (99%)** |
| Best-of-3 (temp 0.3, thinking off) | 31/31 (100%) |
| Average (temp 0.3, thinking off) | 27.0/31 (87%) |
| Throughput (thinking off) | **61.3 tok/s** |
| TTFT | **96 ms** |
| VRAM (32K ctx) | 20,533 MB |
| Max context (turbo4) | 262K (full) |
| Config | `-ctk turbo4 -ctv turbo4 -rea on --reasoning-budget 16384` |

**Thinking ON is decisively better for this model** (+3.7 avg). LRU Cache: 6/6 every run with thinking, 4.0 avg without.

**Note:** Q8_0 version scores identically at 31 GB VRAM. Always pick Q4_K_M — same quality, half the VRAM.

---

### Qwopus 3.5 27B-v3 Q6_K
Embeds reasoning in content output even with thinking off. Produces verbose, thoughtful code — but inconsistently.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Best-of-3 (temp 0.3) | 30/31 (97%) |
| Average (temp 0.3) | 24.0/31 (77%) |
| Throughput | **49.6 tok/s** |
| TTFT | **96 ms** |
| VRAM (32K ctx) | 25,505 MB |
| Max context (turbo4) | 262K (full) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Single-shot score is high (94%). Debug Fix 4/4 every run.
**Weakness:** LRU Cache varies from 5/6 to 0/6 across runs at temp 0.3. Lowest consistency of viable models.

---

## B-Tier: Reasoning Fine-Tune Variants

### Gemma 31B Opus-Distilled Q4_K_M (TeichAI)
Fine-tuned on Claude Opus reasoning data. Slightly worse than the base model on coding — fine-tuning didn't help and lost one A* test.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Throughput | **50.9 tok/s** |
| TTFT | **88 ms** |
| VRAM (32K ctx) | 23,625 MB |
| Max context (turbo4) | ~58K |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Verdict:** Base Gemma 31B-IT (17/17) beats the Opus-distilled version (16/17). Reasoning distillation helps Qwen models (+7 tests on 27B) but slightly hurts Gemma 31B which already had strong coding capability baked in.

---

## C-Tier: Niche Use

### Qwen 3.5 35B-A3B Q4_K_M
Fastest model by far. Use when speed > correctness.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **11/17 (65%)** |
| Best-of-3 (temp 0.3) | 25/31 (81%) |
| Throughput | **174.4 tok/s** |
| TTFT | **75 ms** |
| VRAM (32K ctx) | 24,842 MB |
| Max context (turbo4) | 262K (full) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** 3x faster than Gemma 31B. String Processor and Debug Fix always pass.
**Weakness:** LRU Cache 0/6 on every run in every config. Genuine capability gap on hard data structures. Also: 19/19 (100%) on LM Studio but regresses on llama-server — highly sensitive to inference infrastructure.

---

### Qwen 3.5 27B Q6_K (base model)
The fine-tuned versions (Opus-distilled, Harmonic, Qwopus) all dramatically outperform the base model.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **10/17 (59%)** |
| Throughput | **49.6 tok/s** |
| TTFT | **104 ms** |
| VRAM (32K ctx) | 25,644 MB |

**Bottom line:** No reason to use when fine-tunes exist at the same VRAM cost.

---

## F-Tier: Below Capability Floor

### Gemma 4 E4B Q8_0 (2.3B active params)
Per-Layer Embeddings (PLE) architecture, 2.3B active / 5.1B total parameters. Establishes the lower bound of our benchmark suite.

| Metric | Value |
|---|---|
| Score (4 benchmarks) | **5/22 (23%)** |
| Throughput | 131 tok/s |
| VRAM (32K ctx) | 12,526 MB (f16) / 12,108 MB (turbo4) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Passes String Processor 5/5 (often generates 9-13 bonus tests). Confirms basic coding ability.
**Weakness:** 0/5 on Expression Evaluator, 0/6 on A*, 0/6 on LRU. Too small for medium/hard tasks. turbo4 doesn't change this — same 5/22 on both KV configs.

**Capability floor finding:** Need ~8B+ active parameters for medium coding tasks, ~27B+ for hard data-structure tasks.

---

## Thinking ON vs OFF: It's Model-Dependent

We ran a controlled head-to-head: same models, same benchmarks, temp 0.3, 3 runs each, turbo4 KV. Thinking ON used `--reasoning-budget 16384`.

| Model | ON avg | OFF avg | Winner | Why |
|---|---|---|---|---|
| Harmonic 27B Q4_K_M | **30.7/31** | 27.0/31 | **ON (+3.7)** | Designed for reasoning. LRU 6.0 vs 4.0 avg. |
| Qwopus 27B Q6_K | **29.7/31** | 26.4/31 | **ON (+3.3)** | Thinks regardless; structured ON is better. |
| Gemma 31B Q4_K_M | 29.3/31 | **30.4/31** | **OFF (+1.1)** | Efficient thinker but slightly more consistent without. |
| Gemma 26B Q6_K | 26.7/31 | **29.0/31** | **OFF (+2.3)** | Thinking causes 2 truncations (A*, LRU hit budget). |
| Opus-Distilled 27B | **22.6/31** | 21.7/31 | **ON (+0.9)** | Marginal. Both inconsistent. |

**Key findings:**
- **Reasoning fine-tunes** (Harmonic, Qwopus) benefit from thinking ON — it's what they were trained for
- **Gemma models** do better with thinking OFF under turbo4 KV — the turbo4 quantization noise occasionally disrupts the thinking→content transition, causing truncation
- **The difference is about consistency, not capability** — best-of-3 is often tied; the gap is in average scores

## Cross-Engine Comparison: Same Model, Different Inference Stacks

We tested Gemma 4 31B-IT and Qwen3-8B across multiple inference engines and KV compression strategies to isolate what comes from the model vs the engine vs the compression.

### Gemma 4 31B-IT (3-benchmark suite)

| Engine | Quant | KV Compression | Score | Tok/s | VRAM | Notes |
|---|---|---|---|---|---|---|
| **llama.cpp** | Q4_K_M | turbo4 (3.8x) | **17/17 (100%)** | **53** | **22.3 GB** | Our standard pipeline |
| vLLM 0.19+cu130 | NVFP4-turbo | FP8 KV (2x) | 16/17 (94%) | 41 | 29.6 GB | Blackwell-only, harder setup |

**Verdict:** llama.cpp wins on quality, throughput, AND VRAM efficiency for single-stream coding workloads. NVFP4-turbo's published advantage is **batched/concurrent throughput** (1,244 tok/s with multiple concurrent users) — we don't measure that.

### Qwen3-8B (3-benchmark suite)

| Engine | Quant | KV Compression | Best-of-3 | Notes |
|---|---|---|---|---|
| **vLLM + TriAttention** | BF16 | Token eviction (4096 budget) | **14/17 (82%)** | TriAttention monkeypatches needed for vLLM 0.19 to work on Blackwell at all |
| llama.cpp | Q4_K_M | f16 (baseline) | 12/17 (71%) | |
| llama.cpp | Q4_K_M | turbo4 (3.8x) | 8/17 (47%) | turbo4 hurts 8B models more than 27B+ |

**Verdict on small models:** Token eviction (TriAttention) preserves more quality than weight-precision quantization (turbo4) on 8B models. The pattern flips on 27B+ where turbo4 is essentially free.

### Key Insight

**Compression strategy effectiveness scales with model size.** Larger models have more redundancy to absorb quantization noise. Smaller models are better served by selective retention (TriAttention) than uniform compression (TurboQuant).

## LM Studio Baselines (f16 KV, temp 0, no TurboQuant)

For reference — these were tested before TurboQuant and use LM Studio's default f16 KV cache.

| Model | Quant | Tok/s | Expr Eval (5) | A* Path (6+) | LRU Cache (6) | Total |
|---|---|---|---|---|---|---|
| qwen3.5-35b-a3b | Q4_K_M | 92.9 | 5/5 | 8/8 | 6/6 | **19/19 (100%)** |
| gemma-4-26b-a4b | Q6_K | 161.9 | 5/5 | 6/6 | 5/6 | **16/17 (94%)** |
| qwen3.5-9b | Q8_0 | 113.2 | 5/5 | 6/7 | 4/6 | **15/18 (83%)** |
| nemotron-3-nano | Q4_K_M | 78.4 | 0/5 | 3/7 | 0/6 | **3/18 (17%)** |

---

## Quick Reference: Choosing a Model

| Priority | Pick | Thinking | Why |
|---|---|---|---|
| **Best quality** | Gemma 26B Q6_K | off | 100% single-shot, 97% avg, fastest S-tier |
| **Best consistency** | Harmonic 27B Q4_K_M | on | 30.7/31 avg (99%), 20 GB VRAM |
| **Best consistency (fast)** | Gemma 31B Q4_K_M | off | 30.4/31 avg, requires TurboQuant |
| **Lowest VRAM** | Opus-Distilled 27B Q4_K_M | on | 20 GB, 100% single-shot, full 262K ctx |
| **Fastest** | Qwen 35B Q4_K_M | off | 188 tok/s, 65% quality |
| **Best value** | Harmonic 27B Q4_K_M | on | 20 GB, 99% avg, best all-rounder |

## A Note on TTFT — *Our Numbers Were Measuring the Wrong Thing*

The TTFT values above are **all wrong** in a specific, calibrated way: they reflect Python `requests` library overhead on Windows, not actual model performance. The real per-model TTFT is ~30x lower than what we reported.

### How we found out

We ran a four-cell isolation test using the same Gemma 26B Q6_K model, the same TurboQuant fork built from the same git commit, the same turbo4 KV config — varying only the OS where the server and client ran:

| Server OS | Client OS | Mean TTFT | Decode |
|---|---|---|---|
| Windows | Windows | **2.158s** | 113 tok/s |
| WSL2 (Linux) | WSL2 (Linux) | **0.062s** | 140 tok/s |
| Windows | WSL2 (Linux) | **0.068s** | 144 tok/s |
| WSL2 (Linux) | Windows | **2.104s** | 129 tok/s |

The pattern is unambiguous: **the client OS is the only variable that matters, the server OS makes ~zero difference.** Same Windows server hit from a Linux client returns in 68ms. Same Linux server hit from a Windows client takes 2.1s.

### Drilling further into the Python stack on Windows

Same machine, same server, varying only the HTTP client layer:

| Stack layer | TTFT |
|---|---|
| `socket.socket()` (raw) | **67 ms** |
| `http.client.HTTPConnection` (Python stdlib) | 171 ms |
| `urllib3.PoolManager` | **2,097 ms** |
| `requests.post()` | 2,113 ms |
| `requests.Session().post()` | 2,106 ms |

The ~1.9-second overhead is added **between `http.client` and `urllib3`** on Windows. `requests` inherits it from `urllib3`. `requests.Session()` doesn't help because the slow code path runs per-request, not per-connection.

**Most likely cause:** `urllib3` on Windows does eager SSL context initialization (`ssl.create_default_context()`) and Windows root certificate store enumeration on the first HTTP call — even for plain `http://` connections. This is a known slow path on Python/Windows. Linux's OpenSSL cert loading doesn't have this delay.

### Re-measurement: corrected TTFT and decode for all models

We re-ran throughput measurements for all 9 ranked models from inside WSL2 (Linux client + Linux-built llama-server, same TurboQuant source as the Windows build). The numbers in the per-model cards above have been updated to reflect these clean measurements.

| Model | OLD TTFT | **NEW TTFT** | OLD tok/s | **NEW tok/s** |
|---|---|---|---|---|
| Gemma 26B-A4B Q6_K | 2,320 ms | **61 ms** | 142.2 | 138.9 |
| Gemma 26B-A4B Q4_K_M | 2,310 ms | **58 ms** | 156.1 | 149.5 |
| Gemma 31B-IT Q4_K_M | 2,200 ms | **91 ms** | 53.3 | 50.3 |
| Gemma 31B Opus-Distill | 2,190 ms | **88 ms** | 46.8 | 50.9 |
| Qwopus 27B Q6_K | 2,380 ms | **96 ms** | 51.7 | 49.6 |
| Harmonic 27B Q4_K_M | 2,230 ms | **96 ms** | 65.5 | 61.3 |
| Qwen 27B Opus-Distill | 2,140 ms | **92 ms** | 63.8 | 60.0 |
| Qwen 27B Q6_K (base) | 2,150 ms | **104 ms** | 51.3 | 49.6 |
| Qwen 35B-A3B Q4_K_M | 2,310 ms | **75 ms** | 188.1 | 174.4 |

**Key observations from the re-benchmark:**

1. **TTFT was wrong by 25-40x.** Real values are 58-104ms. The old "every model is 2.3s" cluster was 100% client overhead.
2. **Real TTFT scales with model size, slightly.** Smaller models (Gemma 26B at ~60ms) are faster than larger ones (Qwen 27B base at 104ms). The differences are tiny but visible — masked entirely by the old constant overhead.
3. **Decode throughput was actually *overstated* by ~5%, not understated.** Opposite of my initial guess. The Windows Python pipeline was buffering tokens in larger chunks, making the per-iteration timing look faster than the steady-state rate. Linux gets clean per-token timing, which is slightly slower but more accurate.
4. **Relative rankings are unchanged.** Qwen 35B-A3B is still fastest, Gemma 26B variants still cluster around 140-150 tok/s, dense 27-31B models still 50-60 tok/s.
5. **Gemma 31B Opus-Distill flipped:** previously measured at 47 tok/s (slower than the base 31B at 53), it's now at 51 tok/s (slightly faster than the base at 50). Within noise, but worth noting that the Opus-distilled variant doesn't actually have a throughput penalty.

### Practical implications

- **TTFT ~60-100ms means the 5090 is genuinely fast for short responses.** A "factorial function with one test" response that we said took ~3s actually completes in ~1.4s wall-clock (100ms TTFT + 1.3s decode of ~250 tokens at 175 tok/s on Qwen 35B).
- **Decode rates ~50-175 tok/s depending on architecture** — MoE small-active models (Qwen 35B-A3B, Gemma 26B-A4B) at 140-175 tok/s, dense 27-31B at 50-60 tok/s. Bandwidth math holds up.
- **The Windows Python `requests` bug only affected our measurement, not real users.** Users running the server from any non-Python client (curl, the llama-server web UI, LM Studio) get the real ~70ms TTFT.

### Fixing it for future benchmarks

Three options, in order of effort:

1. **Run benchmark scripts from inside WSL2** against the same Windows server. WSL2 can reach Windows localhost via the NAT bridge (172.20.240.1) once a firewall rule allows it. This is what we used for the re-benchmark.
2. **Use Python `http.client` (stdlib) instead of `requests`** in benchmark scripts. Adds ~100ms TTFT instead of 2,100ms. Easy code change.
3. **Use raw `socket.socket()`** for the absolute lowest overhead (~67ms). More code, more parsing, but matches the cleanest measurement we got.

See [`tools/ttft_isolation_test.py`](../tools/ttft_isolation_test.py), [`tools/ttft_session_test.py`](../tools/ttft_session_test.py), and [`tools/rebench_5090_throughput.py`](../tools/rebench_5090_throughput.py) for the test methodology and re-benchmark script.

## Configuration

```bash
# Gemma models (thinking off — avoids budget truncation under turbo4)
llama-server -m gemma-model.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo4 -ctv turbo4 -np 1 -rea off

# Reasoning fine-tunes: Harmonic, Qwopus, Opus-distilled (thinking on)
llama-server -m harmonic-model.gguf --port 8080 -c 32768 -ngl 99 \
  -fa on -ctk turbo4 -ctv turbo4 -np 1 -rea on --reasoning-budget 16384
```

See [TURBO3_RESULTS_5090.md](TURBO3_RESULTS_5090.md) for full experimental data across 8+ benchmark runs.
See [TURBOQUANT_IMPACT_5090.md](TURBOQUANT_IMPACT_5090.md) for what TurboQuant unlocks on 32 GB VRAM.
