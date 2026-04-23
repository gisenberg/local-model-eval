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

### Gemma 4 26B-A4B Q6_K (llama.cpp + TurboQuant)
The most consistent model tested. Average score nearly equals best-of-3 — produces the same quality every run.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Best-of-3 (temp 0.3) | 30/31 (97%) |
| Average (temp 0.3) | 30.0/31 (97%) |
| Throughput | **138.9 tok/s** |
| TTFT | **61 ms** |
| VRAM (32K ctx) | 26,739 MB |
| Max context (turbo4) | **262K (full native)** |
| Backend | llama.cpp (TurboQuant fork) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Fastest S-tier, extremely consistent, reaches the full 262K native context window with TurboQuant at ~5 GB of VRAM headroom.
**Weakness:** ExprEval caps at 4/5 at temp 0.3 (test wording mismatch, not implementation quality).

> **Correction (2026-04-11):** An earlier version of this card listed "Max context (turbo4) ~230K" based on an architecture table in `CONTEXT_CAPACITY_5090.md` that recorded 15 global-attention layers at head_dim 256. The live gemma4 loader on the rebased johndpope fork shows the correct architecture: **5 global-attention layers at head_dim 512** (plus 25 SWA layers at head_dim 256). Recomputing the per-token KV cost from the live architecture shows turbo4 fits the full 262K native window with ~5 GB of headroom on 32 GB, so rotorquant compression would give only extra VRAM headroom (no additional usable context) on this model. See the addendum in [ROTORQUANT.md](ROTORQUANT.md) for the full correction.

---

### Gemma 4 26B-A4B NVFP4 (vLLM)
Same base model, NVFP4 quantization, served via vLLM. Quality matches the llama.cpp setup at 22/22 (100%) on the expanded 4-benchmark suite. Tested using `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (community variant of the RedHatAI quant that bundles a patched `gemma4.py` for vLLM's NVFP4 MoE expert mapping bug — see [vllm#38912](https://github.com/vllm-project/vllm/issues/38912)).

| Metric | Value |
|---|---|
| Best-of-3 (temp 0.3, 4 benchmarks) | **22/22 (100%)** |
| Average (temp 0.3) | 21.0/22 (95%) |
| Throughput | **130–143 tok/s** (variable per benchmark) |
| Weights | 15.74 GB (NVFP4) |
| VRAM (32K ctx) | **31,311 MB** (~5 GB more than llama.cpp Q6_K) |
| Max context (this config) | ~32K (VRAM-constrained) |
| Backend | vLLM 0.19.0+cu130 |
| Config | `--quantization modelopt --moe-backend marlin --kv-cache-dtype fp8 -e VLLM_NVFP4_GEMM_BACKEND=marlin` |

**Strengths:** Smaller weights (15.7 vs 22 GB) and matches llama.cpp throughput. Useful if you need vLLM's batched serving / OpenAI API features.
**Weaknesses:**
- **Total VRAM is ~5 GB higher than llama.cpp Q6_K** (31.3 vs 26.7 GB) despite the smaller weights — vLLM's CUDA graph cache, PagedAttention page tables, Marlin scratch buffers, and pre-allocated activation memory cost ~13 GB on top of the 16 GB weights.
- **Caps at ~32K context on the 5090** (compared to llama.cpp's ~230K with turbo4 KV) because of the higher compute-buffer overhead, even with FP8 KV cache.
- Setup requires patching vLLM's `gemma4.py` and tracking down orphan EngineCore processes that hold model weights in VRAM after failed launches.
- vLLM detects the 5090 (sm_120) as "no native FP4 support" and falls back to Marlin weight-only FP4 — likely a vLLM detection bug, possibly leaving compute headroom on the table.

**Verdict:** Quality is identical to the llama.cpp setup. For our single-stream coding workload on the 5090, **llama.cpp + turbo4 is still the better choice** — same throughput, ~5 GB less VRAM, ~7x more max context. NVFP4 makes more sense on hardware where llama.cpp's TurboQuant fork doesn't apply (datacenter Blackwell, batched serving) or where you specifically need vLLM's API features.

---

### Gemma 4 31B-IT Q4_K_M (llama.cpp + TurboQuant)
Highest consistency of any model. Perfect single-shot, 99% average. **Cannot run without KV compression** — only 16K context with f16 KV.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **17/17 (100%)** |
| Best-of-3 (temp 0.3) | **31/31 (100%)** |
| Average (temp 0.3) | **30.7/31 (99%)** |
| Throughput | **50.3 tok/s** (32K), **46.4 tok/s** (262K) |
| TTFT | **91 ms** |
| VRAM (32K ctx) | 23,620 MB |
| VRAM (262K ctx) | 28,025 MB |
| Max context (turbo4) | **262K (full native)** |
| Backend | llama.cpp (TurboQuant fork) |
| Config | `-ctk turbo4 -ctv turbo4 -rea off` |

**Strengths:** Most consistent across all benchmarks and runs. Dense architecture produces very stable output. Reaches the full 262K native context window with turbo4 at 28 GB VRAM (4 GB headroom on the 5090).
**Weakness:** 50 tok/s (dense arch penalty vs the 139 tok/s MoE Gemma 26B).

> **Correction (2026-04-11):** An earlier version of this card listed "Max context (turbo4) ~58K" based on a pre-experiment projection of 870 KB/token f16 KV. That projection was ~10× too high because it assumed all 60 layers are full-attention; in reality, Gemma 4 31B-IT has only **10 non-SWA (global attention) layers with ~4 shared-KV heads each** (the other 50 are SWA at a fixed 1536-cell window). The actual per-token turbo4 cost is ~22 KB/token (measured from a VRAM sweep at 96K/160K/262K on the rebased planarquant fork), which puts the full 262K window at only 28 GB total. The iso3/iso3 "long context" subcard that was previously in this tier list has been removed — turbo4 already reaches the full native window, so rotorquant provides zero context advantage on this model. See the addendum in [ROTORQUANT.md](ROTORQUANT.md) for the full story.

---

### Gemma 4 31B-IT NVFP4-turbo (LilaRest, via vLLM)
Same base model, NVFP4 quantization with aggressive attention compression ("turbo" variant). Re-measured from WSL2 client on the 4-benchmark suite — previous (committed) result used a 3-benchmark suite and Windows client.

| Metric | Value |
|---|---|
| Best-of-3 (temp 0.3, 4 benchmarks) | **22/22 (100%)** |
| Average (temp 0.3) | 17.9/22 (81%) |
| Throughput (steady state) | **~42 tok/s** decode |
| Throughput (first runs) | 21-29 tok/s (CUDA graph + torch.compile warmup) |
| Weights | **18.54 GB** (NVFP4-turbo, 68% smaller than BF16) |
| VRAM (16K ctx) | **30,300 MB** (`--gpu-memory-utilization 0.90`) |
| KV cache (FP8) | 9.56 GB available → **~21K tokens** max |
| Max context configured | 16K (used for this run) |
| Max context theoretical | ~21K (KV-cache-limited on 5090) |
| Backend | vLLM 0.19.0+cu130 |
| Config | `--quantization modelopt --dtype bfloat16 --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.90` |

**Strengths:** Smaller weights (18.5 vs 23.6 GB) and perfect best-of-3 on all 4 benchmarks. String Processor passes 5/5 on first run, then drops to 1/5 on runs 2-3 where the model writes 1 combined test function with multiple assertions (code is correct, scoring artifact).

**Weaknesses:**
- **Throughput is ~16% slower than llama.cpp** (42 vs 50 tok/s steady state) due to Marlin fallback kernels.
- **Available context is ~21K vs turbo4's 262K.** The NVFP4 weight savings are more than offset by vLLM's compute buffers + activation pre-allocation. llama.cpp + turbo4 gives you the full native window on the same GPU.
- **Average quality is lower** (17.9/22 = 81%) than the llama.cpp version (30.7/31 = 99%) because of higher run-to-run variance. Best-of-3 looks identical; single-shot is where the gap shows.
- Warmup-sensitive: the first 2-3 requests after server start run at ~half speed due to CUDA graph compilation. Benchmarks that only run a few requests per model will understate NVFP4's steady-state performance.

**Verdict:** For dense 31B on the 5090, llama.cpp + turbo4 still wins on throughput, VRAM, max context, and average-case quality. Turbo4 now reaches **full 262K at 28 GB** (corrected from the earlier wrong ~58K projection), making the VRAM and context gap even wider than previously stated. NVFP4-turbo is useful if you need vLLM features (batched serving, OpenAI API) or are on hardware where TurboQuant doesn't apply (datacenter Blackwell, ARM).

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
| Max context (turbo4) | 262K (full native, same architecture as the base 31B-IT) |
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
| **Best quality** | Gemma 26B Q6_K | off | 100% single-shot, 97% avg, fastest S-tier, full 262K native |
| **Best consistency** | Harmonic 27B Q4_K_M | on | 30.7/31 avg (99%), 20 GB VRAM |
| **Best consistency (dense)** | Gemma 31B Q4_K_M | off | 30.7/31 avg, 50.3 tok/s, full 262K at 28 GB |
| **Lowest VRAM** | Opus-Distilled 27B Q4_K_M | on | 20 GB, 100% single-shot, full 262K ctx |
| **Fastest** | Qwen 35B Q4_K_M | off | 174 tok/s, 65% quality |
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

See [TURBOQUANT.md](TURBOQUANT.md) for the full TurboQuant KV compression story, including the 5090's per-model optimal KV configs, the thinking-budget-exhaustion findings with turbo3, and the context-unlock table. See [ROTORQUANT.md](ROTORQUANT.md) for the newer K-only Givens-rotation KV quantizer (turbo4 remains the 5090 default — rotorquant provides no throughput or context advantage on any model we tested).

---

## Free-tier API model comparison

For calibration against hosted API models on the same 4-benchmark coding suite (temp 0.3, 3 runs best-of-3). Throughput numbers reflect API + network latency, not model-intrinsic decode speed; quality scores (pytest pass/fail) are directly comparable to the local tier list above.

### GPT-OSS 120B (OpenRouter Free)

OpenAI's open-source 120B served via OpenRouter free tier. Model ID: `openai/gpt-oss-120b:free`. Context 131K, max output 16,384.

| Benchmark | Best | Avg | API tok/s |
|---|---:|---:|---:|
| Expression Evaluator | **5/5** | 3.3/5 | 34.8 |
| A* Pathfinding | **6/6** | 5.7/6 | 29.4 |
| LRU Cache with TTL | **0/6** | 0.0/6 | 34.1 |
| String Processor | **5/5** | 5.0/5 | 37.0 |
| **Total** | **16/22 (73%)** | **14.0/22 (64%)** | **~34** |

**Three of four benchmarks are solid.** ExprEval 5/5, A* 6/6 (with bonus tests 7-9 on some runs), String Processor 5/5 — on par with local A-tier models (Qwen 27B Opus-Distilled, Qwopus 27B).

**LRU Cache is a complete failure — 0/6 on every run.** Failure mode is a **SyntaxError**, not a logic bug: runs 1 and 2 use TypeScript non-null assertion syntax (`self._head.next!.prev`) inside Python code, run 3 has an unterminated triple-quoted string. Cross-language contamination on doubly-linked-list pointer manipulation. Same "LRU 0/6 every run" capability gap as Qwen 3.5 35B-A3B on the 5090 (C-tier), though the failure mode there was different (import errors + logic bugs, not syntax contamination).

API throughput is ~34 tok/s through OpenRouter's free tier — network-dependent (1.5-5s TTFT depending on queue), not comparable to local decode rates.

**Where it lands in the 5090 tier list:**

| Tier | Models (5090 local) | Score |
|---|---|---|
| S | Gemma 4 26B-A4B Q6_K, Gemma 4 31B-IT Q4_K_M | 17/17 |
| A | Qwen 27B Opus, Gemma 26B Q4_K_M, Harmonic 27B, Qwopus 27B | 16-17/17 |
| → | **GPT-OSS 120B (free API)** | **16/22 (73%)** |
| B | Gemma 31B Opus-Distilled | 16/17 |
| C | Qwen 35B-A3B (same LRU gap) | 11/17 |

Between A-tier and C-tier on quality: passes 3 of 4 benchmarks cleanly, LRU is categorical (consistent syntax-contamination bug, not variance). Closest local equivalent is Qwen 3.5 35B-A3B Q4_K_M (C-tier, 11/17) which has the same LRU gap but also fails on some ExprEval runs. GPT-OSS 120B is noticeably better — passes ExprEval and A* reliably where Qwen 35B doesn't.

Useful for quick coding assistance; not reliable for tasks requiring doubly-linked-list implementations or complex pointer manipulation.

### MiniMax M2.5 (OpenRouter Free) — not benchmarkable

Model ID: `minimax/minimax-m2.5:free`. Context 196K, max output **8,192 tokens** (free tier cap).

Benchmark abandoned after 2 of 12 runs:

| Run | TTFT | Decode | Tokens | Score | Issue |
|---|---:|---:|---:|---:|---|
| ExprEval run 1 | **492s** (8 min queue) | 50.8 tok/s | 8,192 (hit cap) | 0/5 | Truncated mid-implementation |
| ExprEval run 2 | **1,397s** (23 min queue) | 2.9 tok/s | 5,004 | 0/5 | Truncated, slow decode |

Three compounding problems: free-tier queue times are 8-23 min (full bench would take 4-8 hours), 8K max output truncates the thinking preamble + implementation, decode throughput is inconsistent (50.8 tok/s vs 2.9 tok/s). Not a model-capability judgment — just a free-tier infrastructure issue. May perform well on a paid tier with higher output limits.

### MiniMax M2.7 (NVIDIA NIM Free) — not benchmarkable

Model ID: `minimaxai/minimax-m2.7`. Context 200K, max output 16,384. Served via `integrate.api.nvidia.com`.

Every run timed out at 300s or connection dropped mid-stream. A short "Say one word" (max_tokens=5) succeeded at ~90s, confirming the endpoint is alive — but coding prompts requiring thousands of thinking + code tokens exceed the free-tier time limit. The 16K output cap would be sufficient if the endpoint could stay connected long enough. Pure infrastructure timeout, not a model issue.

### API bench artifacts

- Bench script: [`../tools/api_bench.py`](../tools/api_bench.py)
- Results JSON: [`../experiments/api_bench/results.json`](../experiments/api_bench/results.json)
- Per-run test files: `../experiments/api_bench/{model}_{benchmark}_run{n}_test.py`
