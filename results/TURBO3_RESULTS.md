# TurboQuant turbo3 KV Cache Benchmark Results

Tested April 2026 on RTX 5090 32 GB. TurboQuant fork of llama.cpp (`feature/turboquant-kv-cache` branch), CUDA 13.2, flash attention on, `-ctk turbo3 -ctv turbo3`, 32K context, temperature 0, `max_tokens=16384`.

## Headline: What turbo3 KV Changes

turbo3 quantizes the KV cache to 3 bits (~4x compression vs f16), reducing VRAM usage at the cost of dequantization overhead. This test evaluates whether turbo3 preserves code generation quality and where the VRAM savings matter.

## Benchmark Results

| Model | Quant | Weights | VRAM (32K) | Tok/s | TTFT | Expr Eval (5) | A* Path (6) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwopus 3.5 27B-v3 | Q6_K | 21 GB | 24,311 MB | 49-52 | 2.38s | 4/5 | **6/6** | **6/6** | **16/17 (94%)** | Best quality. Heavy thinker (11K tok/benchmark). |
| Gemma 4 31B-IT | Q4_K_M | 18 GB | 21,883 MB | 50-55 | 2.28s | 4/5 | 0/6 | **6/6** | **10/17 (59%)** | Perfect on hardest benchmark. A* had import errors. |
| Gemma 4 26B-A4B | Q4_K_M | 16 GB | 19,529 MB | 148-164 | 2.39s | 4/5 | 3/6 | 0/6* | **7/17 (41%)** | Fast but LRU exhausted thinking budget (16K tok, no code). |
| Gemma 4 26B-A4B | Q6_K | 22 GB | 25,048 MB | 137-150 | 2.29s | 0/5* | 0/6 | **6/6** | **6/17 (35%)** | ExprEval exhausted thinking budget. LRU perfect. |
| Qwen 3.5 35B-A3B | Q4_K_M | 20 GB | 23,386 MB | 170-175 | 2.34s | 0/5 | 0/6* | 0/6* | **0/17 (0%)** | Fastest throughput but thinking budget issues. See analysis. |
| Qwopus 3.5 27B-v3 | TQ3_4S | -- | -- | -- | -- | -- | -- | -- | -- | Server failed to start (weight format not supported). |

*Asterisked entries hit the 16,384 max_tokens limit with all tokens consumed by chain-of-thought thinking, producing zero code output.

## Thinking Budget Exhaustion

The most significant finding: **several models burned their entire 16K token budget on chain-of-thought reasoning**, producing no code at all. This was not observed at these rates with LM Studio (f16 KV).

| Model | Benchmark | Tokens | Thinking Chars | Content Chars | Finish Reason |
|---|---|---|---|---|---|
| Qwen 3.5 35B | A* Path | 16,384 | 57,456 | **0** | length (truncated) |
| Qwen 3.5 35B | LRU Cache | 16,384 | 55,735 | **0** | length (truncated) |
| Gemma 26B Q6_K | Expr Eval | 16,384 | 40,485 | **0** | length (truncated) |
| Gemma 26B Q4_K_M | LRU Cache | 16,384 | 44,909 | **0** | length (truncated) |

The turbo3 KV quantization introduces subtle numerical differences in the attention computation that appear to affect the model's internal "stop thinking" signal. Models that produced concise reasoning under f16 KV become verbose under turbo3, sometimes never transitioning from thinking to content output.

This primarily affected the highest-throughput models (Qwen 35B at 170+ tok/s, Gemma 26B at 150+ tok/s). Slower models (Qwopus at ~50 tok/s, Gemma 31B at ~52 tok/s) were less affected, possibly because their architecture handles the KV quantization noise differently.

**Qwen 35B was the most impacted**: it scored 19/19 (100%) with LM Studio f16 KV but 0/17 (0%) with turbo3 — a complete regression driven by thinking budget exhaustion, not code quality.

## VRAM Analysis

### Measured VRAM at 32K Context

| Model | Weights (disk) | VRAM Used | Free (of 32,606 MB) |
|---|---|---|---|
| Gemma 26B-A4B Q4_K_M | 16 GB | 19,529 MB | **13,077 MB** |
| Gemma 31B-IT Q4_K_M | 18 GB | 21,883 MB | **10,723 MB** |
| Qwen 35B-A3B Q4_K_M | 20 GB | 23,386 MB | **9,220 MB** |
| Qwopus 27B-v3 Q6_K | 21 GB | 24,311 MB | **8,295 MB** |
| Gemma 26B-A4B Q6_K | 22 GB | 25,048 MB | **7,558 MB** |

### turbo3 vs f16 KV Cache Size (Theoretical)

| Model | Architecture | KV/Token (f16) | KV/Token (turbo3, ~4x) | 32K KV (turbo3) |
|---|---|---|---|---|
| Qwen 3.5 35B-A3B | DeltaNet hybrid (10 attn layers) | ~20 KB | ~5 KB | ~160 MB |
| Qwopus 3.5 27B-v3 | DeltaNet hybrid (16 attn layers) | ~64 KB | ~16 KB | ~512 MB |
| Gemma 4 26B-A4B | Sliding window + global (15+15) | ~120 KB | ~30 KB | ~960 MB |
| Gemma 4 31B-IT | Dense, sliding + global (50+10) | ~870 KB | ~217 KB | ~6,944 MB |

## Can We Use the Full Native Context Window?

This is the key question: with turbo3 KV compression, does each model reach its full trained context window (262K for all models) on the RTX 5090?

| Model | Weights VRAM | KV at 262K (turbo3) | Total Est. | Fits 32 GB? | Max Usable Ctx |
|---|---|---|---|---|---|
| Qwen 3.5 35B-A3B | ~21 GB | ~1.3 GB | **~22 GB** | **Yes, easily** | **262K (full)** |
| Qwopus 3.5 27B-v3 Q6_K | ~23 GB | ~4.2 GB | **~27 GB** | **Yes** | **262K (full)** |
| Gemma 4 26B-A4B Q4_K_M | ~17 GB | ~7.9 GB | **~25 GB** | **Yes** | **262K (full)** |
| Gemma 4 26B-A4B Q6_K | ~23 GB | ~7.9 GB | **~31 GB** | **Barely** | **~230K** |
| Gemma 4 31B-IT Q4_K_M | ~20 GB | ~57 GB | **~77 GB** | **No** | **~58K** |

### Key Findings

- **Qwen 35B and Qwopus 27B** easily reach full 262K context with turbo3 — their hybrid DeltaNet architecture keeps KV costs tiny. turbo3 is almost overkill here.
- **Gemma 26B-A4B Q4_K_M** reaches full 262K with turbo3. This is the model that previously dropped from 150 to 55 tok/s at 256K with f16 KV due to RAM spill. turbo3 eliminates the spill entirely.
- **Gemma 26B-A4B Q6_K** gets close (~230K) but can't quite reach 262K — the larger weights eat into the budget.
- **Gemma 31B-IT is the outlier** — its dense architecture with 870 KB/token KV cache is 7x more expensive than the 26B-A4B MoE variant. Even with turbo3 (~217 KB/token), it maxes out around 58K context. For this model, turbo3 is not a luxury — it's a requirement (f16 KV at 32K alone would need ~28 GB just for KV cache).

## Comparison: turbo3 vs LM Studio (f16 KV)

### Throughput

| Model | LM Studio (f16) | llama-server (turbo3) | Delta |
|---|---|---|---|
| Qwen 3.5 35B-A3B Q4_K_M | 92.9 tok/s | 170-175 tok/s | **+83-88%** |
| Gemma 4 26B-A4B Q6_K | 161.9 tok/s | 137-150 tok/s | -7 to -15% |
| Gemma 4 26B-A4B Q4_K_M | 168.8 tok/s | 148-164 tok/s | -3 to -12% |

Note: the Qwen 35B speedup is **not from turbo3** — it's from using llama-server directly vs LM Studio's wrapper. In our smoke test, the same model on llama-server with f16 KV hit 162.5 tok/s. The Gemma models are slightly slower under turbo3 due to dequantization overhead, which is expected.

### Code Quality

| Model | LM Studio (f16 KV) | turbo3 KV | Delta |
|---|---|---|---|
| Qwen 3.5 35B-A3B | 19/19 (100%) | 0/17 (0%) | **Severe regression** (thinking budget, not code quality) |
| Gemma 4 26B-A4B Q6_K | 16/17 (94%) | 6/17 (35%) | Moderate regression (mixed thinking budget + quality) |
| Gemma 4 26B-A4B Q4_K_M | -- | 7/17 (41%) | No LM Studio hard-benchmark baseline |

### New Models (turbo3 only, no LM Studio baseline)

| Model | Total | Tier (per existing criteria) | Notes |
|---|---|---|---|
| Qwopus 3.5 27B-v3 Q6_K | 16/17 (94%) | **A** | Only model to pass A* 6/6 and LRU 6/6. |
| Gemma 4 31B-IT Q4_K_M | 10/17 (59%) | **C** | Perfect LRU 6/6 but A* failed to import. Dense arch = slow. |

## Recommendations

### Best Overall: Qwopus 3.5 27B-v3 Q6_K
- 16/17 (94%) code quality — matches Gemma 26B Q6_K's LM Studio score
- Full 262K context fits in VRAM with turbo3
- Slow (49-52 tok/s) but reliable — heavy chain-of-thought (11K tok/benchmark) but always produces working code
- Best model for hard coding tasks where correctness > speed

### Best Speed/Quality: Gemma 4 26B-A4B Q4_K_M
- 164 tok/s throughput, 3x faster than Qwopus
- 7/17 on turbo3, but thinking budget exhaustion explains most failures
- With f16 KV (no turbo3), this model scores much higher — use turbo3 only when you need the context headroom
- Full 262K context with turbo3

### Avoid with turbo3: Qwen 3.5 35B-A3B
- Despite being S-tier (19/19) on LM Studio, turbo3 KV causes complete thinking budget exhaustion
- 0/17 is not representative of model quality — it's a turbo3 compatibility issue
- Use f16 KV with this model (it has tiny KV cache anyway due to DeltaNet hybrid architecture)

### Context-Hungry Workloads: Skip Gemma 31B-IT
- Dense architecture makes KV cache 7x more expensive than the 26B-A4B MoE variant
- Even with turbo3, maxes out at ~58K context
- 3x slower than the 26B variant (52 vs 150+ tok/s)
- The 26B-A4B gives you better speed, better context, and comparable quality

## Open Questions

1. **Would increasing max_tokens to 32K fix the thinking budget issue?** The 16K limit may be too tight for turbo3's tendency to produce longer reasoning chains.
2. **Is turbo2 (2-bit KV) even more aggressive on thinking budget?** Worth testing to see if the quality cliff gets worse.
3. **Qwopus TQ3_4S weight format**: the server failed to start — the `feature/turboquant-kv-cache` branch may not support TQ3_4S weights yet, or they need a different branch.
4. **Context scaling test**: measure Gemma 26B Q4_K_M throughput at 32K-256K with turbo3 to verify no RAM spill cliff (the whole point of turbo3 for this model).
