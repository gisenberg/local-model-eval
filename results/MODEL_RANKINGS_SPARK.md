# Local Model Tier List — DGX Spark GB10 128GB

**Platform:** NVIDIA DGX Spark, GB10 (Grace Blackwell), ~120 GB usable unified memory
**Memory bandwidth:** ~273 GB/s LPDDR5X (the dominant constraint for this platform)
**Inference:** llama.cpp standard build, CUDA 13.0, aarch64 Linux
**Standard config:** flash attention (`-fa on`), `max_tokens=16384`, f16 KV (unless noted), thinking off (`-rea off`), `--no-mmap --jinja -np 1`

Tested April 2026.

## The Spark's defining constraint: memory bandwidth

This is the most important thing to understand about the Spark. **It is bandwidth-bound, not capacity-bound.** With 120GB of unified memory you can *load* enormous models, but with only ~273 GB/s of memory bandwidth (vs. ~1792 GB/s on an RTX 5090), each token's autoregressive decoding must read all active weights through a comparatively narrow pipe.

The implication: **dense >20B models are not viable for interactive use.** A dense model reads its entire weight set per token; on the Spark this caps you near the bandwidth ceiling regardless of how much VRAM you have spare.

| Model class | Bytes/token (Q4) | Spark theoretical max | Practical |
|---|---|---|---|
| Dense 31B Q8 | ~33 GB | 8 tok/s | **~7 tok/s** (measured) |
| Dense 70B Q4 | ~40 GB | 7 tok/s | ~6 tok/s expected |
| MoE 80B/3B active Q4 | ~2 GB | 135 tok/s | ~80–100 tok/s expected |
| MoE 122B/10B active Q4 | ~6 GB | 45 tok/s | ~30–40 tok/s expected |
| MoE 230B/10B active Q3 | ~5 GB | 55 tok/s | ~30–40 tok/s expected |

**Conclusion: this hardware exists to run MoE models with small active parameter counts.** Any other model class wastes the platform.

---

## F-Tier: Bandwidth-bottlenecked

### Gemma 4 31B-IT Q8_0 (dense)
Loads fine, runs at ~6.7 tok/s. Not a model quality issue — this is hardware physics. Use a 5090 or A100 if you need a dense 31B.

| Metric | Value |
|---|---|
| Throughput | **6.7 tok/s** |
| TTFT | 0.4s |
| Weight size | 31 GB |
| Bandwidth utilization | ~221 GB/s = 81% of peak |
| VRAM (32K ctx) | ~58 GB |
| Config | `-ctk f16 -ctv f16 -rea off` |

**Takeaway:** Gemma 31B is one of the highest-quality models in the 5090 rankings but on the Spark it is unusable for any task that involves more than a few hundred tokens of output. A 5,000-character expression evaluator solution takes over 3 minutes to generate. Coding benchmarks were aborted as it would take ~30+ minutes per benchmark and the result is fundamentally a hardware limitation, not a model capability question.

---

## A-Tier: Strong Quality at Interactive Speed

### Qwen3.5-122B-A10B Q4_K_M (unsloth)
122B/10B-active hybrid (DeltaNet linear attention in 36 layers + full attention in 12). The flagship Qwen MoE works exactly as the architecture predicts on Spark — 21 tok/s sustained, with code quality matching A-tier 5090 results despite the much lower bandwidth.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **16/17 (94%)** |
| Throughput (sustained) | **21.0 tok/s** |
| TTFT | 0.56 s |
| Weight size | 72 GB (3-shard GGUF) |
| Bandwidth utilization | ~7.5 GB/token × 21 tok/s ≈ 158 GB/s = 58% of peak |
| VRAM (32K ctx) | TBD (small KV: ~24 KB/token via DeltaNet hybrid) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | 4/5 | 2245 | Test-wording mismatch on error_cases (raises ValueError but with "Invalid token" instead of "Mismatched parentheses") |
| A* Pathfinding | 6/6 (+1 bonus test) | 3280 | Wrote a 7th test beyond the 6 required, all passed |
| LRU Cache with TTL | 6/6 | 2758 | Clean pass on the hardest benchmark |

**Strengths:** Production-quality code on the hardest benchmark (LRU). Stable throughput across all 4 generation runs (21.0–21.2 tok/s, no degradation). Spark's first viable A-tier model.

**Caveats:** ExprEval miss is the same kind of error-categorization quibble that costs other A-tier models a point — the implementation is correct, just doesn't tag the error class the test expects. Single-shot result only; multi-run consistency not yet measured.

---

## B-Tier: Fast but Inconsistent

### Qwen3-Coder-Next UD-Q4_K_M (unsloth dynamic)
80B/3B-active hybrid (12 attention layers of 48, gated DeltaNet for the rest). The "coding specialist" billing didn't deliver — quality lands at 76%, comparable to a mid-B-tier model on the 5090. Speed is the strong story: **50 tok/s** sustained, the fastest large model on the platform so far and within striking distance of a 5090 running dense Gemma 31B.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **13/17 (76%)** |
| Throughput (sustained) | **50.2 tok/s** |
| TTFT | 0.39 s |
| Weight size | 46 GB |
| Bandwidth utilization | ~2 GB/token × 50 tok/s = 100 GB/s = 37% of peak |
| VRAM (32K ctx) | TBD (small KV: ~24 KB/token via DeltaNet hybrid) |
| Config | `-fa on -ctk f16 -ctv f16 -np 1 -rea off --no-mmap --jinja` |

**Per-benchmark breakdown:**
| Benchmark | Pass | Tokens | Notes |
|---|---|---|---|
| Expression Evaluator | 4/5 | 2258 | Same error_cases wording mismatch as Qwen 122B (raises ValueError but with "Invalid token" instead of "Mismatched parentheses") |
| A* Pathfinding | 6/6 | 2705 | Clean. Algorithmic code is solid. |
| LRU Cache with TTL | 3/6 | 2407 | Wrote tests with `mock.patch('ttl_cache.time.monotonic', side_effect=[0,0,...])` but mis-counted how many times its own implementation calls `time.monotonic()`, causing `StopIteration` and wrong values. The impl may be correct; the test harness is broken. |

**Strengths:** Fast (50 tok/s — 2.4× Qwen 122B's speed for comparable bandwidth math, because only 3B params are active per token). Solid on stateless algorithmic code (A* perfect). Tiny weight footprint leaves >70 GB free for other workloads or huge contexts.

**Weakness:** Self-inconsistency in stateful test mocking. The model seems to know what `mock.patch` is, knows its API, but doesn't model the runtime call sequence of its own code carefully enough. This is the "coding specialist" failing on a coding task — possibly because the test harness instructions explicitly required mocked time, which forced the model into a code path it's worse at than its happy path.

**Verdict:** 50 tok/s is genuinely useful — interactive enough to use for coding agents — but you'd want to pair it with a more reliable model for verification on complex stateful logic. Treat it as a "fast first-draft" model rather than a one-shot solver.

---

### Pending

- **MiniMax-M2.5 UD-Q3_K_XL** (230B/10B active, Lightning Attention) — downloading. Expected ~30–40 tok/s based on bandwidth math, similar quality profile to large Qwen MoE.

---

## Configuration

```bash
# Standard Spark config for MoE models
~/llama.cpp/build/bin/llama-server \
  -m model.gguf --port 8080 \
  -c 32768 -ngl 99 \
  -fa on -ctk f16 -ctv f16 \
  -np 1 --temp 0 \
  --no-mmap --jinja -rea off
```

Per [Nauful's advice](../../../.claude/projects/-home-gisenberg-git-gisenberg-local-model-eval/memory/feedback_kv_quantization.md), only quantize V cache (not K) if KV quantization is needed: `-ctk f16 -ctv q8_0`. K cache quantization is reported to break tool calling reliability.

For dense models that exceed bandwidth-viable size, TurboQuant fork is built and available at `~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server`, but TurboQuant only addresses KV cache memory, not weight bandwidth, so it does not solve the dense-model throughput problem on this platform.
