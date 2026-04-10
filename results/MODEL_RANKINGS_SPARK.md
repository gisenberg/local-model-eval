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

## S/A/B/C-Tier

*Pending benchmark results for the MoE models that this hardware actually suits:*
- Qwen3.5-122B-A10B Q4_K_M (122B/10B active, hybrid DeltaNet)
- Qwen3-Coder-Next Q4_K_M (80B/3B active, hybrid DeltaNet, coding specialist)
- MiniMax-M2.5 UD-Q3_K_XL (230B/10B active, Lightning Attention)

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
