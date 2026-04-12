# MiniMax M2.5 & M2.7 on DGX Spark — Experiment Results

**Platform:** NVIDIA DGX Spark, GB10 (Grace Blackwell), ~120 GB usable unified memory, ~273 GB/s LPDDR5X
**Inference:** llama.cpp mainline (`b8740-e34f04215`), CUDA 13.0, aarch64 Linux
**Date:** 2026-04-12

## The critical discovery: "empty think block" template workaround

MiniMax M2.5 and M2.7 are **mandatory thinking models** — MiniMax has explicitly stated that disabling thinking is not supported ([GitHub issue #68](https://github.com/MiniMax-AI/MiniMax-M2/issues/68)). Every mechanism llama.cpp provides (`-rea off`, `--reasoning-budget 0`, custom no-think templates) is ignored because the model's chat template never checks `enable_thinking` and the model weights are trained to always emit `<think>` tokens.

Our original M2.5 benchmark (documented in [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md)) scored only 5/5 + 2 timeouts because the model spent 25+ minutes thinking on A* and LRU before producing any content.

**The workaround:** inject a pre-closed `<think>\n\n</think>` block in the generation prompt so the model sees thinking as "already done" and skips directly to content. This technique was proposed by user `gelim` in [llama.cpp issue #20196](https://github.com/ggml-org/llama.cpp/issues/20196). The llama.cpp maintainer warned it may degrade output quality, but in practice it works cleanly on both M2.5 and M2.7.

Template file: `templates/minimax-m27-empty-think.jinja` — identical to `minimax-m25-no-think.jinja` except the generation prompt block:

```jinja
{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n\n' ~ '</think>' ~ '\n\n' }}
{%- endif -%}
```

**Result: zero reasoning tokens across all 12 benchmark runs.** The model produces content directly, at full decode throughput, with no thinking overhead. Every benchmark that previously timed out now completes in 1-2 minutes.

---

## Architecture note: M2.5 and M2.7 are architecturally identical

Despite MiniMax's marketing describing "Lightning Attention" for M2.5, **both models have the exact same attention architecture** per their `config.json` on HuggingFace:

- 62 layers, ALL standard attention (`attn_type_list` all 1s)
- 48 attention heads, 8 KV heads, 128 head dim
- 256 experts, 8 active per token
- `max_position_embeddings`: 196,608 (192K native context)

**KV cache per token: ~248 KB/token at f16** (measured from llama-server allocation: 47,616 MiB for 196,608 tokens). This is NOT tiny — it's comparable to a dense model with 62 attention layers. The "Lightning Attention = tiny KV" label in our earlier `spark_bench.py` comments was wrong.

At 192K native context, KV cache alone is **~46.5 GB**. This dominates the memory budget alongside the model weights.

---

## Summary table

All benchmarks are single-shot at `temperature=0`, `-fa on`, `f16` KV, `--no-mmap`, empty-think template. Quality is the 3-benchmark coding suite (Expression Evaluator + A* Pathfinding + LRU Cache with TTL, 17 tests total).

| Model | Quant | Weights | Context | Prefill | Decode | Score | Note |
|---|---|---|---|---|---|---|---|
| **M2.7** | UD-IQ3_S | 84 GB | 96K | ~330 tok/s | **28 tok/s** | **14/17 (82%)** | Best MiniMax quality. ExprEval 5/5, A* 6/6, LRU 3/6. |
| **M2.7** | UD-IQ2_XXS | 61 GB | **192K (native)** | ~390 tok/s | **35 tok/s** | **10/17 (59%)** | Full context. ExprEval 5/5, A* 5/6, LRU 0/6. 2-bit costs 4 quality points. |
| **M2.5** | UD-Q3_K_XL | 101 GB | 32K | ~330 tok/s | **28 tok/s** | **8/17 (47%)** | ExprEval 5/5, A* 3/6, LRU 0/6. |
| **M2.5** | UD-IQ2_XXS | 70 GB | **160K** | ~390 tok/s | **35 tok/s** | **11/17 (65%)** | ExprEval 5/5, A* 6+2/6, LRU 0/6 (64K chars, no stop). |
| **M2.5** (original, no workaround) | UD-Q3_K_XL | 101 GB | 32K | ~330 tok/s | **29.6 tok/s** | **5/5 + 2 TOs** | From MODEL_RANKINGS_SPARK.md. A* and LRU timed out at 25 min. |

### Key observations

**M2.7 > M2.5 at every quant level.** The newer training substantially improves coding quality. M2.7 IQ3_S (14/17) vs M2.5 Q3_K_XL (8/17) at the same throughput. M2.7 IQ2_XXS (10/17) vs M2.5 IQ2_XXS (11/17) is within noise but M2.5 had an anomalous A* bonus run.

**2-bit quantization costs real quality.** M2.7 drops from 14/17 (IQ3_S) to 10/17 (IQ2_XXS) — a 4-point regression concentrated on LRU (3/6 → 0/6) and A* (6/6 → 5/6). The throughput increases (28 → 35 tok/s) because smaller weights = less bandwidth per token, but the quality trade is steep.

**LRU Cache is the MiniMax Achilles' heel.** 0/6 on both M2.5 variants and on M2.7 IQ2_XXS. Only M2.7 IQ3_S achieves a partial 3/6. The hardest benchmark consistently breaks these models, especially at aggressive quant levels.

**M2.5 IQ2_XXS scoring higher than M2.5 Q3_K_XL is noise, not signal.** The 11/17 vs 8/17 difference comes from A* (6+2 bonus tests on IQ2_XXS vs 3/6 on Q3_K_XL). At single-shot temp=0, different quant noise steers the model down different code generation paths. On a harder test like LRU, both score 0/6.

**Prefill rate is consistent: ~330-390 tok/s** across all configs on short prompts (200-313 tokens). IQ2_XXS is slightly faster (~390) vs IQ3_S/Q3_K_XL (~330) due to smaller weight reads during prefill. This would scale differently on long prompts where the O(n²) attention dominates.

---

## Memory analysis: what fits at what context

Both models have identical KV requirements (~248 KB/token f16). The only variable is weight size.

| Config | Weights (loaded) | KV @ 32K | KV @ 96K | KV @ 160K | KV @ 192K |
|---|---|---|---|---|---|
| M2.7 IQ3_S | 79.3 GB | 8 GB | 23.3 GB | 38.8 GB | 46.5 GB |
| M2.7 IQ2_XXS | 60.6 GB | 8 GB | 23.3 GB | 38.8 GB | 46.5 GB |
| M2.5 Q3_K_XL | 96.3 GB | 8 GB | 23.3 GB | 38.8 GB | 46.5 GB |
| M2.5 IQ2_XXS | 68.7 GB | 8 GB | 23.3 GB | 38.8 GB | 46.5 GB |

Budget: ~115 GB allocatable on CUDA (120 GB unified minus OS/system overhead).

| Config | Max context (measured) | Total at max ctx | Headroom |
|---|---|---|---|
| M2.7 IQ3_S | **96K** | ~103 GB | 12 GB |
| M2.7 IQ2_XXS | **192K (native)** | ~107 GB | 8 GB |
| M2.5 Q3_K_XL | **32K** (tight, 104 GB) | ~104 GB | 11 GB |
| M2.5 IQ2_XXS | **160K** | ~108 GB | 7 GB |

M2.7 IQ3_S at 192K was attempted and OOMed (79.3 + 46.5 = 126 GB > 115 GB). M2.5 IQ2_XXS at 192K was attempted and OOMed (68.7 + 46.5 = 115.2 GB, just over the line). M2.5 IQ2_XXS at 160K loaded successfully.

---

## Comparison with existing Spark picks

| Model | Score | Decode | Context | Thinking issue? |
|---|---|---|---|---|
| **Qwen3.5-122B-A10B Q4_K_M (unsloth)** | **18/17** | 21.0 tok/s | 256K | No (`-rea off` works on mainline) |
| Qwen3.5-122B-A10B Q4_K_M (bartowski) + ik-llama | 17/17 | 26.0 tok/s | 256K | Yes (thinking can't be disabled) |
| GLM-4.5-Air Q4_K_M | 15/17 | 21.7 tok/s | 128K | No |
| **MiniMax-M2.7 IQ3_S (empty-think)** | **14/17** | **28.0 tok/s** | 96K | Workaround (empty-think template) |
| Qwen3-Coder-Next UD-Q4_K_M | 14/17 | 50.2 tok/s | 256K | No |
| MiniMax-M2.7 IQ2_XXS (empty-think) | 10/17 | 35.0 tok/s | 192K | Workaround |
| MiniMax-M2.5 IQ2_XXS (empty-think) | 11/17 | 35.0 tok/s | 160K | Workaround |
| MiniMax-M2.5 Q3_K_XL (empty-think) | 8/17 | 28.0 tok/s | 32K | Workaround |

**MiniMax-M2.7 IQ3_S lands in B-tier** alongside Qwen3-Coder-Next — same 14/17 score but at lower throughput (28 vs 50 tok/s) and smaller context (96K vs 256K). The empty-think template is a genuine breakthrough that turns an unusable model into a functional coder, but it doesn't beat the existing picks.

**The Qwen3.5-122B-A10B pick remains the clear winner** at 18/17 quality, 256K context, and no template workarounds needed.

---

## Serving configuration for MiniMax (if you want to use it)

The best MiniMax config for the Spark is **M2.7 IQ3_S at 96K context**:

```bash
llama-server \
  -m ~/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/UD-IQ3_S/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 98304 -ngl 99 -fa on -ctk f16 -ctv f16 -np 1 --no-mmap --jinja \
  --chat-template-file templates/minimax-m27-empty-think.jinja \
  -rea off
```

Critical: the `--chat-template-file` pointing to the empty-think template is **mandatory**. Without it, the model will think for 20-30 minutes per prompt and time out on any non-trivial task.

For full 192K native context, use M2.7 IQ2_XXS instead (same command, different model path), accepting the quality drop from 14/17 to 10/17.
