# Local Model Tier List — NVIDIA L40S 46GB

**Platform:** NVIDIA L40S 46 GB VRAM, Rocky Linux 9, datacenter Ada Lovelace (sm_89)
**Inference:** vLLM 0.19.0 (Docker: `vllm/vllm-openai:latest`), CUDA 12.x
**Standard config:** BF16 weights (FP8 for >20B), `max_tokens=16384`, thinking off (`chat_template_kwargs: {"enable_thinking": false}`), `--gpu-memory-utilization 0.90`

Tested April 2026.

Rankings combine single-shot accuracy (temp 0), multi-run consistency (temp 0.3, best-of-3 and average across 3 runs), and practical factors (speed, VRAM fit). This is a preliminary dataset — only 2 of 3 attempted models completed benchmarking. More models will be added as testing continues.

## The L40S's defining constraints

The L40S is a datacenter Ada Lovelace GPU: 46 GB GDDR6 ECC, ~864 GB/s memory bandwidth, no NVLink, single-GPU only. It sits between consumer cards (RTX 5090 at 32 GB, ~1792 GB/s) and high-end datacenter GPUs (A100 80GB, H100 80GB).

**VRAM is the binding constraint for vLLM.** BF16 weights plus vLLM's overhead (CUDA graphs, PagedAttention, KV cache, activations) consume significantly more than raw model weight size. Rough budget:

| Model class | Weight size | vLLM total (est.) | Fits? |
|---|---|---|---|
| 9B dense BF16 | ~19 GB | ~25 GB | Yes |
| 14B dense BF16 | ~27 GB | ~33 GB | Yes |
| 27B dense FP8 | ~14 GB weights, ~27 GB effective | ~43 GB | Marginal — OOM with CUDA graphs |
| 35B MoE FP8 (3B active) | ~18 GB weights | ~35+ GB | Loads but extremely slow init |
| 32B dense FP8 | ~17 GB weights | ~37 GB | Likely yes (untested) |

**Key finding:** The 46 GB boundary is tighter than expected. Qwen3.5-27B-FP8 OOMed even with `--enforce-eager` (no CUDA graphs) and 16K context. The 35B-A3B MoE model timed out during loading after 15 minutes. **Practical ceiling for reliable single-model serving is ~14B dense BF16 or ~25B FP8.**

---

## Summary table

All benchmarks use the 3-benchmark coding suite (Expression Evaluator 5 tests + A* Pathfinding 6 tests + LRU Cache with TTL 6 tests = 17 tests total). Throughput is sustained tok/s on a streaming coding generation task. Runs: 1x temp=0, 2x temp=0.3.

| Tier | Model | Precision | VRAM (est.) | Tok/s | Best | Avg | Note |
|---|---|---|---|---|---|---|---|
| **B** | Ministral-3-14B-Instruct-2512 | BF16 | ~33 GB | 28.0 | **13/17 (76%)** | 8.7/17 | Best overall score. LRU rock-solid, ExprEval fragile at temp>0. |
| **B** | Qwen3.5-9B | BF16 | ~25 GB | **44.2** | 11/17 (65%) | 8.0/17 | 1.6x faster. Fails LRU entirely. |
| **—** | Qwen3.5-35B-A3B | FP8 | ~35+ GB | — | — | — | Startup timeout (15 min). Did not load. |

**Quick-pick guide:**
- **Higher quality:** Ministral-3-14B (13/17 best, consistent LRU)
- **Higher speed:** Qwen3.5-9B (44 tok/s, 1.6x faster, but 0/6 on hardest benchmark)
- **Don't attempt on L40S with vLLM:** Qwen3.5-27B-FP8 (OOM), Qwen3.5-35B-A3B FP8 (timeout), any dense >20B at BF16

---

## B-Tier: Functional but Limited

### Ministral-3-14B-Instruct-2512-BF16 (vLLM)

Best overall score on L40S. The only model to pass any LRU Cache tests. ExprEval is perfect at temp 0 but collapses at temp 0.3 (the generated code has syntax errors from test wording mismatches).

| Metric | Value |
|---|---|
| Single-shot (temp 0) | **13/17 (76%)** |
| Best-of-3 (temp 0.3) | 10/12 (83%) |
| Average (temp 0.3) | 7.0/12 |
| Throughput | **28.0 tok/s** |
| TTFT | 197–586 ms |
| Weights | ~27 GB (BF16) |
| Context | 32768 |
| Backend | vLLM 0.19.0 |
| Config | `--max-model-len 32768 --gpu-memory-utilization 0.90` |

**Per-benchmark breakdown:**

| Benchmark | Expected | Run 1 (t=0) | Run 2 (t=0.3) | Run 3 (t=0.3) | Best | Avg |
|---|---|---|---|---|---|---|
| Expression Evaluator | 5 | **5/5** | 0/5 (err) | 0/5 (err) | 5 | 1.7 |
| A* Pathfinding | 6 | 5/6 | 3/6 | 4/6 | 5 | 4.0 |
| LRU Cache with TTL | 6 | **3/6** | **3/6** | **3/6** | 3 | 3.0 |

**Strengths:** Most consistent LRU Cache scores — 3/6 on every single run. Perfect ExprEval at temp 0. Non-thinking model (no wasted tokens on reasoning chains).
**Weaknesses:** ExprEval drops to 0/5 at temp 0.3 (test file has syntax errors, not implementation quality). ~28 tok/s is slow — roughly half the speed of Qwen3.5-9B. A* Pathfinding is inconsistent (3–5 across runs).

---

### Qwen3.5-9B BF16 (vLLM, thinking OFF)

Fastest model tested. Good at ExprEval and A*, but completely fails LRU Cache — zero passes across all 3 runs.

| Metric | Value |
|---|---|
| Single-shot (temp 0) | 9/17 (53%) |
| Best-of-3 (temp 0.3) | 9/12 (75%) |
| Average (temp 0.3) | 5.3/12 |
| Throughput | **44.2 tok/s** |
| TTFT | 189–233 ms |
| Weights | ~19 GB (BF16) |
| Context | 32768 |
| Backend | vLLM 0.19.0 |
| Config | `--max-model-len 32768 --gpu-memory-utilization 0.90`, thinking OFF via `chat_template_kwargs` |

**Per-benchmark breakdown:**

| Benchmark | Expected | Run 1 (t=0) | Run 2 (t=0.3) | Run 3 (t=0.3) | Best | Avg |
|---|---|---|---|---|---|---|
| Expression Evaluator | 5 | **5/5** | 3/5 | 3/5 | 5 | 3.7 |
| A* Pathfinding | 6 | 4/6 | **6/6** | 3/6 | 6 | 4.3 |
| LRU Cache with TTL | 6 | 0/6 | 0/6 | 0/6 | 0 | 0.0 |

**Strengths:** Fast — 44 tok/s with low TTFT (~200ms). Perfect ExprEval at temp 0. A* can hit 6/6 on a good run. Small memory footprint (~25 GB with vLLM overhead) leaves headroom for longer contexts.
**Weaknesses:** Complete LRU Cache failure (0/6 across all runs) — the model cannot produce correct TTL expiration logic at 9B parameters. A* is inconsistent (3–6 range). ExprEval drops from 5/5 to 3/5 at temp 0.3.

> **Note:** This is the same Qwen3.5 family tested on the 5090 as Qwen3.5-35B-A3B (the MoE variant). The 9B dense version is a different model — smaller, faster, but less capable. The 35B-A3B scored 11/17 on the 5090 with llama.cpp Q6_K; it could not load on L40S via vLLM FP8 within the 15-minute timeout.

---

## Models That Could Not Run

### Qwen3.5-35B-A3B FP8 — Startup Timeout

35B total parameters (3B active) MoE model with FP8 online quantization. The model was cached on local NVMe storage but vLLM did not complete initialization within 15 minutes. The GPU showed only 3.1 GB allocated after 7+ minutes — suggesting the FP8 weight conversion or MoE expert mapping was stalling. This model may work with a longer timeout or a pre-quantized FP8 checkpoint.

### Qwen3.5-27B-FP8 — Out of Memory

27B dense hybrid (Mamba + attention) model. At FP8, the weights fit but vLLM's CUDA graph capture exhausted the remaining VRAM. Even with `--enforce-eager` (disabling CUDA graphs) and context reduced to 16K, the model used 42.45 GB of the 44.39 GB available, leaving insufficient room for KV cache. The 27B dense architecture is simply too large for 46 GB with vLLM's overhead.

---

## Cross-platform comparison

Same-family models across hardware platforms (all thinking OFF):

| Model | Platform | Engine | Tok/s | Best Score |
|---|---|---|---|---|
| Qwen3.5-35B-A3B Q6_K | RTX 5090 32GB | llama.cpp (TurboQuant) | 60.1 | 11/17 |
| Qwen3.5-35B-A3B FP8 | L40S 46GB | vLLM | — | Did not load |
| Qwen3.5-9B BF16 | L40S 46GB | vLLM | 44.2 | 11/17 |

The 5090 can run the 35B MoE at Q6_K via llama.cpp because TurboQuant KV compression fits the MoE within 32 GB. vLLM on the L40S cannot — FP8 online quantization is slower to initialize and the MoE expert mapping adds overhead. The 9B dense model on L40S matches the 35B-A3B's best score (11/17) but at different benchmarks — the 9B passes A* more reliably but fails LRU entirely.

---

## Configuration

```bash
# Run all models
python tools/vllm_l40s_bench.py --ssh-host root@miami --api-host 10.0.5.107

# Run specific model
python tools/vllm_l40s_bench.py --models qwen35-9b --api-host 10.0.5.107

# Skip Docker (container already running)
python tools/vllm_l40s_bench.py --models qwen35-9b --skip-docker --api-host 10.0.5.107
```

**See also:** [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) (RTX 5090 32GB), [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) (DGX Spark GB10 128GB)
