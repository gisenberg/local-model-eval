# SWE-bench Lite — Qwen3.6-27B-FP8 vs prior Qwen3.6 + Gemma runs

Full 300-instance SWE-bench Lite test split via SWE-agent v1.1.0 on the RTX Pro 6000 Blackwell. This run puts the dense `Qwen/Qwen3.6-27B-FP8` (vendor vLLM-native FP8 release) head-to-head with the MoE `Qwen3.6-35B-A3B-Q8_0` runs (stock and Opus-distilled) and the Gemma-4-31B Q8_0 run previously scored on the same harness.

## Headline

| Model | Quant | Serving | Resolved / 300 | % resolved |
|---|---|---|---|---|
| **Qwen3.6-27B dense** | FP8 (vendor) | vLLM 0.19 | **172** | **57.3%** |
| Qwen3.6-35B-A3B | Q8_0 Opus-distill | llama.cpp + llama-swap | 156 | 52.0% |
| Qwen3.6-35B-A3B | Q8_0 stock | llama.cpp + llama-swap | 145 | 48.3% |
| Gemma-4-31B-IT | Q8_0 | llama.cpp + llama-swap | 69 | 23.0% |

The dense 27B at FP8 is **the best Qwen3.6 variant we've tested**: +5.3 pp over the Claude-Opus-distilled 35B-A3B, +9.0 pp over stock 35B-A3B, +34.3 pp over Gemma-4-31B. On percent-of-attempts (resolved / submitted-non-empty), FP8 is 172 / 296 = **58.1%** vs Opus 156 / 284 = **54.9%** and stock 145 / 279 = **52.0%**.

This is a **smaller model winning by >5 pp** against the Opus-distilled MoE that was our prior daily-driver pick. Worth attributing carefully — see [methodology differences](#methodology-differences-between-runs) below; several variables changed at once.

## Run metadata

- 300 instances, `lite` / `test` split, 4 parallel SWE-agent workers
- vLLM 0.19.1 serving `Qwen/Qwen3.6-27B-FP8` on port 8090, `--max-model-len 65536`, `--max-num-seqs 8`, `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`
- Temp 0.6, top-p 0.95, `per_instance_call_limit: 75`, sampling matches `tools/sweagent-rtxpro6000-fp8.yaml`
- Wall clock: **18h 20m** for the batch + **25 min** for `swebench.harness.run_evaluation` (`--max_workers 8`) = ~18.75h total
- Throughput: ~47 tok/s single-stream decode measured independently in the coding bench; ~16 instances/hour through the batch

## Exit status breakdown

| Status | Count | Notes |
|---|---|---|
| `submitted` (clean) | 228 | Model finished, submitted a diff |
| `submitted (exit_cost)` | 53 | Hit 75-call ceiling before converging, still submitted |
| `submitted (exit_context)` | 15 | Hit 64K context window, submitted what it had |
| `submitted (exit_command_timeout)` | 1 | Hit a 30s in-container bash timeout, still submitted |
| `submitted (exit_error)` | 1 | Transient harness/model error, recovered + submitted |
| `exit_command_timeout` (no submit) | 2 | Failed to submit — same failure mode that devastated Gemma |
| **Submitted (any)** | **298 / 300 = 99.3%** | Top of our submission-rate range |
| **Resolved (hidden tests pass)** | **172 / 300 = 57.3%** | Top of our resolution-rate range |

Compared to prior runs:

| | FP8-27B | Opus-35 | Stock-35 | Gemma-4-31B |
|---|---|---|---|---|
| Submission rate | **99.3%** | 94.7% | 93.0% | 67.0% |
| Empty-patch rate | **0.7%** | 5.3% | 7.0% | 33.0% |
| `exit_context` count | 15 | n/a* | n/a* | n/a* |
| `exit_cost` count | 53 | — | — | — |

\* Prior runs used llama.cpp at 1M context (YaRN ×4), so `exit_context` was effectively impossible. The dense 27B's KV footprint (~65 KB/token — 3× the MoE's) caps us at 64K in vLLM before VRAM becomes the bottleneck; see [`NVFP4_QWEN36_27B_RTXPRO6000.md`](NVFP4_QWEN36_27B_RTXPRO6000.md) for the underlying math. The 15 `exit_context` failures represent a 5% ceiling we'd recover if this model had more context headroom — the dense-27B FP8 at 128K+ context would likely push past 60% resolved.

## Per-repo breakdown

| Repo (n=total in bucket) | FP8-27B | Opus-35B | Stock-35B | Gemma-31B |
|---|---|---|---|---|
| astropy (n=6) | 3 | 3 | 3 | 2 |
| django (n=112†) | **75** | 67 | 63 | 26 |
| matplotlib (n=23) | 11 | 11 | 12 | 5 |
| mwaskom (n=4) | **3** | 2 | 2 | 3 |
| pallets (n=3) | **1** | 1 | 0 | 0 |
| psf/requests (n=6) | **6** | 3 | 5 | 4 |
| pydata/xarray (n=5) | 2 | 2 | 2 | 1 |
| pylint-dev (n=6) | 2 | 3 | 3 | 0 |
| pytest-dev (n=17) | **9** | 8 | 7 | 3 |
| scikit-learn (n=23) | **16** | 16 | 12 | 8 |
| sphinx-doc (n=16) | 2 | 3 | 2 | 1 |
| sympy (n=75) | **42** | 37 | 34 | 16 |
| **Total resolved / 300** | **172** | 156 | 145 | 69 |

† Completed counts differ slightly across runs because a few instances time out at the docker level on one run and not another; the `n=` shown here is for FP8-27B.

FP8-27B wins or ties its category on **8 of 12 repos** (django, mwaskom, pallets, psf, pytest-dev, scikit-learn, sympy, astropy-tie). It loses by ≤1 instance on matplotlib, pylint-dev, pydata, and sphinx-doc. The biggest absolute gains over the Opus-distilled baseline are django (+8), sympy (+5), and psf (+3).

## Methodology differences between runs

Before concluding "dense 27B is just better", four variables changed simultaneously vs the prior Qwen3.6 runs:

| Variable | FP8-27B run | Opus / Stock 35B-A3B runs |
|---|---|---|
| **Model** | Qwen3.6-27B dense (hybrid attention) | Qwen3.6-35B-A3B MoE |
| **Quantization** | FP8 per-channel (vendor release) | Q8_0 block INT8 (unsloth GGUF) |
| **Serving stack** | vLLM 0.19.1 + flashinfer | llama.cpp + llama-swap |
| **Context** | 64K (VRAM-limited, not run-limited) | 1M (YaRN ×4, -np 1) |

Several of these favor FP8-27B:

1. **Dense-27B vs MoE-35B**: dense models typically generalize better at bench-taking than router-gated MoEs when the MoE has a small fraction active per token. Could account for a few points.
2. **FP8 vs Q8_0**: FP8 per-channel is generally higher-quality than block INT8 at the same byte budget, and on Blackwell the native FP8 tensor cores help. The vendor release is also much more aggressive about which layers get quantized (80% of params vs ~30% on our NVFP4 comparison), so the floor is probably lower quant error across attention too.
3. **vLLM vs llama.cpp**: vLLM's continuous batching and Qwen3.6 function-calling via `qwen3_xml` parser are mature; llama.cpp's `--jinja` + Hermes-style tool parsing is newer. We've had tool-call-format quirks historically.
4. **64K vs 1M context**: this one works **against** FP8-27B (15 `exit_context` failures). The 5% ceiling from context-limited runs suggests the true dense-27B ceiling is higher still if served with more context headroom.

Isolating the "model" variable from "quant + serving + context" would take either `Qwen3.6-27B-Q8_0` via llama-swap (same stack as the prior runs) or `Qwen3.6-35B-A3B-FP8` via vLLM (same stack as this run). Neither was part of this comparison. A Qwen3.6-27B-Q8_0 alias is now in `opencode-config/hosts/rtxpro6000/llama-swap.yaml` in case you want the llama.cpp apples-to-apples run later.

## Recommendation shift

Prior daily-driver for opencode was `rtxpro6000/qwen36-opus-distill-q8` (52.0% SWE-bench Lite, agentic-uplift over stock). On this measurement, **`Qwen3.6-27B-FP8` via vLLM is +5.3 pp better** with a much smaller memory footprint (29 GB vs 35 GB weights). Trade-offs to keep in mind:

- **Context ceiling is tighter**: 64K on vLLM vs 1M on llama.cpp (dense 27B KV is 3× the MoE's per-token cost).
- **Latency profile is different**: FP8 decodes at ~47 tok/s single-stream; the 35B-A3B via llama-swap runs ~90 tok/s single-stream. Batching catches some of this back for multi-agent workloads.
- **vLLM state cost**: startup is 30-60s; llama-swap with llama.cpp swap cost is similar. Not a regression, but not a win either.

For pure SWE-bench-style edit-loop quality, FP8-27B is the new best option. For long-context whole-repo reads (>64K), the 35B-A3B Opus-distilled alias remains useful.

## Files

- `experiments/sweagent_lite_fp8/` — 300 trajectories, `preds.json`, `run_batch.config.yaml`, `run_batch_exit_statuses.yaml`
- `experiments/sweagent_lite_fp8/eval/sweagent_lite_fp8.qwen36-27b-fp8-full.json` — harness evaluation report (resolved / unresolved / error ids)
- `tools/sweagent-rtxpro6000-fp8.yaml` — SWE-agent config used for the batch
- `logs/sweagent_fp8.log`, `logs/swebench_eval_fp8.log`, `logs/vllm_fp8_swebench.log` — run logs (not committed)
