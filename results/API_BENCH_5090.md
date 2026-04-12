# API Model Benchmarks — Free Tier Coding Suite

**Date:** 2026-04-12
**Client:** Windows 11, Python 3.14 `requests` → OpenRouter / NVIDIA APIs
**Suite:** 4-benchmark coding suite (Expression Evaluator + A* Pathfinding + LRU Cache with TTL + String Processor, 22 tests total), temp 0.3, 3 runs best-of-3. Same prompts and scoring as [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md).
**Methodology note:** Throughput numbers reflect API + network latency, not model-intrinsic decode speed. Quality scores (pytest pass/fail) are directly comparable to our local model rankings.

## GPT-OSS 120B (OpenRouter Free)

OpenAI's open-source 120B-parameter model served via OpenRouter's free tier. Model ID: `openai/gpt-oss-120b:free`. Context window: 131K. Max output: 16,384 tokens.

### Results

| Benchmark | Best | Avg | API tok/s |
|---|---:|---:|---:|
| Expression Evaluator | **5/5** | 3.3/5 | 34.8 |
| A* Pathfinding | **6/6** | 5.7/6 | 29.4 |
| LRU Cache with TTL | **0/6** | 0.0/6 | 34.1 |
| String Processor | **5/5** | 5.0/5 | 37.0 |
| **Total** | **16/22 (73%)** | **14.0/22 (64%)** | **~34** |

### Analysis

**Three of four benchmarks are solid.** Expression Evaluator 5/5, A* 6/6 (with bonus tests — some runs generated 7-9 passing tests), String Processor 5/5 consistently. Quality on these three tasks is on par with our local A-tier models (Qwen 27B Opus-Distilled, Qwopus 27B).

**LRU Cache is a complete failure — 0/6 on every run.** The failure mode is a **SyntaxError**, not a logic bug: runs 1 and 2 use TypeScript non-null assertion syntax (`self._head.next!.prev`) inside Python code, and run 3 has an unterminated triple-quoted string. The model confuses TypeScript and Python syntax specifically on doubly-linked-list pointer manipulation — a cross-language contamination issue. This is the exact same "LRU 0/6 every run" capability gap we documented for [Qwen 3.5 35B-A3B on the 5090](MODEL_RANKINGS_5090.md) (C-tier), though the failure mode there was different (import errors and logic bugs, not syntax contamination).

**API throughput is ~34 tok/s** at temp 0.3 through OpenRouter's free tier. This is network-dependent (1.5-5s TTFT depending on queue) and not comparable to local decode rates. The free tier had no queue delays or rate limit errors during our test window.

### Where GPT-OSS 120B sits in the 5090 tier list

| Tier | Models at this tier (5090 local) | Score |
|---|---|---:|
| S | Gemma 4 26B-A4B Q6_K, Gemma 4 31B-IT Q4_K_M | 17/17 (100%) |
| A | Qwen 27B Opus, Gemma 26B Q4_K_M, Harmonic 27B, Qwopus 27B | 16-17/17 (94-100%) |
| → | **GPT-OSS 120B (free API)** | **16/22 (73%)** |
| B | Gemma 31B Opus-Distilled | 16/17 (94%) |
| C | Qwen 35B-A3B (same LRU gap) | 11/17 (65%) |

GPT-OSS 120B's 16/22 = 73% places it **between A-tier and C-tier** on quality. It passes 3 of 4 benchmarks cleanly but the LRU failure is categorical — not a variance issue that more runs would fix, but a consistent syntax-contamination bug. The 4-benchmark suite penalizes this more than the original 3-benchmark suite would have (16/22 vs what would have been ~16/17 on the old suite that lacked String Processor).

For comparison, the closest local equivalent is **Qwen 3.5 35B-A3B Q4_K_M** (C-tier, 11/17 = 65%) which has the same LRU gap but also fails on some Expression Evaluator runs. GPT-OSS 120B is noticeably better than that: it passes ExprEval and A* reliably where Qwen 35B doesn't.

**Verdict:** A free API model that matches local A-tier on 3 of 4 coding tasks but has a hard capability ceiling on complex data structures (LRU Cache). Useful for quick coding assistance; not reliable for tasks requiring doubly-linked-list implementations or complex pointer manipulation. The TypeScript syntax contamination suggests the model's training data mixed Python and TypeScript code paths in a way that bleeds through on certain patterns.

## MiniMax M2.5 (OpenRouter Free)

*Benchmark in progress. Preliminary finding: ~8-minute queue times per request on the free tier, and the model's 8,192 max output token cap is too short for its verbose thinking+code output style — the first run hit the cap and scored 0/5 on Expression Evaluator because the implementation was cut off before tests were written. Full results will be added when the bench completes.*

## MiniMax M2.7 (NVIDIA)

*Benchmark in progress. The NVIDIA NIM endpoint has ~90-second cold start latency but does respond. The model is a thinking model that emits `<think>` tokens in its content output. Full results will be added when the bench completes.*

## Raw artifacts

- Bench script: [`tools/api_bench.py`](../tools/api_bench.py)
- Results JSON: [`experiments/api_bench/results.json`](../experiments/api_bench/results.json)
- Per-run test files: `experiments/api_bench/{model}_{benchmark}_run{n}_test.py`

## See also

- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — local model tier list (same benchmark suite, same scoring)
- [ROTORQUANT_5090.md](ROTORQUANT_5090.md) — KV compression experiments on local models
