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

Model ID: `minimax/minimax-m2.5:free`. Context window: 196K. Max output: **8,192 tokens** (free tier cap).

**Benchmark abandoned after 2 of 12 runs.** The OpenRouter free tier for this model is not viable for our coding benchmark:

| Run | TTFT | Decode | Tokens | Score | Issue |
|---|---:|---:|---:|---:|---|
| ExprEval run 1 | **492s** (8 min queue) | 50.8 tok/s | 8,192 (hit cap) | 0/5 | Output truncated mid-implementation |
| ExprEval run 2 | **1,397s** (23 min queue) | 2.9 tok/s | 5,004 | 0/5 | Truncated, also very slow decode |

**Three compounding problems:**

1. **Free tier queue times are extreme** — 8 to 23 minutes per request. At this rate, the full 12-run bench would take 4-8 hours.
2. **The 8,192 max output cap truncates the model's output.** MiniMax M2.5 is a thinking model that emits reasoning tokens before code. With a long thinking preamble + verbose implementation, 8K tokens is not enough to fit a complete implementation + pytest tests. Run 1 hit the cap exactly (8,192 tokens) with the implementation cut off before tests were written.
3. **Decode throughput is inconsistent** — 50.8 tok/s on run 1 but only 2.9 tok/s on run 2, suggesting the free tier infrastructure has variable performance.

**Verdict:** MiniMax M2.5 on OpenRouter Free is not benchmarkable with our methodology. The output cap is the fundamental issue — even with zero queue time, 8K tokens is too short for this model's thinking+code output style on our prompts. The model may perform well on a paid tier with higher output limits, but we can't assess that from the free tier. Scored 0/5 on both runs we completed — not a model capability judgment, just a truncation artifact.

## MiniMax M2.7 (NVIDIA)

Model ID: `minimaxai/minimax-m2.7`. Context window: 200K. Max output: 16,384 tokens. Served via NVIDIA NIM at `integrate.api.nvidia.com`.

**Benchmark abandoned — every run timed out or disconnected.** The NVIDIA free-tier endpoint for this model cannot complete a coding-benchmark request within 300 seconds:

| Run | Result |
|---|---|
| ExprEval runs 1-3 | All 3 timed out at 300s (no response bytes received) |
| A* run 1 | "Response ended prematurely" (connection dropped mid-stream) |

A short "Say one word" test (max_tokens=5) did succeed at ~90 seconds, confirming the endpoint is alive and the model can generate. But coding prompts that require thousands of thinking + code tokens exceed the free-tier compute budget or time limit. The model is a thinking model that emits `<think>` tokens before code output, which multiplies the generation length.

**Verdict:** MiniMax M2.7 on NVIDIA's free NIM API is not benchmarkable with our coding suite. The endpoint works for trivial prompts but times out on anything that requires sustained generation. Unlike the M2.5 issue (queue + output cap), this is a pure infrastructure timeout — the 16K output cap would be sufficient if the endpoint could stay connected long enough to generate.

## Raw artifacts

- Bench script: [`tools/api_bench.py`](../tools/api_bench.py)
- Results JSON: [`experiments/api_bench/results.json`](../experiments/api_bench/results.json)
- Per-run test files: `experiments/api_bench/{model}_{benchmark}_run{n}_test.py`

## See also

- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — local model tier list (same benchmark suite, same scoring)
- [ROTORQUANT_5090.md](ROTORQUANT_5090.md) — KV compression experiments on local models
