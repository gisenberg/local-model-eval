# Model Rankings

Tested April 2026 on RTX 5090 32GB, LM Studio, all models fully GPU-offloaded with flash attention.

## Tier List

| Tier | Model | Quant | Weights | Verified Max Ctx | Tok/s | TTFT | Expr Eval (5) | A* Path (6+) | LRU Cache (6) | Total | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** | qwen3.5-35b-a3b | Q4_K_M | 22.1 GB | 256K | 92.9 | 2.39s | 5/5 | 8/8 | 6/6 | **19/19 (100%)** | Best code quality. Slowest of the top tier but never wrong. |
| **A** | gemma-4-26b-a4b | Q6_K | ~20 GB | 256K* | 161.9 | 2.43s | 5/5 | 6/6 | 5/6 | **16/17 (94%)** | Best speed/quality ratio. One subtle LRU lazy-cleanup bug. |
| **A-** | gemma-4-26b-a4b | Q4_K_M | 18.0 GB | 256K* | 168.8 | 2.41s | 4/5 | -- | -- | -- | Fastest. Q4 quant introduces test wording error vs Q6_K. |
| **B** | qwen3.5-9b | Q8_0 | 10.4 GB | 256K | 113.2 | 2.20s | 5/5 | 6/7 | 4/6 | **15/18 (83%)** | Good for light tasks. Struggles on hard problems. Overthinks (14K+ chars CoT). |
| **D** | nemotron-3-nano | Q4_K_M | 24.5 GB | ~128K | 78.4 | 4.22s | 0/5 | 3/7 | 0/6 | **3/18 (17%)** | Implementation often sound, test files consistently broken. |
| **F** | nemotron-3-nano-4b | Q8_0 | 4.2 GB | 262K | 209.8 | 2.65s | 0/5 | -- | -- | -- | Fastest raw throughput but produces non-functional code. |

*Gemma at 256K context spills to system RAM (UI estimates 41 GB). Full speed up to ~192K. See [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md).

## Benchmark Details

### Expression Evaluator (Medium)
Recursive descent parser for `+`, `-`, `*`, `/` with operator precedence, parentheses, unary minus, floats, and error handling. 5 pytest tests.

- **S/A tier**: All produced correct implementations. Differences were in test assertions.
- **D/F tier**: Nemotron 30B had a test import bug (`from module import ValueError`). Nemotron 4B had a broken regex (`(?P<plus>+)` — unescaped quantifier) plus undefined variables.

### A* Pathfinding (Medium-Hard)
Weighted 2D grid pathfinding with walls, Manhattan distance heuristic, heapq open set. 6 pytest tests.

- **Qwen 35B**: 8/8 (generated bonus tests, all passed)
- **Gemma Q6_K**: 6/6
- **Qwen 9B**: 6/7 (one test had wrong cost assertion — `== 8` should be `== 5`)
- **Nemotron 30B**: 3/7 (correct implementation, tests hardcode specific paths among multiple optimal alternatives + expect ValueError for non-wall coordinates)

### LRU Cache with TTL (Hard)
O(1) LRU cache with doubly-linked list + hash map, time-based expiry via mocked `time.monotonic()`, lazy cleanup. 6 pytest tests.

- **Qwen 35B**: 6/6 — used `_valid_count` tracking to handle expired-item counting correctly
- **Gemma Q6_K**: 5/6 — `size()` returns `len(cache)` which includes expired-but-unaccessed items
- **Qwen 9B**: 4/6 — `_evict()` uses identity check on node objects vs dict values; `size()` counts stale entries
- **Nemotron 30B**: 0/6 — sentinel nodes created without required `expire_at` argument, every test crashes at construction

## Throughput Observations

- All top-tier models cluster around 2.2-2.4s TTFT regardless of size — dominated by prompt processing
- Nemotron 30B is an outlier at 4.2s TTFT due to memory pressure from 24.5 GB weights
- Thinking models (Qwen 3.5, Gemma 4) spend 6-15K chars on chain-of-thought reasoning. This improves code quality but consumes token budget — context_length must accommodate both thinking + output
- Non-thinking models (Nemotron) are more token-efficient but produce worse code

## Quantization Impact

Direct comparison of Gemma 4 26B at two quant levels, same prompt:

| Quant | Bits/Weight | Tok/s | Expr Eval | Observation |
|---|---|---|---|---|
| Q4_K_M | 4 | 168.8 | 4/5 | Test regex mismatch on error message wording |
| Q6_K | 6 | 161.9 | 5/5 | All tests pass — extra precision eliminated the subtle error |

4% throughput cost for Q6_K, but a measurable code quality improvement. For coding tasks, Q6_K is worth the tradeoff.
