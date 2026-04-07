# Local Model Evaluation

Methodology and results for benchmarking local LLMs on consumer hardware. Focuses on coding task quality, throughput, and practical deployment constraints (VRAM, context window, quantization tradeoffs).

## Hardware

- **GPU:** NVIDIA RTX 5090 (32 GB VRAM)
- **Platform:** Windows 11, LM Studio (llama.cpp backend)
- **All models:** GGUF format, fully GPU-offloaded, flash attention enabled

## Evaluation Dimensions

| Dimension | What We Measure | Why It Matters |
|---|---|---|
| **Throughput** | Tokens/sec (streaming) | Interactive responsiveness |
| **TTFT** | Time to first token | Perceived latency |
| **Code Quality** | Tests passing out-of-the-box | Can you trust the output? |
| **Context Capacity** | Max context before VRAM spill | Long document / conversation support |
| **Thinking Efficiency** | Reasoning tokens vs content tokens | Token budget overhead |

## Benchmark Suite

Three coding tasks of increasing difficulty, each requiring the model to produce a complete implementation + working pytest tests:

| Benchmark | Difficulty | Tests | Skills Evaluated |
|---|---|---|---|
| [Expression Evaluator](benchmarks/expression_evaluator.md) | Medium | 5 | Recursive descent parsing, operator precedence, error handling |
| [A* Pathfinding](benchmarks/astar_pathfinding.md) | Medium-Hard | 6 | Graph algorithms, heap usage, edge cases |
| [LRU Cache with TTL](benchmarks/lru_cache_ttl.md) | Hard | 6 | Doubly-linked list + hash map, time mocking, lazy expiry |

All benchmarks are run at `temperature=0` for deterministic output. Code is extracted and tested **exactly as generated** with zero fixes.

## Results Summary

See [results/MODEL_RANKINGS.md](results/MODEL_RANKINGS.md) for the full ranking table.

See [results/CONTEXT_CAPACITY.md](results/CONTEXT_CAPACITY.md) for VRAM and context window analysis.

## Key Findings

1. **Thinking models produce better code** — Qwen 3.5 and Gemma 4 use chain-of-thought reasoning that catches bugs before they're written. The Nemotron models think less and make more mistakes.

2. **Quantization affects code quality** — Gemma Q6_K scored 5/5 on expression evaluator while Q4_K_M scored 4/5. The extra 2 bits per weight eliminated a subtle error.

3. **Context length is constrained by KV cache, not model weights** — Hybrid architectures (DeltaNet, Mamba, sliding window attention) dramatically reduce per-token KV cost, enabling larger context windows on the same hardware.

4. **"Loads successfully" != "fits in VRAM"** — LM Studio silently spills to system RAM. Gemma at 256K context drops from 150 to 55 tok/s due to RAM spill. Always verify with throughput benchmarks.

5. **Infrastructure params can change output** — `parallel=4` vs `parallel=1` produces different code at `temperature=0` due to floating-point accumulation order changes.

## Tools

- [`tools/lmstudio_bench.py`](tools/lmstudio_bench.py) — Throughput benchmark (streaming tok/s, TTFT, thinking token tracking)
- [`tools/compare_outputs.py`](tools/compare_outputs.py) — Code quality benchmark runner (load/generate/save per model)
- [`tools/tuning_experiments.py`](tools/tuning_experiments.py) — Parameter sweep (parallel, eval_batch_size, KV cache quant)
