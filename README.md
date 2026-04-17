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

Per-platform model rankings (each is self-contained: tier list, throughput, VRAM, max context):

- [results/MODEL_RANKINGS_5090.md](results/MODEL_RANKINGS_5090.md) — RTX 5090 32GB, Windows, llama.cpp/TurboQuant fork
- [results/MODEL_RANKINGS_RTXPRO6000.md](results/MODEL_RANKINGS_RTXPRO6000.md) — RTX Pro 6000 Blackwell 96GB, Ubuntu, stock llama.cpp (Vulkan + CUDA)
- [results/MODEL_RANKINGS_SPARK.md](results/MODEL_RANKINGS_SPARK.md) — DGX Spark GB10 128GB, Linux/aarch64, llama.cpp
- [results/MODEL_RANKINGS_M4MAX.md](results/MODEL_RANKINGS_M4MAX.md) — MacBook M4 Max, Metal/MLX

See [results/HARDWARE_SPECS.md](results/HARDWARE_SPECS.md) for measured side-by-side throughput on the same models across the 3 benchmarked machines, and [results/README.md](results/README.md) for a navigation guide to all the analysis docs.

For per-platform context window / KV-cache analysis:

- [results/CONTEXT_CAPACITY_5090.md](results/CONTEXT_CAPACITY_5090.md) — RTX 5090 32GB (KV cache spill cliff, full 256K context tests)
- [results/CONTEXT_CAPACITY_M4MAX.md](results/CONTEXT_CAPACITY_M4MAX.md) — M4 Max 36GB (Metal working set ceiling, OOM-not-spill failure mode)

For the shortlist of hardware that's a serious live option for local model workflows (Apple Silicon laptops/Studios + NVIDIA workstation class):

- [results/HARDWARE_SHORTLIST.md](results/HARDWARE_SHORTLIST.md) — buyer's guide covering M4/M5 Max, M3 Ultra Studio, RTX 5090, RTX Pro 6000 Blackwell, and DGX Spark. Mix of our measurements, third-party benchmarks (cited inline), and bandwidth-math projections.

## Key Findings

1. **Thinking models produce better code** — Qwen 3.5 and Gemma 4 use chain-of-thought reasoning that catches bugs before they're written. The Nemotron models think less and make more mistakes.

2. **Quantization affects code quality** — Gemma Q6_K scored 5/5 on expression evaluator while Q4_K_M scored 4/5. The extra 2 bits per weight eliminated a subtle error.

3. **Context length is constrained by KV cache, not model weights** — Hybrid architectures (DeltaNet, Mamba, sliding window attention) dramatically reduce per-token KV cost, enabling larger context windows on the same hardware.

4. **"Loads successfully" != "fits in VRAM"** — LM Studio silently spills to system RAM. Gemma at 256K context drops from 150 to 55 tok/s due to RAM spill. Always verify with throughput benchmarks.

5. **Infrastructure params can change output** — `parallel=4` vs `parallel=1` produces different code at `temperature=0` due to floating-point accumulation order changes.

## Repo Layout

```
benchmarks/      Prompt definitions for the 3 coding benchmarks (md files)
templates/       Jinja chat templates for non-default model families
tools/           Python scripts: bench runners, scorers, helpers
results/         Tier-list rankings, hardware specs, comparison tables
experiments/     Per-experiment output directories (one per bench run)
README.md        This file
METHODOLOGY.md   How the benchmarks are scored
```

## Tools

- [`tools/lmstudio_bench.py`](tools/lmstudio_bench.py) — Throughput benchmark (streaming tok/s, TTFT, thinking token tracking)
- [`tools/compare_outputs.py`](tools/compare_outputs.py) — Code quality benchmark runner (load/generate/save per model)
- [`tools/tuning_experiments.py`](tools/tuning_experiments.py) — Parameter sweep (parallel, eval_batch_size, KV cache quant)
- [`tools/extract_and_test.py`](tools/extract_and_test.py) — Extract code blocks from .md outputs and run pytest (1-block-per-file pattern)
- [`tools/score_combined.py`](tools/score_combined.py) — Same idea but combines all blocks into one file (handles bundled impl+test outputs)
- [`tools/m4max_bench.py`](tools/m4max_bench.py) — M4 Max benchmark via direct llama-server (handles `--reasoning-budget 0`)
- [`tools/m4max_mlx_bench.py`](tools/m4max_mlx_bench.py) — Same prompts via `mlx_lm.server` for MLX vs llama.cpp comparison
