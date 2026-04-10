# Testing Methodology

## How to Evaluate a New Model

### 1. Throughput Benchmark

```bash
python tools/lmstudio_bench.py --models "publisher/model-name" --max-tokens 2048 --context-length 32768
```

Measures:
- Tokens per second (streaming)
- Time to first token (TTFT)
- Thinking vs content token breakdown
- Generation time

### 2. Code Quality Benchmarks

Run the expression evaluator benchmark first (medium difficulty):

```bash
# Update MODELS list in tools/compare_outputs.py, then:
python tools/compare_outputs.py
```

If the model passes, run the harder benchmarks:

```bash
# Update MODELS list in tools/hard_benchmarks.py, then:
python tools/hard_benchmarks.py
```

### 3. Extract and Test Code

For each model output `.md` file:

1. Extract Python code blocks (implementation + tests)
2. Save as separate files (e.g., `expression_evaluator.py` + `test_expression_evaluator.py`)
3. Fix import module names if needed (e.g., tests import from `evaluator` but file is `expression_evaluator`)
4. **Do NOT fix any code bugs** — run exactly as generated
5. Run `pytest -v` and record pass/fail counts

### 4. Context Capacity Test

Load the model at increasing context sizes and verify throughput doesn't cliff:

```python
SIZES = [32768, 65536, 114688, 196608, 262144]
# For each size: load, run streaming benchmark, check tok/s, unload
```

A throughput drop >20% indicates VRAM spill to system RAM.

### 5. Record Results

Add a row to the platform-specific rankings file (`results/MODEL_RANKINGS_<PLATFORM>.md`) with:
- Model name, quant, weights size
- Verified max context (where throughput stays stable)
- Tok/s and TTFT
- Test pass counts for each benchmark
- Tier assignment (S/A/B/C/D/F)

## Benchmark Parameters

| Parameter | Value | Rationale |
|---|---|---|
| temperature | 0 | Deterministic output for reproducibility |
| context_length (load) | 32768 | Enough for benchmarks, avoids VRAM pressure |
| max_tokens | 16384 | Room for thinking + code output |
| flash_attention | true | Standard optimization |
| parallel | default (4) | Match typical user configuration |

## What We Evaluate

### Code Quality (Primary)
- **Does it import?** — Syntax errors, broken regex, bad imports
- **Do tests pass?** — Run pytest exactly as generated, zero fixes
- **Are implementations correct?** — Even if tests fail, is the core logic sound?

### Performance
- **Tok/s** — Sustained generation speed (streaming)
- **TTFT** — Time to first token (prompt processing latency)
- **Thinking efficiency** — Ratio of thinking tokens to content tokens

### Practical Deployment
- **VRAM fit** — Does the model + KV cache fit entirely in GPU memory?
- **Context capacity** — Max context before throughput degrades
- **Quant tradeoff** — Quality difference between quant levels (e.g., Q4 vs Q6)

## Tier Criteria

| Tier | Code Quality | Requirements |
|---|---|---|
| **S** | 95-100% tests pass | Correct code on all benchmarks including hard ones |
| **A** | 85-94% tests pass | Correct on medium tasks, minor issues on hard ones |
| **B** | 70-84% tests pass | Correct on easy tasks, struggles on hard ones |
| **C** | 50-69% tests pass | Inconsistent — some working code, some broken |
| **D** | 20-49% tests pass | Implementation sometimes sound but tests rarely work |
| **F** | <20% tests pass | Non-functional code output |

## Known Gotchas

1. **context_length pre-allocates KV cache** — Setting it high doesn't help if you don't need it. It just wastes VRAM.

2. **Thinking models need more max_tokens** — Qwen 3.5 and Gemma 4 use chain-of-thought. At 4096 max_tokens, Qwen models ran out of budget. 16384 gives enough headroom.

3. **"Loads successfully" != "fits in VRAM"** — LM Studio silently spills to RAM. Always verify with a throughput benchmark at target context size.

4. **parallel=4 vs parallel=1 changes output** — Even at temperature=0, different parallel slot counts produce different code due to FP accumulation order. Keep parallel consistent across runs for fair comparison.

5. **API can't switch quant variants** — Must select the quant (Q4_K_M, Q6_K, etc.) in the LM Studio UI before loading via API.

6. **Thinking toggle doesn't work via API** — System prompt "detailed thinking off" and `/no_think` prefix are unreliable through the OpenAI-compatible API. Must be configured in LM Studio model settings if available.
