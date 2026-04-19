# Qwen3.6-35B-A3B Claude-4.6-Opus-Reasoning-Distilled — coding bench (RTX Pro 6000)

**TL;DR — regresses from 21/22 (stock Qwen3.6 + thinking) to 10/22 regardless
of the `-rea on/off` flag. Easy benchmarks still pass 5/5; hard benchmarks
(A* Pathfinding, LRU Cache with TTL) crash with pytest collection errors
because the model writes tests that don't compile. The fine-tune also
appears to no-op on Qwen's thinking convention — the flag changes nothing.
Not recommended as a Qwen3.6 replacement.**

## What we tested

- **Model:** [`hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF`](https://huggingface.co/hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) at Q8_0 (36.9 GB).
- **Base model:** `Qwen/Qwen3.6-35B-A3B` — identical architecture (`qwen35moe` in llama.cpp), loads under the same CUDA build with no changes.
- **What the fine-tune is:** SFT on Claude Opus 4.6 chain-of-thought traces, using three public datasets:
  - `nohurry/Opus-4.6-Reasoning-3000x-filtered`
  - `Jackrong/Qwen3.5-reasoning-700x`
  - `Roman1111111/claude-opus-4.6-10000x`
- **Author's claim:** MMLU-Pro 42.86% → 75.71% on a 70-sample limited eval (author flags this as a smoke test).

## Methodology

Standard 4-benchmark coding suite from `tools/rtxpro6000_levers_bench.py`:

| benchmark | expected tests | what it exercises |
|---|---:|---|
| string_processor | 5 | simple CRUD + edge cases |
| expression_evaluator | 5 | tokeniser + recursive descent |
| astar_pathfinding | 6 | graph algorithm + heuristics |
| lru_cache_ttl | 6 | state + TTL semantics + mocking time |

Served via llama.cpp CUDA (commit `a279d0f`, sm_120), 32K ctx, temp 0.6,
top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0 — the Unsloth Qwen3.6
thinking-mode defaults we use for the stock model. Two runs:

1. `--thinking` mode: `-rea on --reasoning-budget 16384`
2. Baseline: no `-rea` flag

Score is all-or-nothing per generated pytest run; `pytest -v` must collect
and pass every test for a benchmark to score.

## Results

| benchmark | stock Qwen3.6 Q8 + thinking | Opus-distilled, `-rea on` | Opus-distilled, `-rea off` |
|---|---:|---:|---:|
| string_processor     | 5/5  | **5/5** | **5/5** |
| expression_evaluator | 5/5  | **5/5** | **5/5** |
| astar_pathfinding    | 6/6  | 0/6 ❌ | 0/6 ❌ |
| lru_cache_ttl        | 5/6  | 0/6 ❌ | 0/6 ❌ |
| **total**            | **21/22 (95%)** | **10/22 (45%)** | **10/22 (45%)** |

Per-benchmark elapsed time was identical across the two runs (13.7 / 17.4 /
17.4 / 17.2 seconds), strongly suggesting `-rea on` was a no-op: Qwen's
thinking path adds a variable amount of wall-clock proportional to the
reasoning budget, but here we see the same wall-clock regardless of flag.

### Why the failures are failures

Both harder benchmarks hit pytest **collection errors** — the generated
test module cannot even be imported:

- **A\* Pathfinding:** test file declares
  `def calculate_path_cost(grid: AStarGrid, path: List[Tuple[int, int]]) -> int`
  but does not `from typing import List, Tuple`. Impl imports `typing`
  correctly; test file forgets to. Pytest emits:

  ```
  E   NameError: name 'List' is not defined
  !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
  ```

- **LRU Cache with TTL:** test file patches time via a hard-coded module
  name `mock.patch('ttl_cache.xxx')` that doesn't match the actual impl
  module. The bench's `fix_test_imports` rewrites top-level
  `from X import Y` lines but does not touch string-literal module names
  inside `mock.patch()` calls. Pytest emits:

  ```
  E   ModuleNotFoundError: No module named 'ttl_cache'
  ```

  (6× — one per test in the file.)

These are test-code quality issues, not thinking-tag malfunctions or
malformed output. The impl files were plausible; the *tests* shipped
broken. Stock Qwen3.6 (21/22) avoids both classes of mistake.

### Why `-rea on` and `-rea off` gave identical output

Two plausible explanations (can't fully distinguish without inspecting the
SFT data, but both point the same practical direction):

1. **Thinking-tag convention was diluted by SFT.** Claude-style reasoning
   traces don't wrap reasoning in `<think>` tags. Training on 13k Claude
   reasoning examples likely shifted the distribution away from Qwen's
   thinking tag convention, so `-rea on` no longer triggers the extended
   reasoning path the base model was trained to produce.
2. **Reasoning is already baked into default output.** The model may have
   learned to always do Claude-style verbose reasoning as part of its
   default generation, making Qwen's explicit thinking-budget flag
   unnecessary or redundant.

In practice the consequence is the same: flipping the flag doesn't recover
the coding quality stock Qwen3.6 shows on these benchmarks.

## Smoke-test note — the output itself is clean

A small pre-benchmark smoke test (`is_palindrome(s: str) -> bool` with a
200-char prompt, `-rea on`) produced perfectly clean output:
- `reasoning_content` populated with short English rationale
- `content` field held a single Python code block with a correct impl
- `finish_reason: "stop"`, no truncation, no malformed tags

So the model itself is functional and well-behaved. The regression is
specifically in generating *correct test suites* for harder problems, not
in basic code generation.

## Caveats

- **Author's MMLU-Pro claim of 75.71% is on a 70-sample limited eval**
  (author explicitly calls this a smoke test in the model card). Our
  4-benchmark coding suite is a different axis — we're measuring whether
  the model can emit a correct pytest suite, not general-knowledge recall.
  The distillation may well improve MMLU-Pro while regressing on coding.
- **Single trial per benchmark.** We did not re-roll with different seeds.
  Stock Qwen3.6 showed some run-to-run variance in earlier ranking work
  (`MODEL_RANKINGS_RTXPRO6000.md`), so treat the 10/22 as a ±1-test
  estimate rather than a precise value.
- **Anthropic ToS note.** Training on Claude API outputs to build
  competitor models violates Anthropic's commercial terms. The distiller
  is the one who took that risk by producing the training data (or in
  this case, by using third-party datasets already assembled). Running
  an already-distributed model is a grayer zone. Treat this as
  research/personal-use only; do not rely on it for commercial work.

## Recommendation

**Don't swap this in for stock Qwen3.6.** The primary daily-driver pick
(`rtxpro6000/qwen36-35b-a3b-coder` with `-rea on`) stays at 21/22; this
fine-tune drops to 10/22 with no lever that recovers the gap. If you
specifically want the Claude-style verbose reasoning for non-coding tasks
(MMLU-Pro-style general knowledge, long-form explanation) the author's
numbers suggest real uplift — but that's a different job than coding, and
we didn't measure that axis here.

## Reproducing

```bash
# Download (36.9 GB):
mkdir -p ~/models/qwen36-opus-distill-q8
curl -L -o ~/models/qwen36-opus-distill-q8/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q8_0.gguf \
    https://huggingface.co/hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/resolve/main/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q8_0.gguf

# Run the 4-benchmark coding suite (thinking on, then baseline):
cd ~/git/gisenberg/local-model-eval
LLAMA_BACKEND=cuda python3 tools/rtxpro6000_levers_bench.py qwen36-opus-distill-q8 --thinking
LLAMA_BACKEND=cuda python3 tools/rtxpro6000_levers_bench.py qwen36-opus-distill-q8
```

Per-trial output + JSON roll-up lands in
`experiments/rtxpro6000_levers/qwen36-opus-distill-q8__thinking/` and
`.../qwen36-opus-distill-q8__baseline/`. The model key
`qwen36-opus-distill-q8` is registered in `tools/rtxpro6000_bench.py` as
of this PR.
