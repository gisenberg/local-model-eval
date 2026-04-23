# Qwen3.6 on RTX Pro 6000 — deep dives

This doc collects the Qwen3.6-family experiments run on the RTX Pro 6000 Blackwell Workstation (96 GB, sm_120) that don't fit naturally in the model-agnostic tier list ([MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md)). Three experiments, one per section:

1. [**RULER long-context eval on Qwen3.6-35B-A3B**](#ruler-long-context-eval-qwen36-35b-a3b) — YaRN ×2/×4 up to 1M tokens. **48/48 tasks pass**, static-YaRN short-context tax is invisible, Qwen3.6-35B-A3B is a true 1M-token model on this hardware.
2. [**Opus-Reasoning-Distilled Qwen3.6-35B-A3B on the coding bench**](#opus-reasoning-distilled-qwen36-35b-a3b--coding-bench-regression) — fine-tune **regresses** from stock's 21/22 to 10/22 on the 4-benchmark suite, independent of `-rea on/off`. Useful for agentic bug-fixing (separately measured on SWE-bench Lite at +3.7 pp vs stock) but lost on from-scratch code generation.
3. [**NVFP4 vs FP8 vs BF16 on Qwen3.6-27B dense**](#precision-comparison-qwen36-27b-dense-nvfp4-vs-fp8-vs-bf16) — vendor FP8 is the sweet spot (91% avg vs 84% BF16, 1.6× throughput, half the memory). NVFP4 ties on best-of-3 but adds no speed over FP8 on this hybrid architecture because attention stays BF16.

For Qwen3.6's SWE-bench Lite numbers (including the dense 27B FP8 run), see [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md).

---

# RULER long-context eval (Qwen3.6-35B-A3B)

**TL;DR — 48/48 RULER tasks pass across all three configs and all context lengths tested, up to and including 1M tokens at YaRN ×4. Static-YaRN short-context tax is invisible on this hybrid-attention model: prefill and decode throughput at 32K/131K/262K is within noise whether YaRN is off, ×2, or ×4. Qwen3.6-35B-A3B is a true 1M-token model on a single RTX Pro 6000.**

![pass rate heatmap](ruler_qwen36_pass_rate.svg)

![throughput vs context](ruler_qwen36_throughput.svg)

## Why this run

The daily-driver `serve.sh` for this host serves Qwen3.6 at **native 262K**. Going past that would require YaRN rope scaling — and Qwen's model card warns that *"static YaRN keeps the scaling factor constant regardless of input length, potentially impacting performance on shorter texts"*. Every open backend (llama.cpp, vLLM, SGLang) implements static YaRN. So a natural worry: if we flip YaRN on to unlock 1M context, do we regress at 32K?

The other open question: with `partial_rotary_factor=0.25` and only 10 of 40 layers being full-attention (the rest are linear/SSM), the actual "RoPE surface area" being stretched is tiny — maybe YaRN extension on this architecture works differently from a dense full-rotary model. No published RULER data existed for this model above 262K on any backend at run time.

## RULER setup

- **Runtime**: llama.cpp CUDA 13.2 build (commit `a279d0f`), `llama-server`.
- **Model**: `unsloth/Qwen3.6-35B-A3B-GGUF` Q8_0 (`qwen35moe` arch).
- **Native `n_ctx_train`**: 262,144. `rope.freq_base` 10,000,000.
- **Architecture**: hybrid linear+full attention with `full_attention_interval=4` → only **10 of 40 layers** use full attention, each with partial rotary (25% of head dims). The other 30 layers are SSM/linear; YaRN rescaling only affects the 10 full-attention layers' rotary dims.

### Configs (each runs its own `llama-server` with exclusive VRAM)

| id | rope flags | max ctx |
|---|---|---|
| `native_262k` | *(none)* | 262,144 |
| `yarn2_512k` | `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144` | 524,288 |
| `yarn4_1m` | `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` | 1,048,576 |

All three: `temp 0.6, top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0, reasoning budget 8192` (Unsloth's Qwen3.6 thinking-mode defaults).

**llama-server caveat we hit**: current master has a hard `n_ctx_slot ≤ n_ctx_train` cap at `tools/server/server-context.cpp:764-768` (see [issue #17459](https://github.com/ggml-org/llama.cpp/issues/17459)). The rope-scaling path itself is wired through for M-RoPE (`rope type = 40`, dispatched in `ggml/src/ggml-cuda/rope.cu`), so YaRN math is correct — the server just refuses to allocate a slot bigger than `n_ctx_train`. Maintainer-confirmed workaround: `--override-kv qwen35moe.context_length=int:<extended>`. That's what this harness uses when `ctx > 262144`. The misleading startup log (`rope scaling = linear`) prints the GGUF-side *training-time* value, not runtime cparams.

### Tasks (4 of RULER's 13)

- **`niah_single`** — one `magic number for <key> is <value>` needle at depth 0.5 of the filler. Exact-match on 6-digit value.
- **`niah_multi`** — three needles at depths 0.15 / 0.5 / 0.85. Must return all three values.
- **`variable`** — 8-step chain `v0 := N; v1 := v0+1; …; v7 := v6+1` scattered at even depths. Must return `N+7`.
- **`common_words`** — three invented words (`zephyrine`, `quondamly`, `verdacious`) injected 24/19/15 times into a 200-line bag spread through the filler. Must return all three (any order).

Filler is deterministic synthetic prose (sentence-template bank, seeded RNG). Context fills to ~80% of `ctx − output_budget` in tokens (measured ~5.1 chars/token). Output budget 12,288 tokens (reasoning 8,192 + answer headroom).

## RULER results

### Pass rate: 48/48 ✅

| ctx | `native_262k` | `yarn2_512k` | `yarn4_1m` |
|---:|:-:|:-:|:-:|
| 32K | 4/4 | 4/4 | 4/4 |
| 131K | 4/4 | 4/4 | 4/4 |
| 262K | 4/4 | 4/4 | 4/4 |
| 524K | n/a | 4/4 | 4/4 |
| 1024K | n/a | n/a | 4/4 |

Every RULER task passes in every cell the config supports. The 1M cell — a 787K-token input with thinking enabled and a 6-digit needle at depth 0.5 of a 2-million-character synthetic archive — resolves correctly.

### Throughput: prefill + decode tok/s (mean over 4 tasks per cell)

| ctx | native prefill / decode | yarn2 prefill / decode | yarn4 prefill / decode |
|---:|---:|---:|---:|
| 32K | 7,132 / 209.7 | 7,061 / 210.2 | 7,063 / 208.5 |
| 131K | 6,160 / 172.7 | 6,165 / 172.1 | 6,121 / 172.2 |
| 262K | 4,916 / 141.3 | 4,910 / 140.9 | 4,865 / 141.2 |
| 524K | — | 3,460 / 103.4 | 3,445 / 102.8 |
| 1024K | — | — | 2,124 / 65.7 |

**Same length, different config**: the three configs overlay perfectly within their shared range. The biggest delta is **0.9 tok/s of prefill at 32K** (7,132 vs 7,063), well inside per-trial noise. The static-YaRN short-context tax that the model card warns about is **not observable** on this hardware for this hybrid-attention model.

### VRAM

KV cache growth with context (from `nvidia-smi` after server warm-up, full Q8 weights resident):

| config × ctx | VRAM (MB) |
|---|---:|
| any × 32K | 36,510 |
| any × 131K | 38,432 |
| any × 262K | 41,304 |
| yarn2/yarn4 × 524K | 47,192 |
| yarn4 × 1M | 58,970 |

Plenty of headroom even at 1M (37 GB free on a 96 GB card), so running yarn4 with `-np 2` for two concurrent 1M contexts or `-np 4` with ~262K per slot is feasible — the llama-swap config already does the latter.

## RULER findings

1. **YaRN on qwen35moe is free up to the native 262K boundary** on this hardware and in this build. Identical pass rate, identical prefill and decode throughput within noise. The Qwen model-card warning about static-YaRN short-context degradation does not materialize. Likely explanation: with partial rotary (25%) × full-attention-only (25% of layers), YaRN is rescaling ~6% of the model's positional surface. Most of the model — linear-attention layers, non-rotary head dims — is untouched, so the practical distortion is tiny.
2. **Qwen3.6-35B-A3B Q8 is a functional 1M-token model on a single Pro 6000.** Retrieval (niah_single, niah_multi) and light reasoning (variable, common_words) all resolve at 1M with YaRN ×4. Prefill ~6 minutes and VRAM sits at 59 GB.
3. **Prefill scales roughly linearly in context**, because the hybrid architecture's linear layers stay cheap: 7,060 → 6,160 → 4,900 → 3,450 → 2,120 tok/s across 32K / 131K / 262K / 524K / 1M. Decode falls off harder (210 → 172 → 141 → 103 → 66 tok/s) since each decoded token must attend over the full context in the full-attention layers.
4. **No task-wise asymmetry.** `niah_multi` (3 needles), `variable` (8-step chain), and `common_words` (frequency aggregation) all behaved like `niah_single` across configs and lengths. With thinking on, reasoning budgets 8K, and this corpus, every task was a solved problem.

## RULER recommendation

For the opencode-config daily driver: **keep `native_262k` as the default serve config**. YaRN isn't free in the sense that it adds a llama-server start-up flag, a `--override-kv` workaround for the current llama.cpp master, and occupies more KV VRAM per slot. But:

- **Add a `serve-qwen36-yarn4.sh` as a documented alternative** for tasks that genuinely need >262K (very long document Q&A, whole-repo refactor agents, long chat histories). No reason to *avoid* YaRN ×4 — just reach for it only when the job requires it.
- **Don't bother with ×2 separately.** ×2 and ×4 behave identically at shared context lengths, so if you're going to turn YaRN on at all, go straight to ×4.
- **Downstream bench idea**: re-run the 4-benchmark coding suite under `yarn4_1m` at 32K input. If coding score holds at 21/22, YaRN on qwen35moe is a pure upgrade and we can ship it as the default. This RULER pass confirms retrieval/reasoning are unaffected; the only remaining question is code generation.

## RULER reproducing

```bash
cd ~/git/gisenberg/local-model-eval
# Full matrix: ~2 hours end-to-end, single-trial per cell.
python3 tools/rtxpro6000_ruler_bench.py \
    --config native_262k yarn2_512k yarn4_1m \
    --lengths 32768 131072 262144 524288 1048576 \
    --tasks niah_single niah_multi variable common_words \
    --trials 1
# Regenerate charts from whatever's in experiments/ruler_qwen36/
python3 tools/rtxpro6000_ruler_chart.py
```

The harness starts/stops its own `llama-server` with the right rope flags per config, uses port `18090` to avoid colliding with the daily-driver serve on `8080`, and streams per-trial JSON to `experiments/ruler_qwen36/` so a crash mid-matrix keeps partial data.

## RULER caveats

- **Single trial per cell.** Thinking is stochastic; borderline cases could flip on re-run. 48/48 is strong enough that we're unlikely hiding a 50% failure mode, but rerun with `--trials 3` before drawing fine-grained conclusions from any individual cell.
- **Synthetic filler is not Paul Graham essays** (the canonical NIAH corpus). Our filler is more structurally regular (short varied sentence templates), which likely makes needle retrieval slightly easier than the public benchmark. Treat absolute pass rates as indicative of this corpus, not of NIAH literature at large.
- **Scoring is permissive.** The model may return reasoning + answer; we check whether the expected token(s) appear *anywhere* in the content. Chatty correct and terse correct both pass.
- **No community RULER baseline** for `qwen35moe` at >262K exists on any backend at the time of this run, so there's no external number to calibrate absolute scores against. The *relative* finding (YaRN doesn't cost short-ctx quality or throughput) is robust independent of that.
- **8K cells skipped** because the 12,288-token output budget for thinking + answer exceeds 8,192. The 32K cell is the short-context signal.

---

# Opus-Reasoning-Distilled Qwen3.6-35B-A3B — coding bench regression

**TL;DR — regresses from 21/22 (stock Qwen3.6 + thinking) to 10/22 regardless of the `-rea on/off` flag. Easy benchmarks still pass 5/5; hard benchmarks (A* Pathfinding, LRU Cache with TTL) crash with pytest collection errors because the model writes tests that don't compile. The fine-tune also appears to no-op on Qwen's thinking convention — the flag changes nothing. Not recommended as a Qwen3.6 replacement for from-scratch code generation, but **+3.7 pp on SWE-bench Lite vs stock** (see [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md) for the agentic-workload story).**

## What we tested

- **Model**: [`hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF`](https://huggingface.co/hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) at Q8_0 (36.9 GB).
- **Base model**: `Qwen/Qwen3.6-35B-A3B` — identical architecture (`qwen35moe` in llama.cpp), loads under the same CUDA build with no changes.
- **What the fine-tune is**: SFT on Claude Opus 4.6 chain-of-thought traces, using three public datasets:
  - `nohurry/Opus-4.6-Reasoning-3000x-filtered`
  - `Jackrong/Qwen3.5-reasoning-700x`
  - `Roman1111111/claude-opus-4.6-10000x`
- **Author's claim**: MMLU-Pro 42.86% → 75.71% on a 70-sample limited eval (author flags this as a smoke test).

## Opus-distill methodology

Standard 4-benchmark coding suite from `tools/rtxpro6000_levers_bench.py`:

| benchmark | expected tests | what it exercises |
|---|---:|---|
| string_processor | 5 | simple CRUD + edge cases |
| expression_evaluator | 5 | tokeniser + recursive descent |
| astar_pathfinding | 6 | graph algorithm + heuristics |
| lru_cache_ttl | 6 | state + TTL semantics + mocking time |

Served via llama.cpp CUDA (commit `a279d0f`, sm_120), 32K ctx, temp 0.6, top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0 — the Unsloth Qwen3.6 thinking-mode defaults for the stock model. Two runs:

1. `--thinking`: `-rea on --reasoning-budget 16384`
2. Baseline: no `-rea` flag

Score is all-or-nothing per generated pytest run; `pytest -v` must collect and pass every test for a benchmark to score.

## Opus-distill results

| benchmark | stock Qwen3.6 Q8 + thinking | Opus-distilled `-rea on` | Opus-distilled `-rea off` |
|---|---:|---:|---:|
| string_processor | 5/5 | **5/5** | **5/5** |
| expression_evaluator | 5/5 | **5/5** | **5/5** |
| astar_pathfinding | 6/6 | 0/6 ❌ | 0/6 ❌ |
| lru_cache_ttl | 5/6 | 0/6 ❌ | 0/6 ❌ |
| **total** | **21/22 (95%)** | **10/22 (45%)** | **10/22 (45%)** |

Per-benchmark elapsed time was identical across the two runs (13.7 / 17.4 / 17.4 / 17.2 seconds), strongly suggesting `-rea on` was a no-op: Qwen's thinking path adds a variable amount of wall-clock proportional to the reasoning budget, but here we see the same wall-clock regardless of flag.

### Why the failures are failures

Both harder benchmarks hit pytest **collection errors** — the generated test module cannot even be imported:

- **A* Pathfinding**: test file declares `def calculate_path_cost(grid: AStarGrid, path: List[Tuple[int, int]]) -> int` but does not `from typing import List, Tuple`. Impl imports `typing` correctly; test file forgets to. Pytest emits:

  ```
  E   NameError: name 'List' is not defined
  !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
  ```

- **LRU Cache with TTL**: test file patches time via a hard-coded module name `mock.patch('ttl_cache.xxx')` that doesn't match the actual impl module. The bench's `fix_test_imports` rewrites top-level `from X import Y` lines but doesn't touch string-literal module names inside `mock.patch()`. Pytest emits:

  ```
  E   ModuleNotFoundError: No module named 'ttl_cache'
  ```

  (6× — one per test in the file.)

These are test-code quality issues, not thinking-tag malfunctions or malformed output. The impl files were plausible; the *tests* shipped broken. Stock Qwen3.6 (21/22) avoids both classes of mistake.

### Why `-rea on` and `-rea off` gave identical output

Two plausible explanations (can't fully distinguish without inspecting the SFT data, but both point the same practical direction):

1. **Thinking-tag convention diluted by SFT.** Claude-style reasoning traces don't wrap reasoning in `<think>` tags. Training on 13k Claude reasoning examples likely shifted the distribution away from Qwen's thinking tag convention, so `-rea on` no longer triggers the extended reasoning path the base model was trained to produce.
2. **Reasoning is already baked into default output.** The model may have learned to always do Claude-style verbose reasoning as part of its default generation, making Qwen's explicit thinking-budget flag unnecessary or redundant.

In practice the consequence is the same: flipping the flag doesn't recover the coding quality stock Qwen3.6 shows.

## Smoke-test note — the output itself is clean

A small pre-benchmark smoke test (`is_palindrome(s: str) -> bool` with a 200-char prompt, `-rea on`) produced perfectly clean output: `reasoning_content` populated with short English rationale, `content` field held a single Python code block with a correct impl, `finish_reason: "stop"`, no truncation, no malformed tags. So the model itself is functional and well-behaved. The regression is specifically in generating *correct test suites* for harder problems, not in basic code generation.

## Opus-distill caveats

- **Author's MMLU-Pro claim of 75.71% is on a 70-sample limited eval** (author explicitly calls this a smoke test in the model card). Our 4-benchmark coding suite is a different axis — measuring whether the model can emit a correct pytest suite, not general-knowledge recall. The distillation may well improve MMLU-Pro while regressing on coding.
- **Single trial per benchmark.** Stock Qwen3.6 showed some run-to-run variance in earlier ranking work; treat 10/22 as a ±1-test estimate.
- **Anthropic ToS note.** Training on Claude API outputs to build competitor models violates Anthropic's commercial terms. The distiller took that risk by producing training data (here, via third-party datasets). Running an already-distributed model is a grayer zone — research/personal-use only.

## Opus-distill recommendation

**Don't swap this in for stock Qwen3.6 for from-scratch coding.** The primary daily-driver pick (`rtxpro6000/qwen36-35b-a3b-coder` with `-rea on`) stays at 21/22; this fine-tune drops to 10/22 with no lever that recovers the gap.

**But for agentic bug-fixing** (codebase navigation, SWE-bench-style tasks): Opus-distill is +3.7 pp over stock on SWE-bench Lite (156/300 vs 145/300). SWE-bench's harness provides its own hidden tests, so the "model writes broken tests" regression axis never comes up. Details in [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md#qwen36-35b-a3b-opus-distilled-q8_0-llama-swap--520).

**Picking by workload** is the right move — don't expect a single "best Qwen3.6" across benchmarks.

## Opus-distill reproducing

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

Per-trial output + JSON roll-up lands in `experiments/rtxpro6000_levers/qwen36-opus-distill-q8__thinking/` and `.../qwen36-opus-distill-q8__baseline/`. Model key `qwen36-opus-distill-q8` is registered in `tools/rtxpro6000_bench.py`.

---

# Precision comparison (Qwen3.6-27B dense): NVFP4 vs FP8 vs BF16

Three-way comparison of Qwen3.6-27B at different precisions on the 4-bench coding suite, served via vLLM on RTX Pro 6000 Blackwell 96 GB. Different model from the 35B-A3B MoE above — this is the **dense VLM 27B** variant (Qwen3.6-27B-FP8 HF release), which we benchmarked because FP8 is the vendor's primary release format.

## Precision results

| Variant | Weights | Best-of-3 | Avg | Tok/s | Size | Source |
|---|---|---|---|---|---|---|
| **BF16** (baseline) | `Qwen/Qwen3.6-27B` | 21/22 (95%) | 18.4/22 (84%) | 29 | 52 GB | HF release |
| **FP8** (vendor) | `Qwen/Qwen3.6-27B-FP8` | 21/22 (95%) | 20.0/22 (91%) | 47 | 29 GB | HF release |
| **NVFP4** (modelopt) | `mmangkad/Qwen3.6-27B-NVFP4` † | 22/22 (100%) | 18.6/22 (85%) | 46 | 29 GB | community, modelopt 0.42 |

† See [NVFP4 checkpoint note](#nvfp4-checkpoint-note) — we self-quantized with modelopt 0.43 but the export didn't load in vLLM 0.19.1, so we swapped to a community checkpoint with the same methodology.

**Headline**: FP8 is the sweet spot. Matches BF16 on best-of-3, highest avg score, 1.6× faster, halves the memory footprint. NVFP4 gets the only 100% best-of-3 but loses ~1.5 points on avg because it produced one zero run on A*; measured throughput was the same as FP8 (46 vs 47 tok/s) despite the smaller weight footprint, suggesting vLLM's NVFP4 path on Blackwell isn't yet hitting native FP4 tensor cores through flashinfer-cutlass JIT.

## Precision per-benchmark

Best/avg out of 3 runs at T=0.3, max_tokens=15000, max-model-len=16384.

| Benchmark | BF16 best/avg | FP8 best/avg | NVFP4 best/avg |
|---|---|---|---|
| Expression Evaluator (5) | 5 / 4.0 | 5 / 4.3 | 5 / 4.3 |
| A* Pathfinding (6) | 5 / 4.7 | 5 / 5.0 | **6** / 3.3 * |
| LRU Cache w/ TTL (6) | 6 / 5.0 | 6 / 5.7 | **6 / 6.0** |
| String Processor (5) | 5 / 4.7 | 5 / 5.0 | 5 / 5.0 |
| **Total** | **21 / 18.4** | **21 / 20.0** | **22 / 18.6** |

\* NVFP4 A* run 1 produced no extractable code block (0/6). Run 2 was 4/6, run 3 was 6/6. Variance is real; the 100% best-of-3 is one lucky run, not a consistent quality lead.

## Precision methodology

- **Hardware**: RTX Pro 6000 Blackwell 96 GB, driver 580.126.09, CUDA 12.8 runtime.
- **Serving**: vLLM 0.19.1 with transformers built from main (required for `qwen3_5` arch).
- **Bench**: [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py) — identical 4-benchmark suite as `nvfp4_gemma31b_bench_v2.py`, 3 runs each, T=0.3. Raw outputs in [`experiments/nvfp4_qwen36_27b/`](../experiments/nvfp4_qwen36_27b/).
- **Quantization**: [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — NVIDIA Model Optimizer, `MAMBA_MOE_NVFP4_CONSERVATIVE_CFG` + `*linear_attn*` exclusion for Qwen3.5's gated-delta-net layers, calibrated on `abisee/cnn_dailymail` (128 samples, seq_len 2048) — matching NVIDIA's reference recipe for Qwen3-NVFP4 releases.
- **Thinking**: Qwen3.6 is a reasoning model. The bench strips `<think>...</think>` before extracting code blocks (extraction regex would otherwise glue scratchwork fragments into invalid Python).

## NVFP4 checkpoint note

We produced our own NVFP4 checkpoint with `modelopt 0.43.0` using the config above. Resulting weights have the same `quantization_config.ignore` list (66 entries) as `mmangkad/Qwen3.6-27B-NVFP4` (67 entries, same pattern) — all `linear_attn` and some `self_attn` layers stay BF16, MLPs get NVFP4.

Our export loads into vLLM 0.19.1 but crashes during forward:

```
File ".../vllm/model_executor/models/qwen3_next.py", line 500, in forward
KeyError: 'weight_scale'
```

The `mmangkad` checkpoint (produced with `modelopt 0.42.0rc1.dev107`, nearly identical ignore list) loads and runs cleanly. The bench numbers above are from `mmangkad`'s checkpoint. The quality result — that a modelopt NVFP4 of this model can match BF16 on best-of-3 — still holds; we just didn't produce the checkpoint ourselves in this session.

Diagnosis is incomplete: it looks like modelopt 0.43 changed something in how quantized state is serialized for hybrid `linear_attn` + `self_attn` models that vLLM's `qwen3_next` loader doesn't handle yet. Not chased further in this session — pinning `nvidia-modelopt==0.42.*` would likely reproduce a working self-quantized checkpoint.

## Why NVFP4 doesn't beat FP8 on throughput

On paper Blackwell FP4 tensor cores should be ~2× FP8's throughput. We saw 46 vs 47 tok/s — effectively a tie. Two reasons:

1. **Only MLPs are quantized.** Both the `linear_attn` (gated-delta-net) blocks and the `self_attn` blocks stay in BF16 per the MAMBA-MoE conservative config. For a hybrid architecture where attention + gating is a large fraction of FLOPs, halving MLP precision gives a smaller speedup than for a pure-MLP-heavy dense transformer.
2. **flashinfer-cutlass JIT is not the fastest FP4 path on SM 120.** The `flashinfer-trtllm` backend would use pre-compiled TRT-LLM kernels (likely better tuned) but failed to compile in our env. The `cutlass` (non-flashinfer) backend loaded but produced garbage output — looks like a correctness bug in vLLM 0.19.1's native cutlass NVFP4 path for hybrid Qwen3.5.

A Qwen3.6-27B that kept every Linear quantized (no `linear_attn`/`self_attn` exclusions) would almost certainly decode faster, but the "quantize everything" attempt produced garbage — modelopt's default config doesn't survive the hybrid architecture's gated-delta-net projections.

## NVFP4/FP8/BF16 build notes

Getting modelopt + vLLM + flashinfer FP4 working end-to-end on a fresh Ubuntu 24.04 box required more than `pip install`:

1. `sudo apt install -y build-essential python3.12-dev` — Triton JIT needs `gcc` and `Python.h`.
2. `uv pip install ... nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl ninja` — flashinfer's FP4 kernels JIT-compile cutlass code that needs CUDA 13+ vector types (`ulonglong4_16a` etc.). The CUDA 12.x toolchain `apt` ships (`nvidia-cuda-toolkit`) won't compile it.
3. Overlay CUDA 12 libs (`curand`, `cublas`, `cufft`, `cudnn`, `cooperative_groups/`) into the CUDA 13 include tree — flashinfer expects a single `$CUDA_HOME` with everything, but the CUDA 13 PyPI split doesn't include those. Also symlink `libcudart.so` and `libcuda.so` stubs so `ld` resolves `-lcudart -lcuda`.
4. Uninstall `deepspeed` (modelopt pulls it as transitive; its import-time `installed_cuda_version()` raises `MissingCUDAException` if `CUDA_HOME` isn't set before anything else runs).
5. `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` (the default auto-pick works once the JIT chain is fixed; `cutlass` without flashinfer produces garbage output on SM 120, `marlin` errors on layer dims not divisible by 64).

Total wall-clock on first-time setup: ~15 min for the toolchain after the ~25 min pip install + 15 min model downloads.

## Precision files

- [`experiments/nvfp4_qwen36_27b/bf16/results.json`](../experiments/nvfp4_qwen36_27b/bf16/results.json)
- [`experiments/nvfp4_qwen36_27b/fp8/results.json`](../experiments/nvfp4_qwen36_27b/fp8/results.json)
- [`experiments/nvfp4_qwen36_27b/nvfp4/results.json`](../experiments/nvfp4_qwen36_27b/nvfp4/results.json)
- [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — reusable for other models
- [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py)

---

# See also

- [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md) — tier list for this platform; Qwen3.6-35B-A3B is the daily-driver pick.
- [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md) — four-model SWE-bench Lite comparison including Qwen3.6-27B-FP8 (57.3%), Opus-distilled Qwen3.6-35B-A3B (52.0%), stock Qwen3.6-35B-A3B (48.3%).
- [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md) — cross-platform KV cache capacity analysis; Qwen3.6-35B-A3B's hybrid architecture is the cleanest "KV cache is a rounding error" case in the lineup.
