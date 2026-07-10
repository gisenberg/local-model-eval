# Qwen3.6 on RTX Pro 6000 — deep dives

This doc collects the Qwen3.6-family experiments run on the RTX Pro 6000 Blackwell Workstation (96 GB, sm_120) that don't fit naturally in the model-agnostic tier list ([MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md)). Five experiments, one per section:

1. [**RULER long-context eval on Qwen3.6-35B-A3B**](#ruler-long-context-eval-qwen36-35b-a3b) — YaRN ×2/×4 up to 1M tokens. **48/48 tasks pass**, static-YaRN short-context tax is invisible, Qwen3.6-35B-A3B is a true 1M-token model on this hardware.
2. [**Opus-Reasoning-Distilled Qwen3.6-35B-A3B on the coding bench**](#opus-reasoning-distilled-qwen36-35b-a3b--coding-bench-regression) — fine-tune **regresses** from stock's 21/22 to 10/22 on the 4-benchmark suite, independent of `-rea on/off`. Useful for agentic bug-fixing (separately measured on SWE-bench Lite at +3.7 pp vs stock) but lost on from-scratch code generation.
3. [**NVFP4 vs FP8 vs BF16 on Qwen3.6-27B dense**](#precision-comparison-qwen36-27b-dense-nvfp4-vs-fp8-vs-bf16) — Unsloth's July 2026 dynamic NVFP4 release changes the baseline result: **113.2 tok/s with MTP-2**, 2.45× the non-speculative ModelOpt NVFP4 and 2.41× non-speculative vendor FP8, while retaining a 21/22 best-of-3 coding score. Prior hand-tuned NVFP4+MTP and FP8+DFlash configs remain faster.
4. [**Spec-decode method × k-sweep on Qwen3.6-27B FP8**](#spec-decode-method--k-sweep-qwen36-27b-fp8) — 7 configs measured on the same 4-bench harness. **dflash-k15 stays the production winner** at 197.5 tok/s, 22/22. Native MTP-1 (built into the FP8 weights, never deployed before) is a clean drafter-free fallback at 22/22 + 67.5 tok/s. FP8 KV cache is a net loss (uncalibrated scaling) and is incompatible with DFlash at the framework level.
5. [**Qwopus 3.6-27B v2 (dense Opus-distill) — coding bench acceptance run**](#qwopus-36-27b-v2-dense-opus-distill--coding-bench-acceptance-run) — new daily driver lands **22/22 best-of-3** (matches stock + FP8 DFlash), but **avg 16.4/22** — runs at temp 1.0 to dodge the card-documented `<think>`-loop failure mode, and even then 2/12 runs ran out the 15K token budget mid-output. Throughput **~48 tok/s** (matches the llama.cpp Q8_0 fallback). A different Opus-distill outcome than the 35B-A3B fine-tune in section 2 — quality ceiling holds, but variance is real.

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

Four-way comparison of Qwen3.6-27B at different precisions on the 4-bench coding suite, served via vLLM on RTX Pro 6000 Blackwell 96 GB. Different model from the 35B-A3B MoE above — this is the **dense VLM 27B** variant. The original BF16/FP8/ModelOpt-NVFP4 matrix was run in May 2026; the Unsloth dynamic NVFP4 row was added on July 10, 2026 from checkpoint revision `dd75bc4`.

## Precision results

| Variant | Weights | Best-of-3 | Avg | Tok/s | Size | Source |
|---|---|---|---|---|---|---|
| **BF16** (baseline) | `Qwen/Qwen3.6-27B` | 21/22 (95%) | 18.4/22 (84%) | 29 | 52 GB | HF release |
| **FP8** (vendor) | `Qwen/Qwen3.6-27B-FP8` | 21/22 (95%) | 20.0/22 (91%) | 47 | 29 GB | HF release |
| **NVFP4** (modelopt) | `mmangkad/Qwen3.6-27B-NVFP4` † | 22/22 (100%) | 18.6/22 (85%) | 46 | 29 GB | community, modelopt 0.42 |
| **Dynamic NVFP4 + MTP-2** | [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 21/22 (95%) | 16.7 raw / **18.3 adjusted** ‡ | **113.2** | 21.81 GiB | Unsloth, compressed-tensors |

† See [NVFP4 checkpoint note](#nvfp4-checkpoint-note) — we self-quantized with modelopt 0.43 but the export didn't load in vLLM 0.19.1, so we swapped to a community checkpoint with the same methodology.

‡ One Unsloth LRU sample scored 0/6 only because its generated tests patched `ttl_cache.time.monotonic` while the benchmark combines implementation and tests into one temporary module. Providing the module alias permitted by the repo methodology changes that sample to 5/6, moving the total average from 16.7 to 18.3. The remaining failure is genuine (`StopIteration` from too few mock timestamps). Raw results remain unmodified.

**Headline**: Unsloth's documented MTP-2 recipe is the new throughput winner in this baseline precision matrix. At 113.2 tok/s it is **2.45× the older non-speculative NVFP4**, **2.41× non-speculative FP8**, and **3.89× BF16**. Quality does not show a quantization cliff: 21/22 best-of-3 matches BF16 and FP8; the module-alias-adjusted 18.3/22 average matches BF16 (18.4) and the older NVFP4 (18.6) within this suite's sampling variance, though it remains below FP8's 20.0.

## Precision per-benchmark

Best/avg out of 3 runs at T=0.3, max_tokens=15000, max-model-len=16384.

| Benchmark | BF16 best/avg | FP8 best/avg | NVFP4 best/avg | Unsloth NVFP4 + MTP-2 best/avg |
|---|---|---|---|---|
| Expression Evaluator (5) | 5 / 4.0 | 5 / 4.3 | 5 / 4.3 | 5 / 3.3 |
| A* Pathfinding (6) | 5 / 4.7 | 5 / 5.0 | **6** / 3.3 * | 5 / 4.7 |
| LRU Cache w/ TTL (6) | 6 / 5.0 | 6 / 5.7 | **6 / 6.0** | 6 / 3.7 raw, **5.3 adjusted** ‡ |
| String Processor (5) | 5 / 4.7 | 5 / 5.0 | 5 / 5.0 | 5 / 5.0 |
| **Total** | **21 / 18.4** | **21 / 20.0** | **22 / 18.6** | **21 / 16.7 raw, 18.3 adjusted** |

\* NVFP4 A* run 1 produced no extractable code block (0/6). Run 2 was 4/6, run 3 was 6/6. Variance is real; the 100% best-of-3 is one lucky run, not a consistent quality lead.

## Precision methodology

- **Hardware**: RTX Pro 6000 Blackwell 96 GB, driver 580.126.09, CUDA 12.8 runtime.
- **Serving**: vLLM 0.19.1 with transformers built from main (required for `qwen3_5` arch).
- **Bench**: [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py) — identical 4-benchmark suite as `nvfp4_gemma31b_bench_v2.py`, 3 runs each, T=0.3. Raw outputs in [`experiments/nvfp4_qwen36_27b/`](../experiments/nvfp4_qwen36_27b/).
- **Quantization**: [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — NVIDIA Model Optimizer, `MAMBA_MOE_NVFP4_CONSERVATIVE_CFG` + `*linear_attn*` exclusion for Qwen3.5's gated-delta-net layers, calibrated on `abisee/cnn_dailymail` (128 samples, seq_len 2048) — matching NVIDIA's reference recipe for Qwen3-NVFP4 releases.
- **Thinking**: Qwen3.6 is a reasoning model. The bench strips `<think>...</think>` before extracting code blocks (extraction regex would otherwise glue scratchwork fragments into invalid Python).

### Unsloth dynamic NVFP4 run (July 10, 2026)

- **Runtime**: vLLM 0.24.0, PyTorch 2.11.0+cu130, FlashInfer 0.6.12, `nvidia-cutlass-dsl==4.5.2`; driver 580.142; RTX Pro 6000 at its configured 350 W power limit.
- **Serve config**: 16,384 max model length, BF16 KV cache, `--reasoning-parser qwen3`, and the model-card recipe `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
- **Model footprint**: 21.81 GiB checkpoint; vLLM reported 22.13 GiB resident model memory with MTP and a 54.72 GiB KV pool (863,436 cache tokens). The large total VRAM reservation is from `--gpu-memory-utilization 0.90`, not the weights.
- **Throughput**: mean 113.2 tok/s across all 12 coding generations, range 110.5–115.8. All 12 stopped normally; none exhausted the 15K output budget.
- **MTP acceptance**: 65,059 / 74,684 draft tokens accepted = **87.1%** overall. Position 0 accepted 92.1%; position 1 accepted 82.2%.
- **No-MTP control**: two warm Expression Evaluator generations measured 63.4 and 63.8 tok/s (63.6 mean). The new checkpoint + vLLM 0.24 path is **1.38×** the old checkpoint + vLLM 0.19 path without speculation; MTP-2 adds another **1.78×** on the new stack. Combined: 2.45×. Because both checkpoint and runtime changed, 1.38× is not a quant-kernel-only attribution.
- **Against prior optimized configs**: the existing `sakamakismile` ModelOpt NVFP4+MTP sweep measured 116.5 tok/s at k=3 and about 129-131 tok/s at k=5/7; FP8+DFlash k=15 remains the repo-wide leader at 197.5 tok/s. Thus the 2.45× headline is real against the old non-speculative NVFP4 row, but this documented k=2 recipe does **not** set a new tuned local speed record. A new-checkpoint k-sweep would be the next experiment.
- **Raw output**: [`experiments/nvfp4_qwen36_27b/unsloth_nvfp4_mtp2/results.json`](../experiments/nvfp4_qwen36_27b/unsloth_nvfp4_mtp2/results.json), with the generated test modules beside it.

#### Fresh-install caveat

The published install command resolved incompatible CUDA compiler components on July 10: CUDA 13.0 headers/runtime with NVVM/NVCC pieces from 13.2/13.3. FlashInfer first failed with incompatible headers, then with PTX 9.2 passed to a PTX 9.0 assembler. The working environment pins `nvidia-cuda-nvcc`, `nvidia-nvvm`, and `nvidia-cuda-crt` to `13.0.88`, exports the venv's `nvidia/cu13` as `CUDA_HOME`, and restores the missing `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks. These are packaging workarounds, not model changes.

Minimal reproduction after applying those environment fixes:

```bash
CUDA_HOME=~/venvs/qwen36-unsloth-nvfp4/lib/python3.12/site-packages/nvidia/cu13
PATH="$CUDA_HOME/bin:$HOME/venvs/qwen36-unsloth-nvfp4/bin:/usr/bin:/bin" \
CUDA_HOME="$CUDA_HOME" \
  ~/venvs/qwen36-unsloth-nvfp4/bin/vllm serve \
    /mnt/extended/gisenberg/models/qwen36-27b-unsloth-nvfp4 \
    --served-model-name qwen36-27b-unsloth-nvfp4-mtp2 \
    --host 127.0.0.1 --port 8091 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

~/venvs/qwen36-unsloth-nvfp4/bin/python \
  tools/nvfp4_qwen36_27b_bench.py \
  --port 8091 \
  --served-name qwen36-27b-unsloth-nvfp4-mtp2 \
  --output-dir experiments/nvfp4_qwen36_27b/unsloth_nvfp4_mtp2 \
  --temp 0.3
```

## NVFP4 checkpoint note

We produced our own NVFP4 checkpoint with `modelopt 0.43.0` using the config above. Resulting weights have the same `quantization_config.ignore` list (66 entries) as `mmangkad/Qwen3.6-27B-NVFP4` (67 entries, same pattern) — all `linear_attn` and some `self_attn` layers stay BF16, MLPs get NVFP4.

Our export loads into vLLM 0.19.1 but crashes during forward:

```
File ".../vllm/model_executor/models/qwen3_next.py", line 500, in forward
KeyError: 'weight_scale'
```

The `mmangkad` checkpoint (produced with `modelopt 0.42.0rc1.dev107`, nearly identical ignore list) loads and runs cleanly. The bench numbers above are from `mmangkad`'s checkpoint. The quality result — that a modelopt NVFP4 of this model can match BF16 on best-of-3 — still holds; we just didn't produce the checkpoint ourselves in this session.

Diagnosis is incomplete: it looks like modelopt 0.43 changed something in how quantized state is serialized for hybrid `linear_attn` + `self_attn` models that vLLM's `qwen3_next` loader doesn't handle yet. Not chased further in this session — pinning `nvidia-modelopt==0.42.*` would likely reproduce a working self-quantized checkpoint.

## Why the older ModelOpt NVFP4 didn't beat FP8 on throughput

On the May 2026 ModelOpt checkpoint, Blackwell FP4 tensor cores should have been ~2× FP8's throughput but we saw 46 vs 47 tok/s — effectively a tie. Two reasons:

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
- [`experiments/nvfp4_qwen36_27b/unsloth_nvfp4_mtp2/results.json`](../experiments/nvfp4_qwen36_27b/unsloth_nvfp4_mtp2/results.json)
- [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — reusable for other models
- [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py)

---

# Spec-decode method × k-sweep (Qwen3.6-27B FP8)

**TL;DR — dflash-k15 stays the production winner at 197.5 tok/s, 22/22 best-of-3. The model's built-in MTP head (which we'd never deployed) gives 67.5 tok/s at 22/22 with 93% acceptance and adds only ~1 GB to the FP8 weight footprint — useful as a drafter-free fallback. DFlash dominates throughput at every k, with sharply diminishing returns past k=11. FP8 KV cache is a net loss on this stack: uncalibrated scaling factors regress quality, and the BF16 drafter is incompatible with FP8 KV at the vLLM page-allocator level.**

## Why this run

The DFlash refresh (`vLLM PR-40898 + drafter 09196886`, [previous commit](#)) lifted coding score 21/22 → 22/22 at 199 tok/s but left two big knobs un-swept: the spec-decode `num_speculative_tokens` (DFlash card community data shows acceptance falling sharply past k≈10), and whether the FP8 weights' built-in MTP head — which `~/models-vllm/qwen36-27b-fp8/model.safetensors.index.json` carries 22 quantized `mtp.*` tensors for — could replace the external 3.3 GB DFlash drafter. Plus FP8 KV cache as a memory knob: vLLM recipes lists "256K ctx, KV FP8" as the verified Blackwell production config and we'd never tested it.

## Spec-decode setup

- **Hardware**: RTX Pro 6000 Blackwell 96 GB, driver 580.126.09.
- **vLLM**: `0.20.1rc1.dev33+g80561d6ce` (PR-40898 build, same as the production `qwen36-27b` daily driver).
- **Target weights**: `~/models-vllm/qwen36-27b-fp8` (vendor FP8 release, 28.9 GB on load).
- **Bench**: [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py), the same 4-task suite used for the precision sweep above. T=0.3, 3 runs each, 15K max tokens, thinking on (typical generation ~8-10K tokens including organic reasoning prose).
- **Common serve args**: `--max-model-len 262144 --gpu-memory-utilization 0.88 --enable-prefix-caching --max-num-seqs 256 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`. (`max_num_seqs` had to drop from vLLM's default 1024 — MTP-1 errors with `max_num_seqs (1024) exceeds available Mamba cache blocks (914)` since spec-decode adds Mamba slots per draft step.)
- **Backend**: `flash_attn` for non-fp8-kv variants, `flashinfer` for fp8-kv variants (flash_attn rejects `kv_cache_dtype=fp8` with `not supported`).
- **Orchestrator**: [`tools/qwen36_27b_mtp_sweep.py`](../tools/qwen36_27b_mtp_sweep.py) — boots vLLM with the variant's spec config, polls `/v1/models` until ready, runs the bench, scrapes `/metrics` for spec-decode counters, kills vLLM and the orphan `VLLM::EngineCore` worker, verifies VRAM is freed before moving on.

## Spec-decode results

| variant | best/22 | avg/22 | tok/s mean | range | speedup vs baseline | mean accepted | overall accept |
|---|:---:|:---:|---:|---|---:|---|---:|
| baseline (no spec dec) | 21 | 20.1 | 48.3 | 48.3-48.4 | 1.00× | — | — |
| **mtp-1** (native MTP head) | **22** | **21.1** | 67.5 | 65.0-70.5 | 1.40× | 0.93 of 1 | 93.0% |
| mtp-2 | 22 | 18.7 | 93.3 | 90.3-96.4 | 1.93× | 1.74 of 2 | 87.1% |
| dflash-k7 | 22 | 17.9 | 170.8 | 158-180 | 3.54× | 4.09 of 7 | 58.5% |
| dflash-k11 | 22 | 16.4 | 185.8 | 165-203 | 3.85× | 4.67 of 11 | 42.4% |
| **dflash-k15** (production) | **22** | 18.0 | **197.5** | 173-223 | **4.09×** | 4.99 of 15 | 33.3% |
| mtp-1-fp8kv | 21 ↓ | 18.7 ↓ | 75.0 | 70-76 | 1.55× | 0.93 of 1 | 93.1% |
| dflash-k11-fp8kv | — | — | — | — | — | — | — |

Per-position acceptance for the DFlash sweep (all FP8 target, BF16 drafter, single-stream coding bench):

| draft pos | k=7 | k=11 | k=15 |
|---:|---:|---:|---:|
| 0 | 89% | 87% | 87% |
| 2 | 65% | 60% | 59% |
| 5 | 41% | 36% | 35% |
| 8 | — | 23% | 22% |
| 10 | — | 17% | 16% |
| 14 | — | — | 8% |

Mean accepted tokens per draft step (the metric that actually pays for throughput): k=7 → 4.09, k=11 → 4.67, k=15 → 4.99. **k=15's extra 4 draft positions buy only 0.32 more accepted tokens** — sharply diminishing returns. k=11 is the throughput-per-drafter-FLOP sweet spot if drafter compute ever became a bottleneck (it doesn't on single-stream).

## Spec-decode findings

1. **Native MTP-1 works and adds only ~1 GB to weight footprint.** vLLM auto-detected the MTP head from `~/models-vllm/qwen36-27b-fp8`'s `model.safetensors.index.json`, logged `Detected MTP model. Sharing target model lm_head weights with the draft model`, and ran cleanly. Model loading footprint went from baseline 28.0 GB to 28.95 GB — confirming the head is built into the FP8 weights, not a separate file. 93% per-position acceptance gives 1.4× decode throughput at no quality cost. **It's the only variant whose avg score (21.1) beats baseline (20.1)**, although given run-to-run variance on 3 trials this is likely just lucky stochasticity rather than a method-level quality lift.

2. **MTP-2 is faster but quality variance widens.** 93.3 tok/s (1.93×) at 87% mean acceptance, but avg drops to 18.7. Best-of-3 still 22.

3. **DFlash dominates throughput at every k.** Even k=7 hits 170.8 tok/s — 2.5× MTP-2 — because the drafter is parallel (block diffusion: all k positions in one forward pass) where MTP is autoregressive (one MTP-layer call per draft token). On Blackwell + this hybrid arch, the parallel drafter is a much better throughput pick.

4. **k=15 is still the sweet spot for production** despite the diminishing returns. 197.5 mean tok/s confirms the existing dflash_pr40898 reference (~199). k=11 trades 6% throughput (185.8 vs 197.5) for 27% less drafter compute — only worth picking up if drafter VRAM/compute starts mattering (e.g., very long-context where drafter overhead grows with context).

5. **All spec-decode variants preserve 22/22 best-of-3.** Verifier-checked spec dec is lossless by construction. Avg-score variance across variants is run-to-run noise, not method-level quality drift; with only 3 trials per task, ±1-2 points is normal.

## FP8 KV cache: don't ship it

The vLLM recipes page lists `--kv-cache-dtype fp8` as part of the verified Blackwell production stack. We tried it on both the quality leader (mtp-1) and the throughput leader (dflash-k11). Both failed.

**mtp-1 + fp8-kv**: ran cleanly but regressed:
- best-of-3: 22 → **21** (one test flipped from pass to fail)
- avg: 21.1 → **18.7** (real quality drop on multiple runs)
- tok/s: 67.5 → 75.0 (+11%, the only positive)

vLLM logged the cause at startup:
```
Checkpoint does not provide a q scaling factor. Setting it to k_scale.
Using KV cache scaling factor 1.0 for fp8_e4m3.
Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention.
This may cause accuracy issues.
```

The vendor FP8 release ships weight scales but not KV/prob scales. Without per-tensor calibration, FP8 quantize-on-write costs more accuracy than the +11% throughput is worth. Picking up FP8 KV would require calibrating Q/K/V scales on a representative dataset and re-uploading — out of scope for a knob sweep.

**dflash-k11 + fp8-kv**: didn't even start. vLLM v1's KV-cache page allocator asserted:
```
File "vllm/v1/core/kv_cache_utils.py", line 1030, in unify_kv_cache_spec_page_size
    assert new_spec.page_size_bytes == max_page_size
AssertionError
```

The DFlash drafter is BF16. Its KV pages and the FP8 target's KV pages have different byte widths, and vLLM v1 enforces unified page size across draft + target. **Framework-level incompatibility**, not a config issue. Can't be patched without modifying vLLM's KV manager or producing an FP8-quantized DFlash drafter (z-lab doesn't ship one).

**Conclusion**: don't pursue FP8 KV on this model until either (a) the FP8 release picks up calibrated KV scales, or (b) vLLM relaxes the page-size assertion for spec-decode pairs.

## Spec-decode recommendation

- **Keep `qwen36-27b` as dflash-k15.** No reason to change. 22/22 best-of-3, 197 tok/s mean, well above any alternative.
- **Add `qwen36-27b-mtp1` as a drafter-free fallback** in `~/git/gisenberg/opencode-config/hosts/rtxpro6000/llama-swap.yaml`. Same FP8 weights, no DFlash dependency, no z-lab pre-1.0 churn. Gives 22/22 best-of-3 + 67.5 tok/s steady. Useful when:
  - z-lab pushes a regressed drafter checkpoint and we need a known-good
  - we want to ship the same `qwen36-27b-fp8` weights to a host without the 3.3 GB drafter file (e.g., DGX Spark, where every GB matters)
  - sanity-checking whether a quality issue is the target or the drafter

  Suggested entry:
  ```yaml
  "qwen36-27b-mtp1":
    name: "Qwen3.6-27B FP8 + native MTP-1 (drafter-free fallback)"
    description: "Same FP8 weights, model's built-in MTP head; 22/22 coding, 67 tok/s, no DFlash dependency"
    env:
      - "PATH=/home/gisenberg/venvs/dflash-pr40898/bin:..."
      - "CUDA_HOME=/home/gisenberg/venvs/dflash-pr40898/lib/python3.12/site-packages/nvidia/cu13"
    ttl: 600
    cmd: |
      /home/gisenberg/venvs/dflash-pr40898/bin/vllm serve
      /home/gisenberg/models-vllm/qwen36-27b-fp8
      --served-model-name qwen36-27b-mtp1
      --host 127.0.0.1 --port ${PORT}
      --max-model-len 262144
      --gpu-memory-utilization 0.88
      --enable-prefix-caching
      --attention-backend flash_attn
      --max-num-batched-tokens 32768
      --max-num-seqs 256
      --enable-auto-tool-choice --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
  ```
- **Don't bother with mtp-2.** Throughput is well below DFlash and avg quality variance is wider than mtp-1.
- **Drop FP8 KV from the consideration set** until upstream ships calibrated FP8 KV scales for Qwen3.5/3.6.

## Spec-decode files

- [`tools/qwen36_27b_mtp_sweep.py`](../tools/qwen36_27b_mtp_sweep.py) — orchestrator (one variant per invocation; supports `all-round1`)
- [`experiments/qwen36_27b_mtp_sweep/{baseline,mtp-1,mtp-2,dflash-k7,dflash-k11,dflash-k15-rerun,mtp-1-fp8kv}/`](../experiments/qwen36_27b_mtp_sweep/) — per-variant `summary.json`, `results.json`, `metrics.txt`, `vllm.log`

## Spec-decode caveats

- **Best-of-3 is generous methodology.** With only 3 runs at T=0.3, the avg column has ±1-2 points of stochastic noise. The "MTP-1 beats baseline on avg" finding (21.1 vs 20.1) is within that noise band — don't read it as a quality lift, just confirmation that MTP-1 doesn't *cost* quality.
- **Throughput is generation-shape-dependent.** All numbers measured on this 4-task coding suite at ~8-10K tokens per generation with reasoning on. Real opencode workloads (mixed prompt sizes, tool-call interludes, much shorter responses for simple tasks) will see different acceptance distributions and likely lower means. The dflash_pr40898 reference doc has the prompt-shape variance discussion.
- **`enable_thinking: false` collapses these scores.** First baseline run with `--default-chat-template-kwargs '{"enable_thinking": false}'` (matching the production agent preset) scored 0/5 on Expression Evaluator with 1421-token output — model emitted impl only, no tests. Switched off for this comparison; the production agent's no-think preset is a different measurement axis (SWE-bench Lite captures it).
- **Single A/B point on FP8 KV.** mtp-1-fp8kv was the only fp8-kv variant that completed. Whether the regression is mtp-1-specific or a property of uncalibrated FP8 KV in general can't be distinguished from this data alone.

---

# Qwopus 3.6-27B v2 (dense Opus-distill) — coding bench acceptance run

**TL;DR — `Jackrong/Qwopus3.6-27B-v2-GGUF` Q8_0 is the new daily-driver pick on this host. Lands 22/22 best-of-3 on the 4-benchmark coding suite (matches stock Qwen3.6-27B FP8+DFlash) at ~48 tok/s, but avg is only 16.4/22 because 2/12 runs ran out the 15K-token max trying to escape a `<think>`-block loop — the failure mode the model card explicitly warns about for low-temperature sampling. We run this model at `temp=1.0` per the card and still see the cliff. A* and string_processor are rock-solid; expression_evaluator and LRU cache are where the variance lives.**

## Qwopus-coding what we tested

- **Model**: [`Jackrong/Qwopus3.6-27B-v2-GGUF`](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-GGUF) at Q8_0 (28.6 GB) + `mmproj.gguf` (931 MB).
- **Base model**: dense `Qwen/Qwen3.6-27B` (hybrid linear_attn + full_attention, same arch family as the 35B-A3B Opus-distill in section 2, but **dense** not MoE).
- **What the fine-tune is**: Trace Inversion distill of Claude Opus 4.6 + 4.7 chain-of-thought traces (datasets: `Jackrong/Claude-opus-4.6-TraceInversion-9000x`, `Jackrong/Claude-opus-4.7-TraceInversion-5000x`). Card-claimed MMLU-Pro subset 87.43% — author flags this as a 200-sample harness, treat as smoke test.
- **Why we ran this**: promoted to daily driver on 2026-05-22, replacing the `qwen36-27b` FP8+DFlash vLLM entry as the routed default. Needed an acceptance number against the same 4-bench harness we've used for every other Qwen3.6 variant on this host to know whether the swap costs quality.

## Qwopus-coding methodology

Standard 4-benchmark coding suite (`tools/nvfp4_qwen36_27b_bench.py`) — same prompts as section 2, section 3, section 4. 3 runs per benchmark, scored per-test by `pytest -v` (partial credit per run, not all-or-nothing).

Served via llama-swap on port 8080 (the production path, not a one-off vLLM on 8090): the entry exercised is exactly the one opencode hits. llama.cpp CUDA build (`/home/gisenberg/llama-build/src/build/bin/llama-server`), `-c 524288 -np 2 -ngl 99 -fa on -ctk f16 -ctv f16 --no-mmap --jinja`, `--mmproj` wired in. Sampling: **`temp 1.0`** per the model card (top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0).

**Why temp=1.0 not 0.3.** Every prior Qwen3.6 row in this doc was scored at temp 0.3. The Qwopus model card explicitly warns: *"Greedy decoding (temp=0.1) forces the finetune to over-deliberate and loop inside the `<think>` block, whereas a higher temperature enables the model to utilize the full breadth of reasoning paths established during training."* Card-quoted bench range is 0.75–1.0. We took the upper end. This is a deliberate methodology change for this row only — Qwopus at 0.3 would be measuring a known-bad config, not its production-recommended config.

`max_tokens=15000` (the harness default). Thinking left on — that's the whole point of running this distill.

## Qwopus-coding results

| benchmark | best/expected | avg/expected | per-run breakdown |
|---|---:|---:|---|
| expression_evaluator | **5/5** | 4.0/5 | 0 / 0 / 5  (run1 length-capped at 15K, run2 produced bad code at 6.5K, run3 clean) |
| astar_pathfinding | **6/6** | 5.7/6 | 6 / 5 / 6 |
| lru_cache_ttl | **6/6** | 2.0/6 | 0 / 0 / 6  (run1 produced bad code at 6.3K, run2 length-capped at 15K, run3 clean) |
| string_processor | **5/5** | 4.7/5 | 5 / 4 / 5 |
| **total** | **22/22 (100%)** | **16.4/22 (74.5%)** | — |

(`expression_evaluator` run3 actually emitted 12 passing tests — the model wrote a 12-test suite when 5 were requested. The harness caps best-per-benchmark at `expected`, so it counts as 5/5.)

Throughput steady across all 12 runs at **47.6–48.1 tok/s** (single-stream, one slot of `-np 2`). That's within noise of the `qwen36-27b-fallback` baseline (49 tok/s on stock Qwen3.6-27B Q8_0) — Qwopus inherits the base's decode speed exactly.

### Where the variance comes from

Of the 12 runs, 4 scored zero:

| run | finish | tokens | what happened |
|---|---|---:|---|
| expression_evaluator run1 | `length` | 15000 | model spent the whole budget thinking, no code block ever emitted |
| expression_evaluator run2 | `stop` | 6497 | model emitted code that failed pytest collection / test asserts |
| lru_cache run1 | `stop` | 6330 | model emitted code that failed pytest collection / test asserts |
| lru_cache run2 | `length` | 15000 | model spent the whole budget thinking, no code block ever emitted |

Two distinct failure modes:

1. **Length-capped runs (2/12, ~17%).** The `<think>` block runs past the 15K max_tokens, generation gets cut off mid-reasoning, no `</think>` ever fires, no python code block exists to extract. This is the exact failure mode the card warns about — *"the reasoning-loop failure mode where earlier finetunes hit 78 empty patches"* — and we still see a residual rate of it at temp 1.0. The card says temp 1.0 *eliminates* this mode; in our 12-run sample it merely *reduces* it. Plausibly noise (1 failure per benchmark, not 3/3).

2. **Stopped-but-wrong runs (2/12, ~17%).** Model finished cleanly inside the token budget and emitted python code, but the code didn't pass tests. Different from the 35B-A3B Opus-distill failure mode (section 2), which was *pytest collection errors* from broken test files — here the tests collect and run, they just fail. This looks like ordinary "small model writes plausible-but-buggy code" variance, not a fine-tune-specific regression.

The other 8 runs all scored at or near max (4 perfect runs at 5/5 or 6/6, plus the 12/5 over-test outlier, plus 5/6 and 4/5). So **when the model does emit code, it usually emits good code**.

### Comparison to siblings

| variant | source | best | avg | tok/s |
|---|---|---:|---:|---:|
| Qwen3.6-27B FP8 + DFlash (vLLM) | [§ Spec-decode k-sweep](#spec-decode-method--k-sweep-qwen36-27b-fp8) | 22/22 | 21.0 | 197.5 |
| Stock Qwen3.6-27B Q8 (llama.cpp `-fallback`) | rebench (`temp 0.3`) | 22/22 | ~20.5 | 49.2 |
| **Qwopus 3.6-27B v2 Q8 (this row)** | this run (`temp 1.0`) | **22/22** | **16.4** | **48.1** |
| Qwen3.6-35B-A3B Opus-distill Q8 | [§ Opus-distill regression](#opus-reasoning-distilled-qwen36-35b-a3b--coding-bench-regression) | 10/22 | 10.0 | — |

**Quality ceiling held.** Qwopus matches the best-of-3 score of every healthy Qwen3.6-27B variant we've measured. The 35B-A3B Opus-distill regression in section 2 is *not* a general property of Opus-distillation — the dense 27B version of the same idea (different distiller, different dataset, different base) doesn't lose the coding ceiling.

**Avg dropped 4-5 points.** Stock variants average ~20.5-21.0; Qwopus averages 16.4 with a near-50/50 split between perfect runs and zero runs on the two harder benchmarks. The card's `<think>`-loop failure mode is the proximate cause; running with a much larger max_tokens budget (say 32K) would likely recover some of this.

**Throughput is base-determined.** Qwopus is a fine-tune of Qwen3.6-27B with identical architecture, so llama.cpp decode speed is identical to stock at the same quant. The 4× speed gap vs the vLLM FP8+DFlash row is the spec-decode stack we're forgoing — DFlash here is vLLM-only and the existing drafter was trained against stock Qwen3.6, not Qwopus.

## Qwopus-coding caveats

- **Single 3-run trial per benchmark, temp 1.0.** At temp 1.0 with reasoning on, run-to-run variance is structurally larger than at temp 0.3 — the avg column has wider error bars than every other row in this doc. 22/22 best is a real ceiling; 16.4/22 avg is one sample of a noisy distribution.
- **Methodology change vs prior rows.** Prior Qwen3.6 numbers in this doc are temp 0.3. Comparing Qwopus's avg directly to those is unfair — we'd need to re-run Qwopus at temp 0.3 to apples-to-apples it, but the card's `<think>`-loop warning means that's measuring a known-bad config. Best-of-3 (22/22) is the only number that compares cleanly across both temps.
- **`max_tokens=15000` is a real ceiling.** Two runs hit it. A 32K budget would likely lift the avg; if you re-run this bench later, consider bumping that knob and noting the change.
- **Author's MMLU-Pro 87.43% claim is a 200-sample harness** (author-acknowledged). The coding bench is a different axis — code quality + valid pytest emission — not general-knowledge recall.

## Qwopus-coding recommendation

**Keep Qwopus as the daily driver alias** (`opencode.json: "model": "rtxpro6000/qwopus36-27b-v2-q8"`). The ceiling holds, the throughput matches the llama.cpp Q8 baseline, and the variance hits are diagnosable (length cap on hard benchmarks with deep reasoning). For real opencode workflows the model gets to retry on a bad emit, so a 75% per-run hit rate translates to a much higher session-level success rate.

**Route speed-critical work to `qwen36-27b` instead.** The vLLM FP8+DFlash entry is still wired up and runs at 197 tok/s — 4× the daily driver. Coding bench numbers are equivalent at best-of-3; the difference is whether the latency budget matters.

**Don't try to retrofit DFlash to Qwopus.** Our drafter (`qwen36-27b-dflash-drafter-v2026-04-27`) was trained against stock Qwen3.6. Qwopus is a separate distillation — same tokenizer, drifted output distribution — so even llama.cpp's generic `--model-draft` would suffer reduced acceptance. The clean path to spec-decode on this daily driver is a Qwopus-trained drafter, which doesn't exist yet.

## Qwopus-coding reproducing

```bash
# Download (29.5 GB total: Q8_0 + mmproj):
mkdir -p ~/models/qwopus36-27b-v2-q8
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Jackrong/Qwopus3.6-27B-v2-GGUF \
    Qwopus3.6-27B-v2-Q8_0.gguf mmproj.gguf \
    --local-dir ~/models/qwopus36-27b-v2-q8

# Confirm llama-swap sees the alias (the entry is in opencode-config/hosts/rtxpro6000/llama-swap.yaml):
curl -sS http://127.0.0.1:8080/v1/models | jq '.data[].id'

# Run the 4-benchmark coding suite at the card-recommended temp:
cd ~/git/gisenberg/local-model-eval
/home/gisenberg/venvs/dflash-pr40898/bin/python3 tools/nvfp4_qwen36_27b_bench.py \
    --port 8080 --served-name qwopus36-27b-v2-q8 \
    --output-dir experiments/qwopus36_27b_v2_q8 \
    --temp 1.0
```

Per-run output (`<bench>_run<N>_test.py`) + JSON roll-up (`results.json`) lands in `experiments/qwopus36_27b_v2_q8/`. The `--temp` flag was added to `tools/nvfp4_qwen36_27b_bench.py` in the same commit that introduced this section (the harness defaulted to `temp 0.3` before; default is unchanged, so prior rows in this doc are reproducible verbatim).

---

# See also

- [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md) — tier list for this platform; Qwen3.6-35B-A3B is the daily-driver pick.
- [SWEBENCH_LITE_RTXPRO6000.md](SWEBENCH_LITE_RTXPRO6000.md) — four-model SWE-bench Lite comparison including Qwen3.6-27B-FP8 (57.3%), Opus-distilled Qwen3.6-35B-A3B (52.0%), stock Qwen3.6-35B-A3B (48.3%).
- [CONTEXT_CAPACITY.md](CONTEXT_CAPACITY.md) — cross-platform KV cache capacity analysis; Qwen3.6-35B-A3B's hybrid architecture is the cleanest "KV cache is a rounding error" case in the lineup.
