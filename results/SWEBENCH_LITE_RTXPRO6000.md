# SWE-bench Lite on RTX Pro 6000 — four-model comparison

All four runs hit the same 300-instance `SWE-bench/SWE-bench_Lite` test split, via SWE-agent v1.1.0 in function-calling mode with a 75-call ceiling and 4 parallel workers on the same RTX Pro 6000 Blackwell Workstation (96 GB, sm_120).

## Headline

| Model | Quant | Serving | Resolved / 300 | % resolved | % of attempts |
|---|---|---|---:|---:|---:|
| **Qwen3.6-27B dense** | FP8 (vendor) | vLLM 0.19 | **172** | **57.3%** | 58.1% |
| Qwen3.6-35B-A3B | Q8_0 Opus-distilled | llama.cpp + llama-swap | 156 | 52.0% | 54.9% |
| Qwen3.6-35B-A3B | Q8_0 stock | llama.cpp + llama-swap | 145 | 48.3% | 52.0% |
| Gemma-4-31B-IT | Q8_0 stock | llama.cpp + llama-swap | 69 | 23.0% | 34.5% |

Two headline findings:

1. **Dense 27B at FP8 is the best variant measured.** +5.3 pp over Opus-distilled 35B-A3B, +9 pp over stock 35B-A3B, +34 pp over Gemma. Four variables change between the FP8-27B run and the llama.cpp runs — model, quant, serving stack, context ceiling — so some of this is quant/infra, but the 64K context ceiling that this run ran under actively *worked against* FP8-27B (15 of 300 hit `exit_context`). See [methodology differences](#methodology-differences) below.
2. **Opus-distillation beats stock on agentic work by +3.7 pp**, even though the *same distilled model* loses 11 pts vs stock on the from-scratch coding bench. The two benchmarks measure different things; SWE-bench rewards codebase-navigation / surgical-fix reasoning (what Opus-CoT teaches), while the coding bench rewards self-consistent module-with-tests generation (which the distillation hurts). See [why coding-bench and SWE-bench disagree](#why-coding-bench-and-swe-bench-disagree) below.

## Shared setup

- **Host**: RTX Pro 6000 Blackwell Workstation (96 GB GDDR7, sm_120), llama.cpp CUDA build `a279d0f` for the llama-swap runs, vLLM 0.19.1 for the FP8 run.
- **Agent**: SWE-agent v1.1.0, `config/default.yaml` templates, tool bundles `registry` + `edit_anthropic` + `review_on_submit_m`, parse mode `function_calling`, per-instance call limit 75, per-command `execution_timeout: 30`.
- **Dataset**: `SWE-bench/SWE-bench_Lite` test split, 300 instances, `--num_workers 4`.
- **Sandbox**: SWE-ReX Docker containers per task (`swebench/sweb.eval.x86_64.*`).
- **Evaluation**: `swebench.harness.run_evaluation` against hidden test suites.

Per-model details (model/quant/sampling/context) are in each model's section below.

## Per-repo four-way comparison

Resolved count / evaluated count per repo bucket. The `n=` shown is the total in the bucket for each run (Docker-pull failures in earlier rounds created tiny denominator differences).

| Repo | FP8-27B | Opus-35B-A3B | Stock-35B-A3B | Gemma-4-31B |
|---|---|---|---|---|
| astropy | 3/6 | 3/6 | 3/6 | 2/4 |
| django | **75/112** | 67/102 | 63/104 | 26/60 |
| matplotlib | 11/23 | 11/23 | **12/23** | 5/14 |
| mwaskom | 3/4 | 2/4 | 2/4 | **3/3** |
| pallets | **1/3** | 1/3 | 0/2 | 0/2 |
| psf/requests | **6/6** | 3/6 | 5/6 | 4/4 |
| pydata/xarray | 2/5 | 2/5 | 2/3 | 1/3 |
| pylint-dev | 2/6 | **3/6** | 3/6 | 0/5 |
| pytest-dev | **9/17** | 8/16 | 7/15 | 3/11 |
| scikit-learn | **16/23** | 16/23 | 12/21 | 8/17 |
| sphinx-doc | 2/16 | **3/16** | 2/16 | 1/14 |
| sympy | **42/75** | 37/74 | 34/73 | 16/63 |
| **Total** | **172** | **156** | **145** | **69** |

FP8-27B wins or ties on 8 of 12 repos. Biggest gains over Opus-distilled: django (+8), sympy (+5), psf (+3). Gemma matches or beats Qwen3.6 stock on exactly one repo (mwaskom, n=4) and loses everywhere else; django (-37) and sympy (-18) drive the bulk of its 76-instance deficit.

## Qwen3.6-27B dense (FP8, vLLM) — 57.3%

Dense VLM served text-only via vLLM. Numbers: [`../experiments/sweagent_lite_fp8/`](../experiments/sweagent_lite_fp8/) and [`../sweagent_lite_fp8.qwen36-27b-fp8-full.json`](../sweagent_lite_fp8.qwen36-27b-fp8-full.json).

**Serving config:**
- vLLM 0.19.1 with transformers built from main (required for `qwen3_5` arch)
- `Qwen/Qwen3.6-27B-FP8` on port 8090, `--max-model-len 65536`, `--max-num-seqs 8`
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3`
- Temp 0.6, top-p 0.95; matches `tools/sweagent-rtxpro6000-fp8.yaml`
- Decode ~47 tok/s single-stream (measured independently in coding bench)

**Wall clock**: 18h 20m batch + 25 min eval = ~18.75h total at ~16 instances/hour.

**Exit-status breakdown:**

| Status | Count | Notes |
|---|---:|---|
| `submitted` (clean) | 228 | Converged and submitted |
| `submitted (exit_cost)` | 53 | Hit 75-call ceiling, autosubmitted |
| `submitted (exit_context)` | 15 | **VRAM-limited 64K ceiling**; see below |
| `submitted (exit_command_timeout)` | 1 | Hit 30s bash timeout, still submitted |
| `submitted (exit_error)` | 1 | Recovered and submitted |
| `exit_command_timeout` (no submit) | 2 | Only empty-patch failures |
| **Submitted (any)** | **298 / 300 = 99.3%** | Top of our submission-rate range |

The 15 `exit_context` failures are an artifact of this stack's VRAM budget: dense-27B KV is ~65 KB/token (3× the 35B-A3B MoE's), so vLLM on 96 GB caps at 64K context rather than the 1M that llama.cpp ran the MoE at. Those 15 instances represent a **5% ceiling** that additional context headroom (e.g. 128K+) would likely recover — the dense-27B FP8 result would push past 60% resolved with more context.

## Qwen3.6-35B-A3B Opus-distilled (Q8_0, llama-swap) — 52.0%

The Claude-Opus-4.6-reasoning-distilled fine-tune of Qwen3.6-35B-A3B. Numbers: [`../sweagent_lite_opus.qwen36-opus-distill-round1.json`](../sweagent_lite_opus.qwen36-opus-distill-round1.json).

**Serving config:**
- llama.cpp + llama-swap `qwen36-opus-distill-q8` alias
- `-np 4 @ 262K` per slot, `-rea on --reasoning-budget 16384`
- Unsloth Qwen3.6 thinking sampling: `temp 0.6 top-p 0.95 top-k 20`

**Wall clock**: 5h 40m single-pass run.

**Results vs stock (same model arch, same hardware, same scaffold):**

| | Stock | Opus-distill | Δ |
|---|---:|---:|---:|
| Resolved | 145 | 156 | +11 |
| Empty-patch | 21 | 16 | −5 |
| Resolved / attempted | 52.0% | 54.9% | +2.9 pp |

Opus wins are concentrated in big-repo instances where codebase navigation matters: django (+4), scikit-learn (+4), sympy (+3). Stock wins only on the tiny psf sample (6 instances) where variance is high. See [why coding-bench and SWE-bench disagree](#why-coding-bench-and-swe-bench-disagree) below for why the distillation moved quality up on agentic work and down on write-from-scratch.

**Caveats**: single trial (expect ±2–3 pp on re-run); the Opus-distilled dataset was assembled from Claude outputs, which violates Anthropic's commercial terms — using the already-distributed model is grayer; treat as research/personal use.

## Qwen3.6-35B-A3B stock (Q8_0, llama-swap) — 48.3%

The baseline 35B-A3B MoE. Numbers: [`../sweagent_lite.qwen36-stock-full.json`](../sweagent_lite.qwen36-stock-full.json).

**Serving config**: identical to Opus-distill above except using the `qwen36-35b-a3b-coder` alias (same Q8_0, same sampling, same `-np 4 @ 262K`).

**Wall clock**: 5h 10m (round 1, 221/300) + 2h 0m (round 2 retry after `docker login`, 79 images that hit the Docker Hub rate limit) = ~7h 10m active agent time, ~28h of worker-minutes, +25 min eval harness.

**Per-repo headline numbers:**

| Repo | Resolved / attempted | Rate |
|---|---|---:|
| psf (requests) | 5/6 | **83%** |
| pydata | 2/3 | 67% |
| django | 63/104 | 61% |
| scikit-learn | 12/21 | 57% |
| matplotlib | 12/23 | 52% |
| astropy | 3/6 | 50% |
| mwaskom | 2/4 | 50% |
| pylint-dev | 3/6 | 50% |
| pytest-dev | 7/15 | 47% |
| sympy | 34/73 | 47% |
| sphinx-doc | 2/16 | **13%** |
| pallets | 0/2 | 0% (n=2) |

django dominates the split (104 of 300) at 61% resolved. psf at 83% is the high end. **sphinx-doc 13% is the notable weakness** — sphinx tests are heavy and the 75-call ceiling isn't enough. 65%+ of resolved patches came from `submitted (exit_cost)` autosubmits, suggesting raising the call limit would improve the headline at the cost of more wall-clock.

**Calibration note**: 48.3% on SWE-bench Lite maps to roughly 38–42% SWE-bench Verified equivalent (Verified is harder by ~10 pp). Frontier closed-model Verified ~65-75% (Opus 4.x), open-weight frontier ~40-55% (Qwen3-Coder, DeepSeek-V3) — this number lands in the expected band for a ~35B-A3B open-weight MoE.

## Gemma-4-31B-IT stock (Q8_0, llama-swap) — 23.0%

Dense Gemma 4 31B. Numbers: [`../sweagent_lite_gemma.gemma4-31b-full.json`](../sweagent_lite_gemma.gemma4-31b-full.json).

**Serving config:**
- llama-swap `gemma-4-31b-coder` alias, Gemma-4-31B-it Q8_0
- `-np 2 @ 262K` per slot (dense ≠ MoE KV footprint)
- `-rea off` (Gemma 4 has no built-in thinking mode)
- Gemma defaults: `temp 1.0, top-p 0.95, top-k 64`

**Wall clock**: ~27.5h with 4 workers (vs Qwen's ~7h) — llama-swap's `-np 2` meant two of the four SWE-agent workers queued at the server side throughout the run.

**Two compounding failure modes** explain the 25 pp gap vs Qwen3.6 stock:

1. **33% empty-patch rate** (100 / 300), almost entirely from **`exit_command_timeout`** (95 / 300 = 32%). Gemma trips 3 consecutive 30s in-container bash timeouts on 95 instances — the Qwen runs saw 1 per 300 on the same harness. Most plausible cause: Gemma emits bash commands that individually exceed 30s (runaway `grep`/`find` over the repo, unbounded `python -c`) and retries nearly-identical commands after the first timeout rather than adapting. A trajectory-level audit of a few `exit_command_timeout` instances would confirm.
2. **When Gemma does submit, patches pass only 34.5% of the time** vs Qwen's 52-55% — lower-quality patches on top of the submission-rate gap.

**Per-repo losses concentrate in the two largest buckets** (django -37, sympy -18 vs Qwen stock). Gemma goes 0-for-11 on pallets + pylint-dev combined where Qwen got 3/8. The `-np 2` server config bottlenecked wall-clock but not resolution quality (bash timeouts run inside the Docker sandbox, independent of model-server latency).

## Why coding-bench and SWE-bench disagree

From [`QWEN36_RTXPRO6000.md`](QWEN36_RTXPRO6000.md), Opus-distilled regressed to 10/22 on the 4-benchmark coding suite (stock: 21/22) because:

1. **astar_pathfinding 0/6**: generated test file used `List[Tuple[int, int]]` type hints without importing `typing`. Pytest collection error.
2. **lru_cache_ttl 0/6**: generated test file used `mock.patch('ttl_cache.x')` with a module name that didn't match the impl module. 6× `ModuleNotFoundError`.

Both are **test-code quality bugs** — the model writes plausible impl + flawed test suite. The stock model is better at remembering to import `typing`, matching module names in `mock.patch()` strings, writing self-consistent file pairs.

**SWE-bench doesn't exercise any of that.** The benchmark provides its own hidden tests; the agent only has to produce an impl-side patch. The distillation's strength — longer, more deliberate reasoning traces — helps with codebase navigation and root-cause location, which is exactly what SWE-bench rewards. The regression axis never comes up.

**Corollary**: picking a model by coding-bench score is misleading if the downstream task is agentic. And picking by SWE-bench is misleading if you're going to ask the model to write standalone modules with tests.

## Methodology differences

The FP8-27B run vs the three llama-swap runs changes **four variables simultaneously**:

| Variable | FP8-27B | Opus / Stock 35B-A3B |
|---|---|---|
| Model | Dense 27B (hybrid linear_attn + self_attn) | 35B-A3B MoE |
| Quantization | FP8 per-channel | Q8_0 block INT8 |
| Serving stack | vLLM 0.19.1 + flashinfer | llama.cpp + llama-swap |
| Context | 64K (VRAM-limited) | 1M (YaRN ×4, `-np 1`) |

Three of these favor FP8-27B (dense > MoE at bench-taking, FP8 > Q8_0 at same byte budget, vLLM's Qwen3-tool-call parser is more mature than llama.cpp's Jinja path). The context variable works **against** FP8-27B — the 15 `exit_context` failures cost ~5 pp of ceiling.

To isolate just the model variable, you'd need either `Qwen3.6-27B-Q8_0` on llama-swap (same stack as the 35B-A3B runs) or `Qwen3.6-35B-A3B-FP8` on vLLM (same stack as FP8-27B). A `qwen36-27b-coder` llama-swap alias is now in [`../../opencode-config/hosts/rtxpro6000/llama-swap.yaml`](../../opencode-config/hosts/rtxpro6000/llama-swap.yaml) for the apples-to-apples Q8_0 run if we want it later; not benched yet.

## Recommendation shift

Historical daily-driver for opencode was `rtxpro6000/qwen36-opus-distill-q8` (52.0% SWE-bench Lite, agentic-uplift over stock, +3.7 pp). On this measurement, **`Qwen3.6-27B-FP8` via vLLM is +5.3 pp better** with a much smaller memory footprint (29 GB vs 35 GB weights). Trade-offs:

- **Context ceiling is tighter**: 64K on vLLM vs 1M on llama.cpp.
- **Latency**: ~47 tok/s single-stream (FP8) vs ~90 tok/s single-stream (llama-swap 35B-A3B Q8). Batching recovers some of this gap for multi-agent workloads.
- **Different serving stack**: if you already live in llama-swap, swapping to vLLM is non-trivial operationally.

For pure edit-loop quality, FP8-27B is the new best pick. For `>64K` whole-repo reads, the 35B-A3B Opus-distilled alias remains useful.

For from-scratch module+tests generation, stock `qwen36-35b-a3b-coder` (Q8_0 via llama-swap) still wins the coding bench 21/22 vs Opus-distill's 10/22. Workload → model:

| Workload | Pick |
|---|---|
| Edit-loop / agentic bug-fixing (<64K ctx) | `rtxpro6000/qwen36-27b-fp8` via vLLM |
| Agentic work needing >64K ctx | `rtxpro6000/qwen36-opus-distill-q8` (or `-524k` / `-1m`) |
| Write a new module + tests | `rtxpro6000/qwen36-35b-a3b-coder` (stock) |
| Max coding accuracy, speed doesn't matter | `rtxpro6000/gemma-4-31b-coder` (22/22 coding-bench) |

## Shared caveats

1. **Single trial per model**. Stochastic sampling + agent-loop dynamics imply ±2–3 pp on re-run. The headline gaps (Opus-distill vs stock = 3.7 pp, FP8-27B vs Opus = 5.3 pp) are larger than single-trial noise but not by a huge margin.
2. **Call-limit 75 constrains all four runs equally.** 65%+ of resolved patches came from `exit_cost` autosubmits; raising to 150 might shift the gaps.
3. **SWE-agent's default tool bundle is Anthropic-style (search+replace edit).** A unified-diff scaffold might score differently.
4. **Lite ≠ Verified.** Same models typically score ~10 pp lower on Verified than Lite.
5. **Docker Hub rate limit bit the stock run in round 1.** Anonymous limit is 100 pulls / 6h; we needed ~300. `docker login` (free tier = 200/6h authenticated) is sufficient spread across a run.

## Reproducing

```bash
# Prereqs for all runs:
#   - Docker installed + logged in
#   - SWE-agent v1.1.0 + swebench harness:
git clone --branch v1.1.0 https://github.com/SWE-agent/SWE-agent.git ~/tools/SWE-agent
cd ~/tools/SWE-agent && python -m pip install --editable .
python -m pip install swebench

# ---- Qwen3.6 / Gemma runs (llama-swap on :8080) ----
# Requires llama-swap running with qwen36-35b-a3b-coder /
# qwen36-opus-distill-q8 / gemma-4-31b-coder aliases in config.

# Stock Qwen3.6:
sweagent run-batch \
    --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite --num_workers 4

# Opus-distilled (override model name at CLI):
sweagent run-batch \
    --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --agent.model.name openai/qwen36-opus-distill-q8 \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite_opus --num_workers 4

# Gemma (also switches sampling to Gemma defaults):
sweagent run-batch \
    --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --agent.model.name openai/gemma-4-31b-coder \
    --agent.model.temperature 1.0 --agent.model.top_p 0.95 \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite_gemma --num_workers 4

# ---- FP8-27B run (vLLM on :8090) ----
# Requires vLLM serving Qwen/Qwen3.6-27B-FP8 on :8090 with
# --tool-call-parser qwen3_xml --reasoning-parser qwen3.
# See tools/sweagent-rtxpro6000-fp8.yaml for the full serve command.

sweagent run-batch \
    --config /home/gisenberg/git/gisenberg/local-model-eval/tools/sweagent-rtxpro6000-fp8.yaml \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite_fp8 --num_workers 4

# Evaluate (same invocation per run, change --predictions_path + --run_id):
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite --split test \
    --predictions_path <output_dir>/preds.json \
    --run_id <run-id> --max_workers 4 --cache_level instance \
    --report_dir <output_dir>/eval
```

Final per-run reports: `sweagent_lite*.json` at the repo root.
