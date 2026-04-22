# Gemma-4-31B-it Q8 on SWE-bench Lite — RTX Pro 6000

**Headline: 69 of 300 resolved — 23.0% — using stock Gemma-4-31B-it Q8_0
served via llama.cpp + llama-swap, driven by SWE-agent v1.1.0 in
function-calling mode.**

This lands **25 pp below the stock Qwen3.6-35B-A3B Q8 run on the same
harness and hardware (48.3%)** and 29 pp below Opus-distilled Qwen3.6
(52.0%). Two failure modes compound:

1. **33% empty-patch rate** (100 of 300). Gemma frequently never
   produces a final patch — almost always because the agent loop
   terminates on three consecutive in-container bash-command timeouts
   (`exit_command_timeout`, 95 of 300 = 32%). The Qwen runs had
   effectively zero timeouts (1 per 300).
2. **Submitted patches pass the hidden tests only 34.5% of the time**
   (69 / 200 evaluated), vs Qwen3.6's ~52–55%. Even when Gemma does
   submit, its patches are lower quality on this workload.

Net: Gemma-4-31B is substantially weaker than Qwen3.6-35B-A3B as an
agentic bug-fixer on this hardware + scaffold, despite being a dense
31B model (vs Qwen's 35B-A3B MoE with only ~3B active parameters).

## Setup

- **Host:** NVIDIA RTX Pro 6000 Blackwell Workstation (96 GB GDDR7,
  sm_120). Same machine, same llama.cpp build as the Qwen runs.
- **Runtime:** llama.cpp behind llama-swap with the standard
  `gemma-4-31b-coder` alias — Gemma-4-31B-it Q8_0, `-c 524288 -np 2`
  (262K context per slot), `-fa on -ctk f16 -ctv f16 --no-mmap --jinja`,
  `-rea off` (Gemma 4 has no built-in thinking mode), sampling
  `temp 1.0, top-p 0.95, top-k 64, min-p 0, repeat-penalty 1.0` (Gemma
  defaults).
- **Agent:** SWE-agent v1.1.0 (`config/default.yaml`), LiteLLM `openai/`
  provider pointing at `http://localhost:8080/v1`, tool bundles
  `registry` + `edit_anthropic` + `review_on_submit_m`, parse mode
  `function_calling`, per-instance call limit 75, per-command
  `execution_timeout: 30`, `max_consecutive_execution_timeouts: 3`.
- **Dataset:** `SWE-bench/SWE-bench_Lite` test split, 300 instances.
- **Parallelism:** `--num_workers 4`. Note: llama-swap serves Gemma
  with `-np 2` (two server slots), so two workers were always queued
  at the model side — the Qwen runs used `-np 4` for full parallelism.
- **Sandbox:** SWE-ReX Docker containers per task
  (`swebench/sweb.eval.x86_64.*`).

## Results

### Headline

| bucket | count | % of 300 |
|---|---:|---:|
| **resolved ✓** | **69** | **23.0%** |
| unresolved (patch wrong) | 131 | 43.7% |
| empty_patch (no fix proposed) | 100 | 33.3% |
| error | 0 | 0% |
| **total** | **300** | **100%** |

Of the 200 instances with a real patch attempt: **69 / 200 = 34.5%**.

### Per-repo breakdown (sorted by resolution rate)

| repo | resolved | attempted | total | rate |
|---|---:|---:|---:|---:|
| mwaskom (seaborn) | 3 | 3 | 4 | **75%** |
| psf (requests) | 4 | 4 | 6 | **67%** |
| scikit-learn | 8 | 17 | 23 | 35% |
| astropy | 2 | 4 | 6 | 33% |
| django | 26 | 60 | 114 | 23% |
| matplotlib | 5 | 14 | 23 | 22% |
| sympy | 16 | 63 | 77 | 21% |
| pydata (xarray) | 1 | 3 | 5 | 20% |
| pytest-dev | 3 | 11 | 17 | 18% |
| sphinx-doc | 1 | 14 | 16 | **6%** |
| pallets (flask) | 0 | 2 | 3 | **0%** |
| pylint-dev | 0 | 5 | 6 | **0%** |

django dominates the split (114 of 300) and Gemma drops to 23% there —
vs Qwen3.6's 61% on the same repo. sympy (77 instances) lands at 21%
vs Qwen's 47%. The two big-repo losses account for most of the
headline gap. **Gemma goes 0-for-11 on pallets+pylint-dev combined**;
Qwen scored 3/8 on those.

### Agent-loop exit status (all 300)

| exit status | count | meaning |
|---|---:|---|
| submitted (clean) | 145 | agent converged & submitted |
| submitted (exit_cost) | 49 | hit 75-call limit, autosubmitted |
| submitted (exit_command_timeout) | 5 | submitted after hitting timeouts |
| submitted (exit_format) | 3 | submitted after parse errors |
| **exit_command_timeout** | **95** | **3 consecutive 30s bash timeouts, no patch** |
| exit_cost (no submit) | 3 | hit call limit with empty patch |

**95 hard timeouts with no patch is the dominant failure mode.**
The Qwen3.6 runs saw 1 per 300. Same harness, same `execution_timeout`,
same hardware. The most plausible causes:

1. Gemma emits bash commands that individually exceed 30s (runaway
   `grep`/`find` over the full repo; unbounded `python -c` blocks).
2. On first timeout, Gemma retries a nearly-identical command rather
   than adapting, tripping all 3 consecutive timeouts.

A trajectory-level audit of a handful of `exit_command_timeout`
instances would confirm which.

### Wall-clock

- **Agent run:** ~27.5 h with 4 workers (~10–14 instances/hr,
  slowing toward the tail on sympy-heavy batches).
- **Eval harness:** ~45 min to evaluate the 200 non-empty patches.

## Head-to-head vs Qwen3.6

All three runs: same 300-instance Lite test split, SWE-agent v1.1.0,
function calling, 75-call limit, 4 workers, same host.

| | **Gemma-4-31B** | **Qwen3.6 stock** | **Qwen3.6 Opus-distill** |
|---|---:|---:|---:|
| resolved / 300 | **69 (23.0%)** | 145 (48.3%) | 156 (52.0%) |
| empty_patch | 100 (33%) | 21 (7%) | 16 (5%) |
| resolved of attempts | 34.5% | 52.0% | 54.9% |
| hard timeouts | 95 | 1 | 0 |

### Per-repo deltas (Gemma − Qwen3.6 stock)

| repo | Gemma | Qwen stock | Δ resolved |
|---|---|---|---:|
| django | 26 / 114 | 63 / 104 | **−37** |
| sympy | 16 / 77 | 34 / 73 | **−18** |
| matplotlib | 5 / 23 | 12 / 23 | −7 |
| scikit-learn | 8 / 23 | 12 / 21 | −4 |
| pytest-dev | 3 / 17 | 7 / 15 | −4 |
| pylint-dev | 0 / 6 | 3 / 6 | −3 |
| sphinx-doc | 1 / 16 | 2 / 16 | −1 |
| astropy | 2 / 6 | 3 / 6 | −1 |
| pydata | 1 / 5 | 2 / 3 | −1 |
| psf | 4 / 6 | 5 / 6 | −1 |
| mwaskom | 3 / 4 | 2 / 4 | +1 |
| pallets | 0 / 3 | 0 / 2 | 0 |
| **total** | **69** | **145** | **−76** |

Gemma matches or beats Qwen on exactly one repo (`mwaskom`, n=4).
Everywhere else it loses, with django and sympy driving the bulk of
the 76-instance gap.

## Caveats

1. **Single trial.** Stochastic sampling + agent-loop dynamics imply
   ±3 pp on re-run. The 25 pp gap against Qwen3.6 is far too large to
   be trial noise.
2. **The `-np 2` server config bottlenecked throughput but not
   resolution quality.** Bash-command timeouts run inside the Docker
   sandbox and are independent of model-server latency. A faster
   server would finish the run in less wall-clock but wouldn't change
   the 23.0% headline.
3. **Gemma has no thinking mode (`-rea off`).** Comparing against
   Qwen3.6 with `--reasoning-budget 16384` is apples-to-oranges in one
   sense, but thinking is part of Qwen3.6's advertised capability set
   and it's the stock deployment. Treat the comparison as "stock
   config vs stock config," not "matched inference budget."
4. **Call-limit 75 applies to both models.** Raising it would likely
   help Gemma more than Qwen (since Gemma hits `exit_cost` less often
   but `exit_command_timeout` much more; a longer budget only helps
   the former). Bumping `execution_timeout` from 30 → 60 s and
   reducing timeout-retry loops in the tool prompt would likely matter
   more.
5. **Lite ≠ Verified.** Same models typically lose ~10 pp on Verified.

## Reproducing

```bash
# Prereqs: docker + llama-swap running on :8080 with the
# gemma-4-31b-coder alias.
sweagent run-batch \
    --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --agent.model.name openai/gemma-4-31b-coder \
    --agent.model.temperature 1.0 --agent.model.top_p 0.95 \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite_gemma --num_workers 4

python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite --split test \
    --predictions_path experiments/sweagent_lite_gemma/preds.json \
    --run_id gemma4-31b-full --max_workers 4 --cache_level instance \
    --report_dir experiments/sweagent_lite_gemma/eval_full
```

Final report: `sweagent_lite_gemma.gemma4-31b-full.json` at the repo
root.
