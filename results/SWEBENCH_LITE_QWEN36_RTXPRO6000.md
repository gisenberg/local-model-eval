# Qwen3.6-35B-A3B Q8 on SWE-bench Lite — RTX Pro 6000

**Headline: 145 of 300 resolved — 48.3% — using stock Qwen3.6-35B-A3B Q8
served via llama.cpp + llama-swap, driven by SWE-agent v1.1.0 in
function-calling mode with thinking enabled.**

Run in two passes: 5 h 10 min for 300 instances (79 failed to pull
Docker images due to Docker Hub's anonymous rate limit), then a 2 h
backfill for the 77 sympy + 2 sphinx-doc instances after `docker login`.
Of the 279 patches that were actually generated and evaluated, 145
passed the hidden tests (**52.0%**); treating the 21 empty-patch
instances as unresolved gives the **48.3%** headline against the full
300-instance test split.

This lands comfortably in the expected band for a ~35B open-weight MoE.
Frontier closed models on SWE-bench Verified are around 65–75%
(Opus 4.x), and open-weight frontier on Verified sits at 40–55% (Qwen3-
Coder, DeepSeek-V3). Lite is generally easier than Verified, so 48.3%
on Lite maps to a rough 38–42% Verified equivalent — consistent with
what you'd predict from parameter count alone.

See [`SWEBENCH_LITE_COMPARISON_RTXPRO6000.md`](SWEBENCH_LITE_COMPARISON_RTXPRO6000.md)
for a head-to-head against the Opus-distilled variant of the same
architecture.

## Setup

- **Host:** NVIDIA RTX Pro 6000 Blackwell Workstation (96 GB GDDR7, sm_120).
- **Runtime:** llama.cpp CUDA 13.2 build (commit `a279d0f`) behind
  llama-swap with the standard `qwen36-35b-a3b-coder` alias — Qwen3.6
  -35B-A3B Q8_0, `-np 4 @ 262K` per slot, Unsloth thinking sampling
  defaults (`temp 0.6, top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0`),
  `-rea on --reasoning-budget 16384`.
- **Agent:** SWE-agent v1.1.0 (`config/default.yaml`), LiteLLM `openai/`
  provider pointing at `http://localhost:8080/v1`, tool bundles
  `registry` + `edit_anthropic` + `review_on_submit_m`, parse mode
  `function_calling`, per-instance call limit 75.
- **Dataset:** `SWE-bench/SWE-bench_Lite` test split, 300 instances.
- **Parallelism:** `--num_workers 4`, exploiting llama-swap's `-np 4`
  slots. Aggregate throughput ~2–2.5× single-worker.
- **Sandbox:** SWE-ReX Docker containers per task from Docker Hub
  (`swebench/sweb.eval.x86_64.*`).

## Results

### Headline

| bucket | count | % of 300 |
|---|---:|---:|
| **resolved ✓** | **145** | **48.3%** |
| unresolved (patch wrong) | 134 | 44.7% |
| empty_patch (no fix proposed) | 21 | 7.0% |
| **total** | **300** | **100%** |

Of the 279 instances with a real patch attempt: **145 / 279 = 52.0%**.

### Per-repo breakdown

| repo | resolved | attempted | rate |
|---|---:|---:|---:|
| psf (requests) | 5 | 6 | **83%** |
| pydata | 2 | 3 | 67% |
| django | 63 | 104 | 61% |
| scikit-learn | 12 | 21 | 57% |
| matplotlib | 12 | 23 | 52% |
| astropy | 3 | 6 | 50% |
| mwaskom | 2 | 4 | 50% |
| pylint-dev | 3 | 6 | 50% |
| pytest-dev | 7 | 15 | 47% |
| sympy | 34 | 73 | 47% |
| sphinx-doc | 2 | 16 | **13%** |
| pallets | 0 | 2 | 0% (tiny sample) |

django dominates the test split (104 of 300); Qwen3.6 scores 61% there.
psf (`requests`) 83% is the highest resolution rate. **sphinx-doc 13%
is the notable weakness** — sphinx tests tend to be heavier than
average and the 75-call ceiling apparently isn't enough for those
instances.

### Agent-loop exit status

| exit status | count | meaning |
|---|---:|---|
| submitted | 133 + retry-submits | agent converged & submitted |
| submitted (exit_cost) | 71 + retry | hit 75-call limit, autosubmitted |
| exit_cost (no submit) | 12 | hit limit with empty patch |
| exit_format | 3 | couldn't parse model's action output |
| exit_command_timeout | 2 | in-container command stalled |
| (+ 79 DockerPullError on round 1, fixed by round 2) |

65%+ of resolved patches came from `submitted (exit_cost)` autosubmits,
suggesting raising the call limit would improve the headline at the
cost of more wall-clock.

### Wall-clock

- **Round 1 (221/300 completed):** 5 h 10 m with 4 workers.
- **Round 2 retry (79 instances):** 2 h 0 m with 4 workers (sympy images
  now cached from prior attempts + authenticated pulls).
- **Total active compute:** ~7 h 10 m wall, ~28 h of worker-minutes.
- **Eval harness:** ~25 min to run all 279 submissions through the
  hidden-test Docker images.

## Caveats

1. **Docker Hub rate limit bit us in round 1.** Anonymous limit is 100
   pulls / 6h; we needed ~300. `docker login` (free tier = 200/6h
   authenticated) is sufficient when spread across the run.
2. **Single trial per instance.** sweagent is stochastic (model
   sampling + agent-loop dynamics). Expect ±3 pp on the aggregate
   if re-run.
3. **Call-limit 75 is a soft cap, not a hard quality measurement.**
   Longer budgets would likely improve the number 2–5 pp at 2× wall-clock.
4. **SWE-agent's default tool bundle is Anthropic-style (search+replace
   edit).** A unified-diff scaffold might score differently.
5. **Lite ≠ Verified.** Same model typically scores ~10 pp lower on
   Verified than Lite.

## Reproducing

```bash
# Prereqs: docker installed + logged in, llama-swap running on :8080.
git clone --branch v1.1.0 https://github.com/SWE-agent/SWE-agent.git ~/tools/SWE-agent
cd ~/tools/SWE-agent && python -m pip install --editable .
python -m pip install swebench

# Full run (7 hours with 4 workers; see config at /home/gisenberg/tools/sweagent-rtxpro6000.yaml):
sweagent run-batch \
    --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite --num_workers 4

# Evaluate the merged predictions:
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite --split test \
    --predictions_path experiments/sweagent_lite/preds_merged.json \
    --run_id qwen36-stock-full --max_workers 4 --cache_level instance \
    --report_dir experiments/sweagent_lite/eval_full
```

Final report: `sweagent_lite.qwen36-stock-full.json` at the repo root.
