# SWE-bench Lite: Stock Qwen3.6 vs Opus-Reasoning-Distilled (Pro 6000)

**Opus-distilled Qwen3.6 resolves more SWE-bench Lite instances than the
stock model: 156/300 (52.0%) vs 145/300 (48.3%). Same base architecture,
same agent scaffold, same sampling defaults, same hardware — the
distillation produced a real +3.7 pp gain on agentic bug-fixing even
though it regressed on from-scratch code generation.**

This is worth calling out because it's the opposite of what the
4-benchmark coding suite showed: the Opus-distilled model scored 10/22
there, vs the stock model's 21/22 with thinking. So the fine-tune is
a net regression on "write a complete module + tests from a prompt" and
a net improvement on "navigate a real codebase and surgically fix a
bug." The benchmarks measure different things, and the distillation
moved the model in opposite directions on each.

See individual writeups for methodology and caveats:
- Stock Qwen3.6: [`SWEBENCH_LITE_QWEN36_RTXPRO6000.md`](SWEBENCH_LITE_QWEN36_RTXPRO6000.md)
- Coding-bench comparison: [`OPUS_DISTILL_QWEN36_RTXPRO6000.md`](OPUS_DISTILL_QWEN36_RTXPRO6000.md)

## Head-to-head

Both runs: 300-instance SWE-bench Lite test split, SWE-agent v1.1.0 with
function calling, 75-call limit, 4 parallel workers, same sampling flags
(`temp 0.6, top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0`,
`-rea on --reasoning-budget 16384`), same llama.cpp CUDA build
(commit `a279d0f`).

| | **stock** | **opus-distilled** | **Δ** |
|---|---:|---:|---:|
| resolved (headline) | **145 / 300 (48.3%)** | **156 / 300 (52.0%)** | **+11 / +3.7 pp** |
| of patches evaluated | 145 / 279 (52.0%) | 156 / 284 (54.9%) | +2.9 pp |
| empty_patch (no fix attempted) | 21 | 16 | −5 |
| agent wall-clock (run 1) | 5 h 10 m | 5 h 40 m | +30 m |

Opus-distilled is slightly slower per instance — consistent with Claude-
style reasoning being a bit more verbose than stock Qwen's — and submits
more patches (fewer empty-patch instances). Both effects push its
resolution count up vs the stock model.

## Per-repo comparison

| repo | stock res/att | opus res/att | Δ (Opus − Stock) |
|---|---|---|---:|
| psf (requests) | 5/6 (83%) | 3/6 (50%) | **−2** |
| pydata | 2/3 (67%) | 2/5 (40%) | 0 res, −27 pp rate* |
| django | 63/104 (61%) | **67/102 (66%)** | **+4** |
| scikit-learn | 12/21 (57%) | **16/23 (70%)** | **+4** |
| matplotlib | 12/23 (52%) | 11/23 (48%) | −1 |
| astropy | 3/6 (50%) | 3/6 (50%) | 0 |
| mwaskom | 2/4 (50%) | 2/4 (50%) | 0 |
| pylint-dev | 3/6 (50%) | 3/6 (50%) | 0 |
| pytest-dev | 7/15 (47%) | **8/16 (50%)** | +1 |
| sympy | 34/73 (47%) | **37/74 (50%)** | **+3** |
| sphinx-doc | 2/16 (13%) | **3/16 (19%)** | +1 |
| pallets | 0/2 (0%) | 1/3 (33%) | +1 |
| **total** | **145 / 279** | **156 / 284** | **+11** |

\* pydata has more "attempted" entries in the Opus run because the
Opus run submitted patches on instances that the stock run left as
`empty_patch`; the denominator shift doesn't mean Opus is strictly
worse there.

Opus's wins are concentrated in **big-repo instances** where
codebase navigation matters: django (+4), scikit-learn (+4), sympy (+3).
Stock wins on the tiny psf sample (6 instances) where variance is high.
The other repos are statistical ties.

## Why these benchmarks disagree about the same fine-tune

From our earlier coding-bench writeup
[`OPUS_DISTILL_QWEN36_RTXPRO6000.md`](OPUS_DISTILL_QWEN36_RTXPRO6000.md),
Opus-distilled regressed to 10/22 because:

1. **astar_pathfinding 0/6:** generated test file used `List[Tuple[int, int]]`
   type hints without importing `typing`. Pytest collection error.
2. **lru_cache_ttl 0/6:** generated test file used `mock.patch('ttl_cache.x')`
   with a module name that didn't match the impl module. 6× `ModuleNotFoundError`.

Both failures are **test-code quality bugs** — the model writes
plausible impl + flawed test suite. The stock model is better at
remembering to import `typing`, matching module names in `mock.patch()`
strings, and generally writing self-consistent file pairs.

**SWE-bench doesn't exercise any of that.** The benchmark provides its
own hidden tests; the agent only has to produce an impl-side patch.
The distillation's strength — longer, more deliberate reasoning traces
— helps with codebase navigation and root-cause location, which is
exactly what SWE-bench rewards. The regression axis never comes up.

Corollary: picking a model by coding-bench score is misleading if the
downstream task is agentic. And picking by SWE-bench is misleading if
you're going to ask the model to write standalone modules with tests.

## Recommendation

- **For opencode daily-driver coding (write-from-scratch + tests):**
  keep stock `qwen36-35b-a3b-coder`. The 21/22 coding-bench score is
  real; the SWE-bench gain from Opus-distilled is smaller than the
  coding-bench loss.
- **For whole-repo agent work via sweagent, aider, OpenHands, etc.:**
  Opus-distilled is modestly better. Use the `qwen36-opus-distill-q8`
  alias that's already in the llama-swap config.
- **Don't expect a single "best model" across benchmarks.** The two
  variants genuinely have different tuning. Picking by workload is
  the right move.

## Caveats

1. **Single trial.** Both runs would shift ±2–3 pp on re-run. The +3.7
   pp Opus lead is larger than single-trial noise but not by a huge
   margin. Re-running with a different seed would firm up the number.
2. **Call-limit 75 constrains both models equally.** 65%+ of resolved
   patches came from exit_cost autosubmits. Raising to 150 might reveal
   a different gap between the models.
3. **Anthropic ToS note:** the Opus-distilled dataset was assembled
   from Claude outputs, which violates Anthropic's commercial terms.
   Using an already-distributed model is a grayer zone. Research /
   personal use only.

## Reproducing

```bash
# Both configs assume docker + llama-swap running, SWE-agent v1.1.0
# installed, and the qwen36-opus-distill-q8 alias present in
# llama-swap.yaml.

# Stock:
sweagent run-batch --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite --num_workers 4

# Opus-distilled (overrides the model name at CLI level):
sweagent run-batch --config /home/gisenberg/tools/sweagent-rtxpro6000.yaml \
    --agent.model.name openai/qwen36-opus-distill-q8 \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --output_dir experiments/sweagent_lite_opus --num_workers 4

# Evaluation of merged predictions (must run inside `sg docker -c` wrapper
# if your shell dropped docker group membership across reboots):
sg docker -c "python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite --split test \
    --predictions_path <output_dir>/preds.json \
    --run_id <run-id> --max_workers 4 --cache_level instance \
    --report_dir <output_dir>/eval"
```
