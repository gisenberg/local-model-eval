# Ornith 1.0 35B GGUF on RTX Pro 6000

Tested 2026-07-01 on the RTX PRO 6000 Blackwell workstation.

## Candidate

- Repo: <https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF>
- File: `ornith-1.0-35b-Q8_0.gguf`
- Size: `36,903,138,880` bytes = 34.4 GiB
- Quant: Q8_0
- Base family: Qwen-style model with `<think>` reasoning and XML `<tool_call>` output.

The model card matters operationally. It recommends a serving runtime that can
parse Qwen XML tool calls:

- vLLM: `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- SGLang: `--tool-call-parser qwen3_coder --reasoning-parser qwen3`

The llama.cpp card example only shows a generic `llama-server` invocation. In
practice, llama.cpp sometimes returned Ornith's XML tool calls inside
`reasoning_content` instead of OpenAI `tool_calls`, which made SWE-agent requery
or fail format checks. I added a small OpenAI-compatible proxy to normalize that
case for this run.

## Run Config

Download:

```bash
hf download deepreinforce-ai/Ornith-1.0-35B-GGUF \
  ornith-1.0-35b-Q8_0.gguf \
  --local-dir /home/gisenberg/models/ornith-1.0-35b-q8_0
```

Server:

```bash
LD_LIBRARY_PATH=/home/gisenberg/llama-build/src-deepseek-v4/build/bin:/home/gisenberg/.micromamba/envs/cuda/lib \
/home/gisenberg/llama-build/src-deepseek-v4/build/bin/llama-server \
  -m /home/gisenberg/models/ornith-1.0-35b-q8_0/ornith-1.0-35b-Q8_0.gguf \
  --alias ornith-1.0-35b-q8 --host 127.0.0.1 --port 8091 \
  -c 1048576 -np 4 -ngl 99 -fa on -ctk f16 -ctv f16 \
  --no-mmap --jinja --reasoning-format deepseek \
  -rea on --reasoning-budget 8192 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
```

Tool-call proxy:

```bash
/home/gisenberg/.micromamba/envs/cuda/bin/python tools/ornith_tool_proxy.py \
  --host 127.0.0.1 --port 8092 --upstream http://127.0.0.1:8091
```

SWE-agent:

```bash
sweagent run-batch \
  --config tools/sweagent-rtxpro6000-ornith-35b.yaml \
  --instances.type swe_bench --instances.subset lite --instances.split test \
  --output_dir experiments/sweagent_lite_ornith_q8_proxy \
  --num_workers 4
```

Effective context:

| Setting | Value |
|---|---:|
| `-c` | 1,048,576 |
| slots | 4 |
| context per slot | 262,144 |
| KV cache | f16 K/V, fully in VRAM |
| Steady VRAM | ~56.4 GiB used |
| VRAM headroom | ~40.8 GiB free |

## Tool-Call Fix

`tools/ornith_tool_proxy.py` proxies `POST /v1/chat/completions` and converts
Ornith/Qwen XML blocks like:

```xml
<tool_call>
<function=bash>
<parameter=command>
python -m pytest
</parameter>
</function>
</tool_call>
```

into OpenAI-style `message.tool_calls`. It also coerces argument types from the
tool schema, so `view_range` becomes an array rather than a string.

The proxy converted 185 XML tool calls during the aborted SWE-bench run. After
adding it, the run had no fatal `exit_format` statuses, although SWE-agent still
issued occasional function-calling requeries when the model emitted no tool call
or multiple tool calls.

## SWE-bench Lite Partial Run

I aborted the full SWE-bench Lite run after about 3h47m of SWE-agent wall time.
The run had enough operational signal and was trending too slowly to justify a
full 300-instance pass.

Exit status at abort:

| Status | Count |
|---|---:|
| `submitted` | 13 |
| `submitted (exit_cost)` | 12 |
| `submitted (exit_error)` | 1 |
| `exit_cost` | 3 |
| Active at abort | 4 |
| Completed or exited | 29 |

Official SWE-bench harness result for the 29-prediction partial snapshot:

| Metric | Count |
|---|---:|
| Total Lite instances | 300 |
| Predictions submitted before abort | 29 |
| Non-empty patches evaluated | 26 |
| Resolved | 16 |
| Unresolved | 10 |
| Empty patches | 3 |
| Harness errors | 0 |
| Resolved / evaluated non-empty patches | 61.5% |
| Resolved / submitted predictions | 55.2% |

Important caveat: this is an early-slice partial, almost entirely Astropy and
Django. It is useful for operational triage, not as a full SWE-bench score.

Raw output:

- `experiments/sweagent_lite_ornith_q8_proxy/preds.partial-29.json`
- `experiments/sweagent_lite_ornith_q8_proxy/eval/sweagent_lite_ornith_q8_proxy.ornith-1.0-35b-q8-proxy-partial-29.json`
- `experiments/sweagent_lite_ornith_q8_proxy/run_batch_exit_statuses.yaml`

## Throughput and Behavior

llama.cpp timing from the run:

| Metric | Value |
|---|---:|
| Prompt eval mean | 4,318 tok/s |
| Prompt eval median | 4,420 tok/s |
| Decode mean | 219.7 tok/s |
| Decode median | 220.8 tok/s |
| Decode p10 / p90 | 206.9 / 230.7 tok/s |
| Slot release count | 1,845 |
| Max released slot tokens | 262,143 |
| Truncated slot releases | 2 |

Decode speed is excellent for a 35B Q8 model, and the 262K-per-worker context
fit comfortably. The problem was agent behavior:

- The model repeatedly hit the 75-call ceiling: 12 autosubmitted at `exit_cost`,
  plus 3 no-submit `exit_cost` failures.
- It generated many slow or repeated shell operations, producing frequent
  `CommandTimeoutError` events in the traces.
- At abort time, one active slot was still generating a runaway response beyond
  110K decoded tokens. Two earlier requests had already filled the 262K slot.
- The proxy fixed the fatal XML parsing mismatch, but did not fix verbosity,
  repetition, or poor stop behavior.

At 29 completed/exited instances in about 3h47m, the observed progress rate was
roughly 7.7 completed instances/hour. A naive full-run projection would land
near 39 hours before evaluation. That is far slower than the Qwen3.6 35B-A3B
llama.cpp runs in this repo, which completed full 300-instance passes in roughly
5-7 hours of active agent time.

## Read

Ornith 1.0 35B Q8 is technically runnable on this host with a large context:
four 262K slots fit in about 56 GiB VRAM, leaving about 41 GiB headroom. The
model is also fast at the token level.

The SWE-bench partial score was not bad: 16/26 non-empty patches resolved. The
problem is that the agent loop is inefficient and unstable enough that the good
per-token speed does not translate into good wall-clock throughput.

Practical verdict: do not promote this over the current RTX Pro 6000 coding
choices. Keep the proxy and config as a useful runbook for Qwen XML tool-call
models served through llama.cpp, but Ornith itself is not a daily-driver win on
this harness.

Compared with the current measured choices:

- Qwen3.6-27B FP8 remains the best full SWE-bench Lite result: 172/300.
- Qwen3.6-35B-A3B Opus-distilled remains the best long-context llama.cpp agentic
  choice: 156/300 with a full run in about 5h40m.
- Stock Qwen3.6-35B-A3B remains a stronger from-scratch coding model.
- DeepSeek V4 Flash q2 had better partial submitted-patch quality than Ornith
  on its aborted slice, but was much tighter on VRAM and slower per token.
