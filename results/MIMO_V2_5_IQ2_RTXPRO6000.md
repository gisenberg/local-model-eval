# MiMo V2.5 IQ2 on RTX Pro 6000

## Outcome

MiMo V2.5 UD-IQ2_XXS fits on one RTX Pro 6000 at a 262,144-token allocation when llama.cpp moves part of the weights to host memory.
The tested preset uses 92,585 MiB VRAM and leaves 4,657 MiB free.
A real 250,013-token OpenAI chat request completed without truncation and retrieved a needle placed at 75% depth.
Generation, preserved reasoning, strict tool arguments, and a multi-turn tool result all passed.
The original Q8 KV preset is not stable under repeated long-context decode on driver 595.84.
Use F16 K/V cache with flash attention and `-ub 128` for long-lived serving on this stack.

The default unbounded-thinking preset is not usable for agent work.
It reached a final answer on only one of four local coding tasks before the 16,384-token output ceiling.
A 4,096-token reasoning budget corrected that behavior and scored 22/22 after rescoring with the fixed module-path handling described below.

## Pinned inputs

The [official MiMo V2.5 model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) reports 310B total parameters, 15B active parameters, hybrid 5:1 sliding-window to global attention, three MTP layers, and a native 1M-token ceiling.
The exact GGUF files come from the [Unsloth MiMo V2.5 GGUF repository](https://huggingface.co/unsloth/MiMo-V2.5-GGUF).

| Component | Pin |
|---|---|
| Model | `unsloth/MiMo-V2.5-GGUF`, `UD-IQ2_XXS` |
| Hugging Face revision | `f7aff7868d5f79da58b505f84626d7a807393c37` |
| Quant size | 96,480,165,152 bytes, or 89.85 GiB |
| llama.cpp | `ea63b4d32ea1b66bdbe369be7f9443f6c00f8b31`, build 10198 |
| CUDA target | CUDA 13.0, `sm_120`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB |
| CPU and RAM | AMD EPYC 4585PX, 16 cores and 32 threads, 125 GiB RAM |

The model files are:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `00001-of-00003` | 5,953,760 | `361f5fbffbff4b5273a3984f1a488ba8160adc4a0879420a2495b66f17d59f84` |
| `00002-of-00003` | 49,525,684,032 | `e67f0ccf61539d5ba21a9c60fd8c37cff9e58eda129d97a9f2ac53d99872ecf4` |
| `00003-of-00003` | 46,948,527,360 | `35a32af82acc03ef94c0baf9e8c1f21c2eb9a1ba13f2991162e9fb134acaba3b` |

The IQ2_XXS quant was selected because the repository's `UD-Q2_K_XL` file is about 103 GB and cannot leave enough GPU space for a 250K context allocation on this card.
IQ2_XXS still requires partial CPU weight offload, but it leaves enough GPU space for Q8 KV cache and runtime buffers.

## Original quality and context serving configuration

```bash
rtk proxy env \
  LD_LIBRARY_PATH=/home/gisenberg/llama-build/src-mimo-v25-ea63b4d/build/bin:/home/gisenberg/.micromamba/envs/cuda/lib \
  /home/gisenberg/llama-build/src-mimo-v25-ea63b4d/build/bin/llama-server \
  -m /mnt/extended/gisenberg/models/mimo-v2.5-ud-iq2-xxs-f7aff786/UD-IQ2_XXS/MiMo-V2.5-UD-IQ2_XXS-00001-of-00003.gguf \
  --host 127.0.0.1 \
  --port 8091 \
  -c 262144 \
  -ngl auto \
  -fa on \
  -np 1 \
  --jinja \
  --fit on \
  --fit-target 4096 \
  --fit-ctx 262144 \
  -b 1024 \
  -ub 512 \
  -ctk q8_0 \
  -ctv q8_0 \
  --threads 16 \
  --threads-batch 32 \
  --reasoning-format deepseek \
  --reasoning-preserve \
  --reasoning-budget 4096 \
  --metrics
```

`--fit-ctx 262144` prevents auto-fit from silently shrinking the requested context.
`--fit-target 4096` reserves 4 GiB per GPU for operational headroom.
The server warns that mmap with CPU tensor overrides can be slower than `--no-mmap`.
Mmap was kept because it avoids a high transient host-memory copy while the 89.85 GiB model is loaded.

For the SWE-bench canary, the same total context allocation is divided into two 131,072-token slots with `-np 2`.
The two-slot server holds at about 92,625 MiB VRAM and leaves about 4,617 MiB free.

The stability-safe single-slot configuration changes `-ub 512 -ctk q8_0 -ctv q8_0` to `-ub 128 -ctk f16 -ctv f16`.
It retains `-fa on`.

## Long-context CUDA stability isolation

The full SWE-bench retry exposed a deterministic CUDA failure after repeated generations from a shared 100K-token prefix.
The isolation matrix used the same model, 262,144-token slot, llama.cpp commit, driver, reasoning settings, and OpenAI-compatible streaming API for every variant.
Each stable variant was asked to complete ten sequential generations of up to 4,096 tokens while reusing the prefix.

| Flash attention | KV cache | Ubatch | 100K prefill | Decode | Peak GPU memory | Result |
|---|---|---:|---:|---:|---:|---|
| On | Q8_0 | 512 | 1,138.35 tok/s | 71.8 to 72.1 tok/s | 92,673 MiB | Xid 8 after 6 complete responses and 22,974 completion tokens |
| On | Q8_0 | 128 | 442.81 tok/s | 74.3 to 74.4 tok/s | 93,151 MiB | Xid 8 during request 4 after 3 complete responses |
| Off | F16 | 512 | Did not finish | N/A | 97,215 MiB | CUDA OOM at 64,529 prompt tokens, no Xid |
| Off | F16 | 128 | 164.38 tok/s | 35.0 to 35.2 tok/s | 94,839 MiB | Passed 10/10, 40,960 completion tokens |
| On | F16 | 128 | 370.12 tok/s | 78.7 to 79.8 tok/s | 93,225 MiB | Passed 10/10, 38,391 completion tokens |

Both Q8 KV runs terminated at `cudaStreamSynchronize` with `CUDA error: the launch timed out and was terminated`.
The kernel recorded Xid 8 against the corresponding `llama-server` PID in both cases.
Reducing ubatch from 512 to 128 delayed prefill and did not prevent the failure.

Disabling flash attention and using F16 KV eliminated the watchdog at ubatch 128, but more than halved decode throughput.
Keeping flash attention enabled while changing only K/V cache from Q8_0 to F16 also eliminated the watchdog and improved decode throughput by about 6.5% over the Q8 ubatch-128 run.
That control isolates the current trigger to quantized Q8 KV handling under flash attention, or their interaction, rather than the model's IQ2 weight kernels or flash attention generally.
The F16 ubatch-512 non-flash OOM is a separate memory-capacity failure.

The exact tested stack was llama.cpp `ea63b4d32ea1b66bdbe369be7f9443f6c00f8b31`, NVIDIA open kernel module and driver `595.84`, GSP firmware `595.84`, CUDA 13.0, and kernel `7.0.0-28-generic`.
Raw server logs, kernel logs, GPU telemetry, commands, and API responses are stored in [`../experiments/mimo_v25_cuda_isolation_59584_ea63b4d/`](../experiments/mimo_v25_cuda_isolation_59584_ea63b4d/).

## API protocol checks

The API smoke test passed 7/7.

| Check | Result |
|---|---|
| Exact basic generation | Pass |
| Correct reasoning answer | Pass |
| Separate `reasoning_content` preserved | Pass |
| Exactly one native function call | Pass |
| Function name | Pass |
| Strict JSON arguments | Pass |
| Tool-result follow-up | Pass |

The model returned `MIMO_OK` as final content while placing its explanation in `reasoning_content`.
It called `lookup_record` once with exactly `{"record_id":"alpha-7"}` and used the returned value in the next turn.

Evidence is stored in [`../experiments/mimo_v25_iq2_xxs_rtxpro6000/api_smoke.json`](../experiments/mimo_v25_iq2_xxs_rtxpro6000/api_smoke.json).

## Long-context result

| Metric | Result |
|---|---:|
| Raw user-content tokens | 249,987 |
| API prompt tokens | 250,013 |
| Cached prompt tokens | 16 |
| Completion tokens | 7 |
| Needle depth | 75% |
| Exact retrieval | Pass, `739184` |
| Truncated | No |
| End-to-end time | 276.676 s |
| Server prompt-eval throughput | 904.78 tok/s |

Prefill started near 1,395 tok/s and declined gradually as the full-attention layers accumulated context.
The complete prompt evaluation averaged 904.78 tok/s.
The request finished with 4,567 MiB free VRAM.

Evidence is stored in [`../experiments/mimo_v25_iq2_xxs_rtxpro6000/long_context_250k.json`](../experiments/mimo_v25_iq2_xxs_rtxpro6000/long_context_250k.json).

## Throughput

The standard short-prompt throughput benchmark used one warmup and five timed 254-token generations at the full 262,144-token allocation.

| Metric | Result |
|---|---:|
| Warm restart load time | 9.0 s |
| VRAM after load | 92,585 MiB |
| Mean TTFT | 76.3 ms |
| Median TTFT | 75.3 ms |
| Mean decode | 100.26 tok/s |
| Median decode | 100.50 tok/s |
| Timed range | 99.48 to 100.98 tok/s |

The 32K coding runs decode faster because less KV capacity and fewer CPU-offloaded layers are needed.
Observed coding-run decode was about 117 to 124 tok/s with one active slot.
Two concurrent SWE-agent slots share the same GPU and have lower per-slot decode.

Evidence is stored in [`../experiments/rtxpro6000_bench_cuda/mimo-v2.5-ud-iq2-xxs.json`](../experiments/rtxpro6000_bench_cuda/mimo-v2.5-ud-iq2-xxs.json).

## Lightweight coding benchmark

| Preset | String | Expression | A* | LRU | Total |
|---|---:|---:|---:|---:|---:|
| Unbounded reasoning | 5/5 | 0/5 | 0/6 | 0/6 | 5/22 |
| Reasoning budget 4,096 | 5/5 | 5/5 | 6/6 | 6/6 | 22/22 |

The unbounded preset consumed all 16,384 output tokens without final content on Expression Evaluator, A*, and LRU Cache.
The 4,096-token budget reached final code on all four tasks.

MiMo generated 25 String Processor tests despite being asked for five.
The scorer previously summed all passing self-generated tests and produced the impossible raw value `34/22`.
The corrected scorer retains `raw_passed`, caps the scored pass count at each task's declared test count, and flags a test-count mismatch.

The LRU response initially appeared to fail 0/6 because the evaluator rewrote `from ttl_cache import TTLCache` to the local filename but did not rewrite `patch('ttl_cache.time.monotonic')`.
The evaluator now rewrites quoted module references consistently.
The saved LRU response then passes all six tests without regenerating model output.

The corrected rescore leaves Gemma 4, gpt-oss-120b, and both Qwen3.6-35B-A3B headline baselines unchanged.
It changes the available saved Qwen3-Coder-Next baseline from 15/22 to 17/22 because two LRU tests were also hidden by the same evaluator defect.

Evidence is stored in:

- [`../experiments/rtxpro6000_coding/mimo-v2.5-ud-iq2-xxs.json`](../experiments/rtxpro6000_coding/mimo-v2.5-ud-iq2-xxs.json)
- [`../experiments/rtxpro6000_coding/mimo-v2.5-ud-iq2-xxs-r4k.json`](../experiments/rtxpro6000_coding/mimo-v2.5-ud-iq2-xxs-r4k.json)

## Comparison with current local leaders

| Model | Quant | VRAM at tested context | Context | Decode | Coding |
|---|---|---:|---:|---:|---:|
| MiMo V2.5 | UD-IQ2_XXS | 92.6 GB | 262K | 100.26 tok/s | 22/22 with 4K reasoning budget |
| Qwen3.6-27B | Dynamic NVFP4 plus MTP-2 | about 96 GB reserved | 262K | 113.2 tok/s single | 21/22 |
| Qwen3.6-27B | FP8 plus DFlash | about 88 GB total runtime | 262K | about 199 tok/s | 22/22 |
| Gemma 4 31B | Q8_0 | 54.5 GB | 262K | 43.76 tok/s | 22/22 |
| gpt-oss-120b | Q8_0 | 65.8 GB | 131K | 264.38 tok/s | 21/22 |
| Qwen3.6-35B-A3B | Q8_0 | 41.6 GB | 262K | 221.04 tok/s | 15/22 |

MiMo matches the best local coding score and is faster than Gemma 4 Q8.
It is slower and substantially more memory-hungry than the strongest Qwen and gpt-oss deployments.
Its practical advantage is fitting a 310B-total, 15B-active model with real 250K context on one card while retaining correct long-context retrieval and native tool calls.

## SWE-bench Lite canary

The five-case canary resolved 3/5 instances with no empty patches or harness errors.
It used the first five SWE-bench Lite test instances, all from Astropy, with the official local sampling recommendation of temperature 1.0 and top-p 0.95, two 131K slots, and the required 4K reasoning budget.

| Instance | Calls | Exit | Hidden tests |
|---|---:|---|---|
| `astropy__astropy-12907` | 10 | `submitted` | Resolved |
| `astropy__astropy-14182` | 64 | `submitted` | Unresolved |
| `astropy__astropy-14365` | 21 | `submitted` | Unresolved |
| `astropy__astropy-14995` | 12 | `submitted` | Resolved |
| `astropy__astropy-6938` | 76 | `submitted (exit_cost)` | Resolved |

All five instances produced non-empty patches.
Four submitted normally.
The fifth repeatedly rechecked a one-line FITS fix, exceeded the 75-call limit, and was autosubmitted with the correct patch.
Two instances each needed one retry after an 8,192-token response contained no tool call.
There were no malformed tool arguments, terminal format exits, context exits, or harness errors.

The agent and serving phase took 28 minutes 39 seconds from launcher start through the final patch.
The hidden harness added 5 minutes 16 seconds, for 33 minutes 55 seconds end to end.

The exact same three instances are resolved by the full-run Qwen3.6 NVFP4, Qwen3.6 FP8, Qwen3.6 Opus-distilled, and Qwen3.6 stock artifacts.
Gemma 4 Q8 resolves two of the five.
This slice therefore shows parity with the leading Qwen deployments rather than material improvement.
It is also too small and too Astropy-heavy to estimate a 300-case score.
A full 300-instance run was subsequently launched at user request on July 30, 2026.
The retry was stopped after the serving process reproduced the long-context CUDA Xid 8 failure described above.
Seventeen non-empty patches and their logs were preserved in `experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k`.
Future retries should use F16 K/V cache with flash attention and ubatch 128.

Evidence is stored in:

- [`../experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5/preds.json`](../experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5/preds.json)
- [`../experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5/eval/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5.mimo-v2.5-iq2-xxs-r4k-canary5.json`](../experiments/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5/eval/sweagent_lite_mimo_v2_5_iq2_xxs_f7aff786_r4k_canary5.mimo-v2.5-iq2-xxs-r4k-canary5.json)

## Reproduction files

- [`../tools/rtxpro6000_bench.py`](../tools/rtxpro6000_bench.py)
- [`../tools/rtxpro6000_coding_bench.py`](../tools/rtxpro6000_coding_bench.py)
- [`../tools/mimo_v25_api_smoke.py`](../tools/mimo_v25_api_smoke.py)
- [`../tools/mimo_v25_long_context_smoke.py`](../tools/mimo_v25_long_context_smoke.py)
- [`../tools/mimo_v25_cuda_decode_probe.py`](../tools/mimo_v25_cuda_decode_probe.py)
- [`../tools/run_mimo_v25_cuda_isolation.sh`](../tools/run_mimo_v25_cuda_isolation.sh)
- [`../tools/sweagent-rtxpro6000-mimo-v2.5-iq2-xxs.yaml`](../tools/sweagent-rtxpro6000-mimo-v2.5-iq2-xxs.yaml)
- [`../tools/run_swebench_lite_mimo_v2_5_iq2_xxs.sh`](../tools/run_swebench_lite_mimo_v2_5_iq2_xxs.sh)

## Recommendation

Use the 4,096-token reasoning budget for every agentic deployment.
Do not use the unbounded default with finite output limits.
Keep the 4 GiB fit margin.
Use F16 K/V cache, flash attention, and ubatch 128 for the 262K single-slot preset on driver 595.84.
Do not use Q8 KV for long-lived serving until an upstream fix passes this isolation workload.
Use two 131K slots only for parallel benchmark or agent work where a 131K per-request ceiling is acceptable.
Do not promote this quant over the current Qwen3.6 NVFP4 agent preset based on the five-case SWE-bench canary.
Revisit that decision after a stable full 300-instance rerun is available.
