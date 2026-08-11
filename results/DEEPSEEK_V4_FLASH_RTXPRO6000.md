# DeepSeek V4 Flash on RTX Pro 6000

## DeepSeek V4 Flash 0731 EXL3

Tested 2026-08-09 on the RTX PRO 6000 Blackwell workstation.

### Outcome

The `turboderp/DeepSeek-V4-Flash-0731-exl3` 2.04 bpw quant is the recommended DeepSeek V4 Flash configuration for this host.
It fits entirely in 96 GB of VRAM with a preallocated 262,144-token FP16 cache, scores 22/22 on the lightweight coding suite with reasoning disabled, and officially resolves 121/300 cases on SWE-bench Lite.
The full run produced 173 non-empty patches, resolved 121 of them, and completed with zero harness errors for a final score of 40.33%.
It averages 95.33 tok/s on short decode and processes the filled 250K-context probe at 2,265.18 prompt tok/s.
At the observed 250K peak it leaves about 22.1 GiB of raw VRAM free.

The 2.52 bpw quant also fits entirely in VRAM at 256K context.
It scores 21/22 on the lightweight suite and improves filled-context prompt processing to 2,584.89 tok/s, but it averages a slightly lower 94.35 tok/s on short decode and officially resolves only 2/5 matched SWE-bench cases.
Its observed 250K peak leaves only about 6.5 GiB of raw VRAM free.
The extra precision therefore did not improve measured agent quality on this slice and gives up most of the useful operating margin.

MTP speculative decoding raises steady short decode to 180.61 tok/s with 72% to 73% acceptance, but changes deterministic outputs and drops the lightweight score to 15/22.
MTP is rejected for quality-sensitive deployment.

### Pinned inputs

| Component | Pin |
|---|---|
| Model | `turboderp/DeepSeek-V4-Flash-0731-exl3` |
| Recommended branch | `2.04bpw`, revision `b5526babe907141a0750f265a5a2a5cf414f5a64` |
| Higher-precision branch | `2.52bpw`, revision `9e893778f0ea88d4bc85ccef658a156c8e1becca` |
| 2.04 bpw target weights | `74,487,254,345` bytes |
| 2.52 bpw target weights | `91,398,685,177` bytes |
| ExLlamaV3 | `1.4.1` |
| TabbyAPI | `3d2848d03184344664b9a8ed7685033e87744742` |
| PyTorch | `2.11.0+cu130` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB raw VRAM |

The 2.04 bpw download also contains the optional 7,593,735,242-byte MTP draft tensor.
The recommended configuration leaves that draft disabled.

### Serving configuration

Both configurations use one GPU, `max_seq_len: 262144`, `cache_size: 262144`, an FP16 cache, a 2,048-token cache chunk, batch size one, and no CPU MoE offload.
All 48 target modules load on the GPU, so neither configuration relies on host DRAM overflow.
The DeepSeek V4 tool-call parser is enabled through TabbyAPI.

Configuration artifacts:

- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/tabby_config.yml`
- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/tabby_config_mtp.yml`
- `experiments/deepseek_v4_flash_0731_exl3_2.52bpw/tabby_config.yml`

### Performance and quality comparison

| Configuration | Active VRAM | Raw headroom | Decode | 250K prompt | Lightweight | SWE-bench |
|---|---:|---:|---:|---:|---:|---:|
| EXL3 2.04 bpw, no thinking | 74,122 MiB | 23,765 MiB | 95.33 tok/s | 2,265.18 tok/s | 22/22 | **121/300 full** |
| EXL3 2.52 bpw, no thinking | 90,154 MiB | 7,733 MiB | 94.35 tok/s | 2,584.89 tok/s | 21/22 | 2/5 canary |
| EXL3 2.04 bpw with MTP | 82,298 MiB | 15,589 MiB | 180.61 tok/s | Not run | 15/22 | Not run |
| antirez IQ2_XXS GGUF, no thinking | 85,115 MiB | 12,772 MiB | 78.16 tok/s | 395.19 tok/s | 15/22 | 2/5 |

The EXL3 2.04 bpw quant improves short decode by 22.0% over the antirez GGUF and improves the filled-context prompt rate by 5.7 times.
Its 250K retrieval run completed in 110.774 seconds versus 632.934 seconds for the GGUF.
The 2.52 bpw run completed the same probe in 97.018 seconds.
Both returned the exact needle `739184` without truncation.

The 2.52 bpw lightweight miss is an internally inconsistent generated TTL test that inserts a key against real monotonic time and then expects expiry under a mocked clock value of six.
The established scorer still counts it as a miss.
Even setting that anomaly aside, the official SWE-bench result provides no evidence that 2.52 bpw is a better operational choice.

Raw output:

- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/no_think/results.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/native/results.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/mtp_throughput/results.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/mtp_no_think/results.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.52bpw/no_think/results.json`

### Filled-context retrieval

The long-context probe placed a six-digit verification number at 75% depth in a deterministic 249,982-token raw prompt.
TabbyAPI reported 249,985 prompt tokens after chat templating.

| Metric | EXL3 2.04 bpw | EXL3 2.52 bpw |
|---|---:|---:|
| Retrieval | Pass | Pass |
| End-to-end time | 110.774 s | 97.018 s |
| Average prompt evaluation | 2,265.18 tok/s | 2,584.89 tok/s |
| Decode after filled context | 79.25 tok/s | 81.78 tok/s |
| Peak observed VRAM | 74,604 MiB | 90,636 MiB |
| Free VRAM at peak | 22,638 MiB | 6,606 MiB |

Raw output:

- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/long_context_250k.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.52bpw/long_context_250k.json`

### API and tool protocol

The 2.04 bpw native server and the 2.52 bpw no-thinking server each passed all seven API and tool-call protocol checks.
The API smoke harness now sends the same explicit `enable_thinking` chat-template setting used by the benchmark and SWE-agent clients.

Raw output:

- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/api_smoke_native.json`
- `experiments/deepseek_v4_flash_0731_exl3_2.52bpw/api_smoke_no_think.json`

### SWE-agent canary

Both EXL3 quants ran the same five SWE-bench Lite cases with one sequential worker, reasoning disabled, and a configured 20-call ceiling.
SWE-agent checks that ceiling after incrementing the request counter, so all five cases reached 21 API calls and working-tree diffs were captured at exit.

| Instance | EXL3 2.04 bpw | EXL3 2.52 bpw |
|---|---|---|
| `astropy__astropy-12907` | Resolved | Resolved |
| `astropy__astropy-14182` | Resolved | Unresolved |
| `astropy__astropy-14365` | Unresolved | Unresolved |
| `astropy__astropy-14995` | Resolved | Resolved |
| `astropy__astropy-6938` | Empty patch | Empty patch |
| **Official score** | **3/5** | **2/5** |

Each quant produced four non-empty patches and one empty patch with zero harness errors.
The 2.04 bpw quant sent 766,714 prompt tokens, received 14,490 completion tokens, and finished agent generation in 7 minutes 57 seconds.
The 2.52 bpw quant sent 708,053 prompt tokens, received 15,900 completion tokens, and finished in 12 minutes 28 seconds.
The 2.04 bpw result clears the existing 3/5 expansion gate.

Raw output:

- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_canary5/preds.json`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_canary5/*/*.traj`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_canary5/eval/deepseek-v4-flash-0731-exl3-2.04bpw-canary5.json`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.52bpw_canary5/preds.json`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.52bpw_canary5/*/*.traj`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.52bpw_canary5/eval/deepseek-v4-flash-0731-exl3-2.52bpw-canary5.json`

### Full SWE-bench Lite run

The expanded 2.04 bpw run completed all 300 SWE-bench Lite test instances with four concurrent SWE-agent workers and a 75-call ceiling.
TabbyAPI served four 262,144-token slots from a shared 1,048,576-token FP16 cache, kept all target modules on the GPU, disabled MTP, and explicitly disabled thinking in the chat template.

| Metric | Result |
|---|---:|
| Official resolved | **121 / 300 = 40.33%** |
| Non-empty patches evaluated | 173 / 300 |
| Resolved among non-empty patches | 121 / 173 = 69.94% |
| Unresolved non-empty patches | 52 |
| Empty patches | 127 |
| Harness errors | 0 |
| Incomplete evaluations | 0 |

The run was interrupted after 287 completed trajectories when its original wmux workspaces disappeared.
SWE-agent resumed into the same output directory, skipped the 287 completed cases, regenerated the incomplete work, and merged a validated 300-prediction `preds.json`.
The official harness then evaluated all 173 non-empty patches.
The final long-running `psf__requests-2317` case passed, raising the provisional score from 120 to 121 resolved cases.

The lightweight 22/22 result did not translate into top-tier repository-agent performance.
At 40.33%, this quant trails Qwen3.6-27B dynamic NVFP4 by 19 percentage points and trails stock Qwen3.6-35B-A3B Q8 by 8 points on the same 300-instance split.
The main failure mode is patch production rather than hidden-test quality: 42.3% of instances ended with an empty patch, while 69.94% of non-empty patches resolved.

Artifacts:

- `experiments/deepseek_v4_flash_0731_exl3_2.04bpw/tabby_config_batch4.yml`
- `tools/sweagent-rtxpro6000-deepseek-v4-flash-0731-exl3.yaml`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_batch4_full300/preds.json`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_batch4_full300/run_batch.config.yaml`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_batch4_full300/run_batch_exit_statuses.yaml`
- `experiments/sweagent_lite_deepseek_v4_flash_0731_exl3_2.04bpw_batch4_full300/eval/deepseek-v4-flash-0731-exl3-2.04bpw-batch4-full300.json`

## DeepSeek V4 Flash 0731 antirez IQ2_XXS imatrix

Tested 2026-07-31 on the RTX PRO 6000 Blackwell workstation.

### Outcome

The single-file antirez importance-matrix quant fits completely on the GPU at a 262,144-token allocation.
llama.cpp offloaded all 44 model layers to CUDA, used 85,115 MiB of VRAM after load, and left 12,772 MiB of raw VRAM headroom.
No model layers spilled into host DRAM.

The standard short-prompt benchmark averaged 78.16 tok/s.
A real 249,985-token API prompt completed without truncation and retrieved the correct needle at 75% depth.
The quant passed all seven API and tool-call protocol checks.

Quality is the weakness.
The lightweight coding suite scored 15/22 with reasoning disabled and 16/22 in native mode.
Native mode then consumed the entire 16,384-token allowance without emitting code on the TTL/LRU task, and the 4,096-token reasoning budget was ignored by this GGUF chat template.
A SWE-agent canary found the correct patch and passed 11/11 targeted tests, but still had not submitted after 20 model calls and entered a repeated empty-command loop.
A fresh five-case canary with a configured 20-call ceiling captured three patches and officially resolved 2/5 cases.
This missed the 3/5 expansion gate, so a full 300-case run is not justified.

### Pinned inputs

| Component | Pin |
|---|---|
| Model | `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| Hugging Face revision | `1cd7b564460821938add0475a60b942c409295e0` |
| Quant size | `86,720,111,488` bytes, or 80.76 GiB |
| SHA-256 | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| llama.cpp | `876a4321163249c43ca4e986818fab5ab081f282`, build 10216 |
| CUDA build | CUDA 13.0, `sm_120`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB raw VRAM |
| CPU and RAM | AMD EPYC 4585PX, 16 cores and 32 threads, 125 GiB RAM |

The GGUF metadata declares DeepSeek V4, 43 transformer blocks plus the output layer, a 1,048,576-token context, YaRN factor 16 over a 65,536-token original context, 256 routed experts, and six active experts per token.

### Serving configuration

```bash
llama-server \
  -m /mnt/extended/gisenberg/models/deepseek-v4-flash-antirez-imatrix-0731-1cd7b564/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --host 127.0.0.1 --port 8091 \
  -c 262144 -ngl auto -fa on -np 1 \
  --jinja --fit on --fit-target 4096 --fit-ctx 262144 \
  -b 1024 -ub 128 -ctk f16 -ctv f16 \
  --threads 16 --threads-batch 32 \
  --ctx-checkpoints 0 --metrics \
  --reasoning-format deepseek --reasoning off --reasoning-budget 0
```

The no-thinking configuration is the operational default because native mode can spend the entire output allowance in reasoning without reaching code.
The embedded template does not support `--reasoning-preserve`, and llama.cpp warned that the option had no effect.

### Exact 256K memory placement

The verbose fit audit projected 83,617 MiB of device allocations against 95,920 MiB free before loading.
The process reached 85,115 MiB reported GPU usage after load and peaked at 85,299 MiB during the 250K request.

| Allocation | Size |
|---|---:|
| CUDA model buffer | 81,687.67 MiB |
| CPU-mapped auxiliary model buffer | 1,010.00 MiB |
| Raw sliding-window KV | 10.75 MiB |
| CSA compressed KV | 1,344.00 MiB |
| HCA compressed KV | 40.00 MiB |
| Lightning-indexer KV | 336.00 MiB |
| Compressor state | 11.64 MiB |
| CUDA compute buffer | 187.57 MiB |
| Host compute buffer | 36.63 MiB |
| Host output buffer | 0.49 MiB |

llama.cpp reported `offloaded 44/44 layers to GPU`.
The 1,010 MiB CPU-mapped auxiliary buffer is not layer-weight overflow.
The live server used about 2.8 GiB resident host memory during the 250K request and incurred zero major page faults.

### Standard throughput benchmark

The standard benchmark used one warmup followed by five timed 256-token generations at the full 262,144-token allocation.

| Metric | Result |
|---|---:|
| Warm load time | 9.0 s |
| VRAM after load | 85,115 MiB |
| Raw VRAM headroom | 12,772 MiB |
| Mean TTFT | 73.5 ms |
| Median TTFT | 72.7 ms |
| Mean decode | 78.16 tok/s |
| Median decode | 78.17 tok/s |
| Timed range | 78.05 to 78.24 tok/s |

Raw output:

- `experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731.json`

### Lightweight coding quality

| Mode | String | Expression | A-star | TTL/LRU | Total | Read |
|---|---:|---:|---:|---:|---:|---|
| Reasoning off | 5/5 | 5/5 | 5/6 | 0/6 | 15/22 | TTL/LRU emitted implementation without runnable tests. |
| Native | 5/5 | 5/5 | 6/6 | 0/6 | 16/22 | TTL/LRU hit 16,384 completion tokens and returned no extractable code. |
| Reasoning budget 4,096 | Identical prefix | Identical prefix | Identical prefix | Aborted after 5,332 tokens | Not scored | The template ignored the budget and reproduced native output token-for-token. |

Raw output:

- `experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731-no-think.json`
- `experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731-no-think/*.md`
- `experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731.json`
- `experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731/*.md`

### API and tool protocol

The reasoning-off server passed 7/7 checks.
It produced the exact basic response, returned the correct arithmetic answer, suppressed reasoning content, emitted one correctly named tool call with exact arguments, and consumed the tool result without calling the tool again.

Raw output:

- `experiments/deepseek_v4_flash_antirez_imatrix_0731_rtxpro6000/api_smoke_no_think.json`

### Filled-context retrieval

The long-context probe placed a six-digit verification number at 75% depth in a deterministic filler prompt.
The response returned exactly `739184`.

| Metric | Result |
|---|---:|
| Raw prompt tokens | 249,981 |
| API prompt tokens | 249,985 |
| Completion tokens | 3 |
| Truncated | No |
| Retrieval | Pass |
| End-to-end time | 632.934 s |
| Average prompt evaluation | 395.19 tok/s |
| Decode after filled context | 51.93 tok/s |
| Peak observed VRAM | 85,299 MiB |
| Free VRAM at peak | 12,588 MiB |

Prompt processing began near 642 tok/s and declined smoothly to about 399 tok/s at 97% fill.
VRAM rose by only 184 MiB from the post-load reading, with no PCIe spill cliff.

Raw output:

- `experiments/deepseek_v4_flash_antirez_imatrix_0731_rtxpro6000/long_context_250k.json`

### SWE-agent canary

The first SWE-bench Lite case was `astropy__astropy-12907`.
The model reproduced the bug, made the correct one-line `_cstack` fix, and passed all 11 targeted tests by its sixth model call.
It then ran a broader suite, pursued unrelated baseline failures, repeated already-passing tests, searched for test changes that the task explicitly said not to make, and finally repeated empty `sed` ranges.
The run was stopped after 20 model calls with no submission.

A fresh five-case run configured a 20-call per-instance ceiling.
SWE-agent checks the ceiling after incrementing the request counter, so ceiling exits are recorded at 21 API calls.
The framework captured an existing working-tree diff at that exit, but produced an empty prediction when no diff existed.

| Instance | API calls | Exit | Patch | Official result |
|---|---:|---|---|---|
| `astropy__astropy-12907` | 21 | `submitted (exit_cost)` | 504 bytes | Resolved |
| `astropy__astropy-14182` | 21 | `exit_cost` | Empty | Unresolved |
| `astropy__astropy-14365` | 21 | `submitted (exit_cost)` | 475 bytes | Unresolved |
| `astropy__astropy-14995` | 16 | `submitted` | 671 bytes | Resolved |
| `astropy__astropy-6938` | 21 | `exit_cost` | Empty | Unresolved |

The official SWE-bench harness evaluated all three non-empty patches with zero harness errors.
Two of those three patches resolved their cases, for 2/5 across the full bounded slice.
`astropy__astropy-14365` applied cleanly and preserved all pass-to-pass tests, but failed its target `test_roundtrip[True]` regression.
Only one case submitted voluntarily; two useful diffs were recovered by the ceiling, and two cases reached the ceiling without editing.
The five cases used 100 model calls, sent 779,936 prompt tokens, received 13,432 completion tokens, and took 44 minutes 36 seconds end to end with one sequential agent worker.
The 2/5 score missed the 3/5 expansion gate.

Raw output:

- `experiments/sweagent_lite_deepseek_v4_flash_antirez_imatrix_0731_canary5/astropy__astropy-12907/astropy__astropy-12907.traj`
- `experiments/sweagent_lite_deepseek_v4_flash_antirez_imatrix_0731_call20_canary5/preds.json`
- `experiments/sweagent_lite_deepseek_v4_flash_antirez_imatrix_0731_call20_canary5/*/*.traj`
- `logs/run_evaluation/deepseek-v4-flash-antirez-imatrix-0731-call20-canary5/`

### Comparison with the Unsloth 0731 quant

The antirez file is 3.86 GiB smaller and reduces post-load VRAM by 4,580 MiB.
That extra memory is not needed to fit 256K on this 96 GB card because the Unsloth quant already left more than 7 GiB free.
The antirez quant is 2.8% slower on short decode and 7.4% slower on the 250K average prompt evaluation.
Its best lightweight score is 16/22 versus 19/22 for the deleted Unsloth checkpoint, and its practical reasoning-off score is 15/22.

The antirez importance-matrix build is therefore a clean memory and protocol fit, but not the better model for coding or agent work on this host.

## Previous DeepSeek V4 Flash 0731 UD-IQ2_XXS baseline

Tested 2026-07-31 on the RTX PRO 6000 Blackwell workstation.

### Outcome

The Unsloth `UD-IQ2_XXS` quant fits completely in VRAM at a 262,144-token allocation.
llama.cpp offloaded all 44 model layers to CUDA and did not move model weights into host DRAM.
The standard short-prompt benchmark averaged 80.40 tok/s, and a real 249,985-token API prompt completed without truncation and retrieved a needle at 75% depth.

### Pinned inputs

| Component | Pin |
|---|---|
| Model | `unsloth/DeepSeek-V4-Flash-0731-GGUF`, `UD-IQ2_XXS` |
| Hugging Face revision | `cf5e97a3b3c1192e7628ffb5137dd2a793404e25` |
| Quant size | `90,860,736,928` bytes, or 84.62 GiB |
| llama.cpp | `876a4321163249c43ca4e986818fab5ab081f282`, build 10216 |
| CUDA build | CUDA 13.0, `sm_120`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB raw VRAM |
| CPU and RAM | AMD EPYC 4585PX, 16 cores and 32 threads, 125 GiB RAM |

The downloaded shards are:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `00001-of-00003` | 5,257,664 | `c58c9d62eac7b62e9578b52613f425e48313d7212ab8d1d76caed8ea8de26595` |
| `00002-of-00003` | 49,890,588,800 | `65a113df6d4469f16db6882b6919e153c464c3c78c833f5e1b41a33803cdbd52` |
| `00003-of-00003` | 40,964,890,464 | `a69102ddfaf4a84426e11fdb66716654f4260dc3a1de3ade9fd50e006b8691d3` |

### Serving configuration

```bash
llama-server \
  -m /mnt/extended/gisenberg/models/deepseek-v4-flash-0731-iq2-xxs-cf5e97a3/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf \
  --host 127.0.0.1 --port 8091 \
  -c 262144 -ngl auto -fa on -np 1 \
  --jinja --fit on --fit-target 4096 --fit-ctx 262144 \
  -b 1024 -ub 128 -ctk f16 -ctv f16 \
  --threads 16 --threads-batch 32 \
  --ctx-checkpoints 0 --metrics
```

MTP speculative decoding was deliberately disabled for the baseline.
The server used the GGUF chat template and correctly returned separate `reasoning_content` in the protocol smoke test.

### Exact 256K memory placement

The verbose fit audit projected 88,292 MiB of CUDA allocations and left 7,628 MiB free before loading.
The process reached 89,695 MiB reported GPU usage after load and peaked at 89,943 MiB while processing the 250K request.

| Allocation | Size |
|---|---:|
| CUDA model buffer | 86,362.40 MiB |
| Raw sliding-window KV | 10.75 MiB |
| CSA compressed KV | 1,344.00 MiB |
| HCA compressed KV | 40.00 MiB |
| Lightning-indexer KV | 336.00 MiB |
| Compressor state | 11.64 MiB |
| CUDA compute buffer | 187.57 MiB |
| Host-mapped model buffer | 284.06 MiB |
| Host compute buffer | 36.63 MiB |

llama.cpp reported `offloaded 44/44 layers to GPU`.
This configuration therefore has no model-weight DRAM overflow.

### Standard throughput benchmark

The repository's standard benchmark used one warmup followed by five timed 256-token generations at the full 262,144-token allocation.

| Metric | Result |
|---|---:|
| Warm restart load time | 9.0 s |
| VRAM after load | 89,695 MiB |
| Mean TTFT | 74.1 ms |
| Median TTFT | 73.6 ms |
| Mean decode | 80.40 tok/s |
| Median decode | 80.47 tok/s |
| Timed range | 80.17 to 80.49 tok/s |

The first cold load immediately after downloading the shards took 53.8 seconds.
The warm restart benefited from the host page cache.

Raw output:

- `experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-0731-ud-iq2-xxs.json`

### Filled-context retrieval

The long-context probe placed a six-digit verification number at 75% depth in a deterministic filler prompt.
The response returned exactly `739184`.

| Metric | Result |
|---|---:|
| Raw prompt tokens | 249,981 |
| API prompt tokens | 249,985 |
| Completion tokens | 3 |
| Truncated | No |
| Retrieval | Pass |
| End-to-end time | 586.062 s |
| Average prompt evaluation | 426.83 tok/s |
| Decode after filled context | 51.40 tok/s |
| Peak observed VRAM | 89,943 MiB |
| Free VRAM at peak | 7,299 MiB |

Prompt processing began near 745 tok/s and declined gradually as the compressed attention state grew.
The decline was smooth, and VRAM remained essentially flat, with no PCIe spill cliff.

Raw output:

- `experiments/deepseek_v4_flash_0731_iq2_xxs_rtxpro6000/long_context_250k.json`

### Read

The 0731 Unsloth IQ2 quant is a substantially better operational fit than the preview-era antirez quant tested below.
It provides a four-times-larger tested context allocation, more than doubles short-context decode throughput on the newer llama.cpp stack, and still leaves over 7 GiB of VRAM free.
The next useful experiment is MTP with a small draft ceiling, but the non-MTP configuration is already fast enough for interactive agent work and is the safer production baseline.

The remainder of this document records the earlier preview-model investigation and is retained as historical comparison.

Tested 2026-06-30 on the RTX PRO 6000 Blackwell workstation.

## Candidate

The useful target for this host is the antirez q2 GGUF:

- Repo: <https://huggingface.co/antirez/deepseek-v4-gguf>
- File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
- Size: `86,720,111,200` bytes = 80.8 GiB
- Quant recipe: routed expert gate/up `IQ2_XXS`, routed expert down `Q2_K`, attention/shared/output mostly `Q8_0`, routers/indexer/compressor/HC kept higher precision.

The sokann near-lossless GGUF is useful as a runbook, but not as the target on this machine:

- Repo: <https://huggingface.co/sokann/DeepSeek-V4-Flash-GGUF>
- File: `DeepSeek-V4-Flash.gguf`
- Size: `156,378,344,544` bytes = about 146 GiB
- Card guidance: fits on 160 GiB system RAM plus 48 GiB VRAM. This host has 62 GiB RAM plus 63 GiB swap, so the 146 GiB file is the wrong fit.

## llama.cpp Build

DeepSeek V4 support landed in llama.cpp PR #24162:
<https://github.com/ggml-org/llama.cpp/pull/24162>

The existing CUDA build on this host was too old:

- Existing: `/home/gisenberg/llama-build/src/build/bin/llama-server`
- Version: `207 (bbeb89d)`, before DeepSeek V4 support

I built a separate worktree so existing benchmarks keep their old binary:

- Source: `/home/gisenberg/llama-build/src-deepseek-v4`
- Binary: `/home/gisenberg/llama-build/src-deepseek-v4/build/bin/llama-server`
- Version: `1021 (0eca4d490)`, tag `b9851`
- CUDA: `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`

## Run Config

The benchmark entry is `deepseek-v4-flash-iq2xxs` in `tools/rtxpro6000_bench.py`.

Download:

```bash
hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir /home/gisenberg/models/deepseek-v4-flash-iq2xxs \
  --max-workers 1
```

Throughput run:

```bash
LLAMA_BACKEND=cuda \
LLAMA_DIR=/home/gisenberg/llama-build/src-deepseek-v4/build/bin \
LLAMA_PORT=8091 \
python3 tools/rtxpro6000_bench.py deepseek-v4-flash-iq2xxs 32768
```

Effective server flags:

```bash
llama-server \
  -m /home/gisenberg/models/deepseek-v4-flash-iq2xxs/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --port 8091 --host 127.0.0.1 \
  -c 32768 -ngl auto -fa on -np 1 --no-mmap \
  --jinja -cram 0 --fit on -b 2048 -ub 2048
```

## Results

Throughput at 32K context:

| Metric | Value |
|---|---:|
| Load time | 12.0 s |
| VRAM after load | 92,168 MiB |
| VRAM headroom | ~5,719 MiB |
| Mean TTFT | 184 ms |
| Mean decode | 38.73 tok/s |
| Timed runs | 5 x 255 completion tokens |

Raw output:

- `experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-iq2xxs.json`

Coding suite:

| Benchmark | Raw | Capped | Notes |
|---|---:|---:|---|
| String Processor | 4/5 | 4/5 | One generated test has a wrong expected value: `"Python is fun"` has 3 vowels, not 4. |
| Expression Evaluator | 5/5 | 5/5 | Clean. |
| A* Pathfinding | 8/6 | 6/6 | Model generated two extra passing tests; cap to expected score for rankings. |
| LRU Cache with TTL | 0/6 | 0/6 | Implementation-only response; pytest collected zero tests. |
| Total | 17/22 raw | 15/22 capped | Use capped score for comparison. |

Raw output:

- `experiments/rtxpro6000_coding/deepseek-v4-flash-iq2xxs.json`
- `experiments/rtxpro6000_coding/deepseek-v4-flash-iq2xxs/*.md`

## SWE-bench Lite Partial Run

I aborted the full SWE-bench Lite run on 2026-07-01 after collecting enough signal. This is not a full 300-instance score.

Harness:

- SWE-agent `v1.1.0`, function-calling tools, 75 calls per instance.
- `num_workers=1`; the 80 GiB GGUF is too VRAM-tight for parallel slots on this card.
- Server started at 32K context first, then retried the 32K context failures at 64K.

Context findings:

| Config | Result |
|---|---|
| `-c 32768 -b 2048 -ub 2048` | Stable, ~92.2 GiB VRAM used, ~5.7 GiB headroom. Two SWE-agent instances hit client-side context limits. |
| `-c 65536 -b 2048 -ub 2048` | Loaded, but failed during the Django retry with CUDA OOM. |
| `-c 65536 -b 1024 -ub 512` | Stable enough for the context-failure retry and the resumed run, ~87.9 GiB VRAM used, ~9.3 GiB headroom. |

The 64K retry helped the low-context failures:

- `astropy__astropy-14365`: changed from context failure to submitted.
- `django__django-11283`: changed from context failure to submitted, then hit the per-instance call limit.

Official SWE-bench harness result for the merged partial snapshot:

| Metric | Count |
|---|---:|
| Total Lite instances | 300 |
| Predictions submitted before abort | 72 |
| Non-empty patches evaluated | 58 |
| Resolved | 42 |
| Unresolved | 16 |
| Empty patches | 14 |
| Harness errors | 0 |
| Resolved / evaluated non-empty patches | 72.4% |
| Resolved / submitted predictions | 58.3% |

Important caveat: the partial run is early-slice biased, mostly Astropy and Django. The submitted-patch quality is real enough to notice, but it should not be linearly compared to the full 300-instance Qwen/Gemma runs.

Raw output:

- `experiments/sweagent_lite_deepseek_full/preds.snapshot_with_64k_retries.json`
- `experiments/sweagent_lite_deepseek_context_retries_64k/preds.json`
- `experiments/sweagent_lite_deepseek_full/eval/sweagent_lite_deepseek_full.deepseek-v4-flash-iq2xxs-partial-72.json`

## vLLM-Moet Smoke Test

I also tried `kacper-daftcode/vLLM-Moet` at commit `591250b` on 2026-07-09 using the official
`deepseek-ai/DeepSeek-V4-Flash` FP8/NVFP4 checkpoint on the root NVMe volume.

Setup:

- Docker image: `vllm-moet-sm120:v024`, 33.9 GB, built locally from `Dockerfile.sm120-v024`.
- Model footprint: 149 GB, 46 safetensors shards.
- Host: RTX PRO 6000 Blackwell, 125 GiB RAM, 64 GiB swap.
- Runtime config: FP8 KV, `deepseek_v4` tokenizer mode, 24K context, MTP with two draft tokens.

Results:

| Attempt | MoE knobs | Result |
|---|---|---|
| Delta pool | `VLLM_MOE_W2=1`, `VLLM_MOE_W2_DELTA_GB=1` | Engine reached DeepSeek V4/MTP setup and the FP4/MXFP4 MoE path, then was OOM-killed during load. GPU use was only about 10-13 GB; host OOM log showed about 101 GiB anonymous RSS. |
| No delta pool | `VLLM_MOE_W2=1`, `VLLM_MOE_W2_DELTA=0`, `VLLM_MOE_W2_DELTA_GB=0` | Same pre-readiness failure. Docker recorded `OOMKilled=true`; GPU use was about 13 GB; host OOM log showed about 117 GiB anonymous RSS. |

The surviving container log also warned that the `VLLM_MOE_W2*` variables were unknown vLLM
environment variables. Those variables do exist in the vLLM-Moet patch/docs, so this warning is
not by itself proof that the knobs were ignored, but it is an operational risk to keep in mind.

Verdict: on this host, the vLLM-Moet path failed on host RAM/load staging before the server became
ready. This was not a VRAM-fit failure and produced no benchmarkable throughput or SWE-bench result.
The official checkpoint snapshot was removed after the failed smoke test.

Raw notes:

- `experiments/vllm_moet_deepseek_v4_flash/README.md`

## Read

This is runnable on the 96 GB card, and the partial SWE-bench submitted-patch quality was better than the small coding suite suggested. It is still not an obvious replacement for the current Pro 6000 winners.

- The fit is real at 32K, but tight: 92.2 GiB used leaves only ~5.6 GiB on a 97.9 GiB card.
- 64K can run if batch sizing is reduced, but the stable configuration trades away more throughput and still leaves no room for parallel SWE-agent workers.
- Throughput is similar to dense Gemma 4 31B Q8 territory, but with much higher VRAM use.
- Coding quality is not S-tier on this harness. Capped score is 15/22, with the same broad LRU weakness that shows up in several Qwen-family runs.
- SWE-bench Lite looked much better on the partial slice: 42/58 resolved among non-empty evaluated patches, 42/72 among all submitted predictions. The run was aborted because of wall-clock cost and operational fit, not because the model was failing the harness.
- The antirez imatrix quant's fresh bounded canary resolved 2/5 cases, with only one voluntary submission. It missed the 3/5 expansion gate and showed weaker agent efficiency than the earlier partial run.
- The antirez card explicitly says these quants are specific for the DS4 inference engine and "may work with other inference engines or not." llama.cpp loads and runs this file, but quality should be treated as a real empirical question, not assumed from the base model.
- The sokann card's KLD comparison also flags the q2 quant as lossy versus the near-lossless 146 GiB baseline, despite "works amazingly well" qualitative behavior.
- The vLLM-Moet official-checkpoint route did not get past model load on this 128 GB RAM host; it appears to need a much lower host-staging footprint or more RAM before it can be evaluated here.

Practical verdict: keep it as a runnable curiosity / DeepSeek-specific behavior test. For daily Pro 6000 coding, Qwen3.6-27B FP8 + DFlash, Opus-distilled Qwen3.6-35B-A3B, stock Qwen3.6-35B-A3B, Gemma 4 31B Q8, and gpt-oss-120b Q8 remain better-supported choices in this repo.
