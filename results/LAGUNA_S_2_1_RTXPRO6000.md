# Laguna S 2.1 NVFP4 on RTX PRO 6000

## August 1 RC2/main retest

### Decision

The August RC2/main checkpoint improves native tool-call formatting with Poolside's corrected sampling defaults, but it is not practical on a single 96 GB RTX PRO 6000 at 256K context.
The target grew from 66.961 GiB to 92.850 GiB because layers 40 through 47 exclude their MoE expert weights from NVFP4 quantization and store 36.000 GiB of those tensors in BF16.
Those same eight layers occupied 10.125 GiB in the July NVFP4 checkpoint, accounting for 25.875 GiB of the 25.889 GiB total increase.
The native 1M RoPE configuration does not materially increase checkpoint weight storage.

The target cannot load entirely on the GPU.
At 256K, the viable non-speculative configuration offloads 24 GiB of experts and runs at 16.042 serial output tok/s.
The viable DFlash configuration offloads 30.23 GiB and runs at 11.685 serial output tok/s, so DFlash is slower for the interactive single-request case.
The bounded SWE-agent retest was stopped after it reproduced a 4,096-token runaway reasoning turn and began another long turn before making any source edit.
A full SWE-bench Lite run is not warranted.

### Pinned RC2 artifacts

| Component | Revision |
|---|---|
| Target | `poolside/Laguna-S-2.1-NVFP4` at `f8fdfcdc4e7b0c474a0102430a8cae0a3a358669` |
| DFlash draft | `poolside/Laguna-S-2.1-DFlash-NVFP4` at `b3b5921a900b9e0a1e27e50bdaeb480692a6d19b` |
| Target storage | 99,697,287,856 bytes, or 92.850 GiB |
| DFlash storage | Approximately 2.08 GiB |
| Native maximum context | 1,048,576 tokens |
| Retest context | 262,144 tokens |
| Sampling | Temperature `1.0`, top-p `1.0`, top-k `20`, min-p `0.0` |

Poolside's fix announcement identifies a thinking loop caused by the combination of DFlash and TensorRT-LLM plus an incorrect default serving temperature.
This retest used vLLM 0.26, so it tests the new weights and corrected temperature but not the exact TensorRT-LLM integration path.

### Why RC2 is larger

| Checkpoint | Total | All BF16 tensors | Layers 40-47 expert tensors |
|---|---:|---:|---:|
| July `07614121` | 66.961 GiB | 7.476 GiB | 10.125 GiB NVFP4 data and scales |
| RC2 `f8fdfcdc` | 92.850 GiB | 43.476 GiB | 36.000 GiB BF16 |
| Change | +25.889 GiB | +36.000 GiB | +25.875 GiB for the same expert block |

The RC2 compression config adds `re:^model\.layers\.4[0-7]\.mlp\.experts(\..*)?$` to the quantization ignore list.
This is a selective mixed-precision checkpoint, not an all-expert NVFP4 checkpoint.

### 256K fit and performance

| Mode | Host expert offload | Loaded model VRAM | KV capacity | Serial output tok/s | Four-way output tok/s | Mean serial TTFT | Measured VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-speculative | 24.00 GiB | 69.25 GiB | 305,269 tokens, 1.16x | 16.042 | 22.874 | 0.4347 s | 92,352 MiB |
| DFlash, 15 tokens | 30.23 GiB | 65.42 GiB including draft | 278,455 tokens, 1.06x | 11.685 | 26.019 | 1.1840 s | 94,108 MiB |

DFlash is 27.2% slower serially and 13.7% faster at four-way aggregate throughput.
Its draft-token acceptance rate is 22.264%, with a 4.34-token mean acceptance length.
The full-context DFlash server also requires much more offload and operates with less transient GPU headroom.

Both modes passed the direct reasoning and native tool-call protocol checks.
The simple reasoning probe consumed its full 1,024-token cap without producing final content in both modes.

### SWE-agent retest

The intended regression case was `astropy__astropy-14182` with DFlash disabled, 24 GiB expert offload, and the corrected sampling defaults.
The model completed 17 valid native tool calls without a `FunctionCallingFormatError` before the run was stopped during call 18.
It inspected relevant code, created and ran a reproducer, and diagnosed the failing behavior.
It then emitted an 11,230-character reasoning trace that consumed the full 4,096-token per-call cap before issuing another inspection command.
It had not edited source by call 18 and had entered another long generation when the run was aborted.

An accidental control run on `astropy__astropy-6938` also produced 21 valid native calls with no format error.
It reached the 20-call cap without editing source and submitted an empty patch.
The control therefore supports improved syntax but not improved agent efficiency or task completion.

### RC2 evidence

- Non-speculative fit and performance: `experiments/laguna_s_2_1_nvfp4_f8fdfcdc_dflash_b3b5921a_vllm026_uva24_gpu0925`
- DFlash fit and performance: `experiments/laguna_s_2_1_nvfp4_f8fdfcdc_dflash_b3b5921a_vllm026_uva30_gpu094`
- All-GPU load failure: `experiments/laguna_s_2_1_nvfp4_f8fdfcdc_dflash_b3b5921a_vllm026_gpu090`
- Correct regression case, stopped at call 18: `experiments/sweagent_lite_laguna_s_2_1_nvfp4_f8fdfcdc_baseline_uva24_gpu0925_astropy14182_exact`
- Accidental control case: `experiments/sweagent_lite_laguna_s_2_1_nvfp4_f8fdfcdc_baseline_uva24_gpu0925_astropy14182`

## July 27 decision

The July 27 refresh fixes native NVFP4 serving on the RTX PRO 6000 and materially improves DFlash draft acceptance.
It does not improve the lightweight coding score or the third-party agent tool-schema behavior enough to justify a full 300-case SWE-bench Lite run.
The official Q4_K_M fallback was not used because there is no remaining NVFP4 compatibility blocker.

## Pinned artifacts

| Component | Revision |
|---|---|
| Target | `poolside/Laguna-S-2.1-NVFP4` at `07614121b31898586430f189d27a25a0be310843` |
| DFlash draft | `poolside/Laguna-S-2.1-DFlash-NVFP4` at `4cdcc6e9b29105e8ff5790885cadccbeb4f33f54` |
| vLLM | `0.26.0` |
| PyTorch | `2.11.0` with CUDA 13.0 |
| FlashInfer | `0.6.15.dev20260712` Python, cubin, and JIT-cache packages |
| NVIDIA driver | `595.84` |
| Hardware | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, `sm_120` |

The isolated environment is `/home/gisenberg/venvs/laguna-s-2.1-07614121-vllm026`.
The target, draft, and tokenizer overlay are stored under `/mnt/extended/gisenberg/models`.
The benchmark evidence includes package versions, artifact file lists, SHA-256 hashes, launch commands, API responses, performance JSON, and server logs.

## Serving configuration

The matched comparison used the following common settings:

- 262,144-token maximum model length.
- FP8 KV cache from the checkpoint configuration.
- FlashInfer attention and `FLASHINFER_CUTLASS` NVFP4 MoE kernels.
- `max_num_batched_tokens=8192`.
- `max_num_seqs=32` for the local performance suite.
- Prefix caching enabled.
- Poolside v1 reasoning and tool parsers.
- Thinking and preserved reasoning enabled.
- Temperature `0.7`, top-p `0.95`, and top-k `20`.
- GPU memory utilization `0.90`.

The DFlash run used the matched July 27 draft with `num_speculative_tokens=7`, which is the draft model card's recommended setting.

vLLM 0.26 introduced CUDA-graph memory estimation during KV-cache sizing.
At the prior `gpu_memory_utilization=0.87`, DFlash had 9.13 GiB available for KV cache while the full 262K context required 10.16 GiB, so startup correctly failed.
At `0.90`, DFlash had 11.98 GiB of KV cache, enough for 309,122 tokens and 1.18 full-length concurrent requests.
Both baseline and DFlash were rerun at `0.90` to keep memory and performance comparisons matched.

## OpenAI-compatible API checks

Both final matched servers passed:

- Normal generation through `/v1/chat/completions`.
- Parsed reasoning in `reasoning_content`.
- Preserved reasoning supplied in conversation history.
- Exactly one native tool call with OpenAI-compatible arguments.
- A tool-result follow-up without an additional tool call.

One earlier baseline probe returned an empty preserved-reasoning follow-up with `finish_reason=stop`.
Three subsequent protocol invocations passed, including both final matched baseline and DFlash runs.
The harness now preserves raw API envelopes and independent protocol flags so a stochastic empty response does not discard the remaining evidence.

## Performance

| Mode | Serial output tok/s | Four-way output tok/s | Mean serial TTFT | Mean four-way TTFT | Idle VRAM | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Non-speculative | 106.764 | 277.956 | 0.0388 s | 0.0495 s | 88,770 MiB | 88,772 MiB |
| DFlash, 7 tokens | 229.794 | 374.723 | 0.0747 s | 3.3689 s | 93,234 MiB | 93,234 MiB |
| DFlash improvement | 2.15x | 1.35x | 1.93x slower | 68.1x slower | +4,464 MiB | +4,462 MiB |

DFlash drafted 12,019 tokens and accepted 6,484.
The draft-token acceptance rate was 53.948%, and mean acceptance length was 4.776 tokens.

The prior 15-token draft run reached 244.973 serial tok/s and 445.445 four-way tok/s, with 30.33% draft-token acceptance and a 5.55-token mean acceptance length.
The refreshed draft is much better matched, but the official 7-token setting leaves less speculative depth and produces lower aggregate throughput than the prior 15-token configuration.

## Lightweight coding quality

| Task | Best of three | Mean |
|---|---:|---:|
| Expression evaluator | 5/5 | 3.3/5 |
| A* pathfinding | 5/6 | 4.0/6 |
| LRU cache with TTL | 6/6 | 5.7/6 |
| String processor | 5/5 | 5.0/5 |
| Total | 21/22 | 18.0/22 |

The prior target run scored 22/22 best of three and averaged 19.0/22.
The refreshed serving stack therefore shows no material quality improvement on this benchmark.

## Tool-schema findings

The direct API tool smoke passes because it uses a simple schema and requires a call.
The SWE-agent gate still emits undeclared arguments such as `description` and returns long responses with no native tool call.
This matches Poolside's documented harness-overfitting limitation, where the model may use a memorized native-harness interface instead of a similar third-party schema.

The open vLLM declarative parser rewrite may improve incremental parsing and argument coercion.
It does not guarantee that invented but syntactically valid arguments will be removed.
The separate open reasoning-token and bare-`</think>` fixes address reasoning-channel correctness rather than schema adherence.

## SWE-bench Lite gate

The refreshed five-case gate is recorded under `experiments/sweagent_lite_laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090_sanity`.
Four agents produced nonempty patches.
The fifth case, `astropy__astropy-14182`, remained stuck in repeated 16K no-tool generations and was stopped.

The four completed patches scored 3/4 in the SWE-bench harness:

- Resolved: `astropy__astropy-12907`, `astropy__astropy-14995`, and `astropy__astropy-6938`.
- Unresolved: `astropy__astropy-14365`.
- Empty patches: zero.
- Harness errors: zero.

The agent generated 28 `FunctionCallingFormatError` retries before the stop:

- 24 responses contained no native tool call.
- Two calls invented an undeclared `description` argument.
- One call invented an undeclared `command` argument.
- One call invented an undeclared `path` argument.

The prior stricter-schema five-case gate completed 4/5 with 19 retries, including 17 no-tool responses and two undeclared-argument failures.
The refreshed stack is therefore worse on the behavior that blocked the earlier full run.
No full 300-case SWE-bench Lite run is warranted.

## Evidence

- Final matched benchmark: `experiments/laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090`
- Initial vLLM 0.26 baseline and protocol failure: `experiments/laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026`
- Successful baseline at 0.87 plus DFlash KV-cache sizing failure: `experiments/laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_rerun1`
- Five-case agent gate: `experiments/sweagent_lite_laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090_sanity`

## Reproduction

```bash
rtk proxy bash tools/setup_laguna_s_2_1_nvfp4.sh

rtk proxy env \
  OUT_DIR=experiments/laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090 \
  SPECULATIVE_TOKENS=7 \
  GPU_MEMORY_UTILIZATION=0.90 \
  rtk proxy bash tools/run_laguna_s_2_1_nvfp4_eval.sh

rtk proxy env \
  SWE_SLICE=0:5 \
  OUT_DIR=experiments/sweagent_lite_laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090_sanity \
  RUN_ID=laguna_s_2_1_nvfp4_07614121_dflash_4cdcc6e9_vllm026_gpu090_sanity \
  SPECULATIVE_TOKENS=7 \
  GPU_MEMORY_UTILIZATION=0.90 \
  rtk proxy bash tools/run_swebench_lite_laguna_s_2_1_nvfp4.sh
```
