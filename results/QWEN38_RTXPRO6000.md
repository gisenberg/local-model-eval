# Qwen3.8-27B FP8 + MTP-3 on RTX Pro 6000

Qwen3.8-27B FP8 fits comfortably on one RTX Pro 6000 Blackwell at the full native 262,144-token allocation.
The retained configuration uses the official FP8 checkpoint, the dedicated Qwen3.8 vLLM build, the checkpoint's native MTP head with three speculative tokens, FP8 KV cache, and text-only loading.
It passes the API protocol suite, retrieves an exact needle from a 249,999-token chat prompt, reaches 114.0 tok/s isolated at medium reasoning effort, and reaches 633.6 tok/s on the warmed eight-stream trial.
The retained vLLM deployment resolved 206/300 SWE-bench Lite instances, the best result measured on this host.

## Result summary

| Metric | Result |
|---|---:|
| Checkpoint | `Qwen/Qwen3.8-27B-FP8` |
| Checkpoint revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Checkpoint size | 28.75 GiB |
| Model load | 28.02 GiB VRAM |
| Runtime image digest | `sha256:4a2f33a884222f7049b983263ad9976f89452bb81affecf5b67d89ad35c1bc31` |
| vLLM commit | `3a0914114705fa38d4c3171d0746c1a6b6f10209` |
| Context allocation | 262,144 tokens |
| Reserved VRAM at idle | 87,398 MiB |
| Physical VRAM free under load | about 9,844 MiB |
| GPU KV cache | 47.91 GiB, 1,351,725 tokens |
| Full-context cache concurrency | 5.16x at 262,144 tokens |
| API and tool protocol | 7/7 |
| 250K retrieval | Pass in 113.622 seconds |
| Medium isolated decode | 113.96 tok/s mean |
| Medium warmed eight-stream aggregate | 633.58 tok/s |
| Medium lightweight coding | 18/22 first pass, 22/22 best after two expression reruns |
| Xhigh lightweight coding at 16K | 10/22 |
| SWE-bench Lite, vLLM + MTP-3 | 206/300, 68.7% |
| SWE-bench Lite, SGLang + MTP-4 | 196/300, 65.3% |

## Retained server configuration

The first server used `--gpu-memory-utilization 0.90` and `--max-num-batched-tokens 32768`.
That configuration served short and concurrent workloads correctly, but a 250K prompt exhausted workspace memory during prefill and killed the engine.
The failure occurred while allocating a 2.08 GiB temporary buffer with only 218 MiB physically free.

The retained configuration uses:

- `--max-model-len 262144`.
- `--gpu-memory-utilization 0.85`.
- `--max-num-batched-tokens 16384`.
- `--max-num-seqs 64`.
- `--kv-cache-dtype fp8`.
- `--language-model-only`.
- `--enable-prefix-caching`.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Qwen3 reasoning and Qwen3-Coder tool parsers.

This reduces idle reserved VRAM from 90,524 MiB to 87,398 MiB and leaves enough temporary workspace for the real 250K prefill.
The resulting 47.91 GiB KV pool is still large enough for 1.35 million cached tokens, or 5.16 simultaneous full-context requests in the allocator's theoretical capacity calculation.

## Throughput

The final 85% memory configuration was measured at Qwen's recommended `temperature=1.0`, `top_p=0.95`, and `top_k=20` with medium reasoning effort.

| Workload | Throughput | TTFT |
|---|---:|---:|
| Isolated, three-run mean | 113.96 tok/s | 62.8 ms mean |
| One stream, two-trial mean | 116.48 tok/s | 62.9 ms mean |
| Eight streams, two-trial mean | 486.75 tok/s | 1.713 s mean |
| Eight streams, warmed second trial | 633.58 tok/s | 154 ms mean |

The first eight-stream trial included cold shape work and reached 339.91 tok/s.
The second trial reached 633.58 tok/s with 76.18% draft-token acceptance and 79.2 tok/s per agent.
The warmed result is essentially level with Muse Glimmer FP8 + DFlash-15 on this host, which measured 626.22 tok/s at eight streams.

## Context evidence

The long-context request contained 249,987 raw prompt tokens and 249,999 prompt tokens after the chat template.
It retrieved the exact six-digit needle from 75% depth in 113.622 seconds.
Peak observed VRAM during the request was 87,398 MiB, leaving about 9,844 MiB physically free.
The server reported roughly 25,034 prompt tok/s near completion of the chunked prefill.

The artifact is [`../experiments/qwen38_27b_fp8/context_250k.json`](../experiments/qwen38_27b_fp8/context_250k.json).

## Protocol behavior

The model passed all seven API checks.
It returned the exact basic response, produced a correct arithmetic answer with separated reasoning, emitted one correctly named tool call with exact arguments, and consumed the tool result without making another call.

The artifact is [`../experiments/qwen38_27b_fp8/api_smoke.json`](../experiments/qwen38_27b_fp8/api_smoke.json).

## Lightweight coding quality

Reasoning effort materially changes this model's behavior.

| Policy | String | Expression | A* | LRU + TTL | Total |
|---|---:|---:|---:|---:|---:|
| Xhigh, 16K output | 5/5 | 5/5 | 0/6 | 0/6 | 10/22 |
| Medium, 16K output | 5/5 | 1/5 | 6/6 | 6/6 | 18/22 |
| Medium expression rerun 2 | - | 5/5 | - | - | 5/5 |
| Medium expression rerun 3 | - | 5/5 | - | - | 5/5 |
| Xhigh, 32K selected hard tasks | - | - | 6/6 | 0/6 | 6/12 |

At xhigh with a 16,384-token allowance, A* and LRU consumed the entire budget in reasoning and emitted no usable code.
Increasing the allowance to 32,768 tokens recovered A* at 19,680 completion tokens and 206.13 seconds.
LRU still consumed all 32,768 tokens, produced 128,122 reasoning characters, and emitted no final code after 390.19 seconds.

Medium effort solved both hard tasks, including LRU, but the first expression-evaluator sample contained a parser bug and scored 1/5.
Two independent medium reruns of expression evaluation scored 5/5 each in 40.39 and 57.12 seconds.
The diagnostic best-of-three ceiling is therefore 22/22, but only the expression task received three medium trials in this first pass.

Medium is the better default for local agent work.
Xhigh should be opt-in, and callers should not assume that doubling the output budget will prevent reasoning loops.

## Comparison with nearby local deployments

| Deployment | Isolated decode | Eight streams | Lightweight | Full SWE-bench Lite |
|---|---:|---:|---:|---:|
| Qwen3.8-27B FP8 + MTP-3, medium | 113.96 tok/s | 633.58 warmed | 18/22 first, 22/22 diagnostic best | **206/300** |
| Qwen3.8-27B FP8 + MTP-4, SGLang | 109.3 tok/s | 827.58 at c15 | Pass | 196/300 |
| Qwen3.6-27B dynamic NVFP4 + MTP-2 | 113.2 tok/s | - | 21/22 best-of-three | 178/300 |
| Qwen3.6-27B FP8 + DFlash-15 | 197.5 tok/s | - | 22/22 best-of-three | 172/300 |
| Muse Glimmer 30B FP8 + DFlash-15 | 120.12 tok/s | 626.22 tok/s | 22/22 | 129/300 |
| Nemotron 3.5 Lightning NVFP4 + DSpark-3 | 455.61 tok/s | 1,777.59 tok/s | 11/22 native | Not run locally |

Qwen3.8 is not a raw throughput upgrade over the tuned Qwen3.6 DFlash stack.
It is roughly tied with Qwen3.6 dynamic NVFP4 in isolated speed and with Muse Glimmer at eight streams.
Its 206/300 SWE-bench Lite result is 28 cases above the former Qwen3.6 NVFP4 daily driver, which makes Qwen3.8 the new production choice despite the lower raw decode rate than Qwen3.6 DFlash.

## SWE-bench Lite

Both Qwen3.8 serving stacks completed all 300 predictions and finished official harness evaluation with zero harness errors.

| Stack | Workers | Resolved | Empty patches | Non-empty resolution |
|---|---:|---:|---:|---:|
| vLLM FP8 + MTP-3 | 8 | **206/300 (68.7%)** | 10 | 206/290 (71.0%) |
| SGLang FP8 + MTP-4 | 15 | 196/300 (65.3%) | 8 | 196/292 (67.1%) |

SGLang found a much higher serving saturation point and finished its agent phase in 9h19m52s at 15 workers.
That throughput advantage did not translate into the best completion quality: it trailed vLLM by 10 resolved cases, or 3.3 percentage points.
The serving stacks also used different reasoning controls, so this is a deployment-level selection rather than a clean runtime-only attribution.
The vLLM run used medium reasoning through the native Qwen parser, while the queued SGLang run enforced the recommended 4,096-token reasoning budget and used MTP-4.

The official reports are [`../sweagent_lite_qwen38_27b_fp8_mtp3_medium_c8.qwen38-27b-fp8-mtp3-medium-c8-full300.json`](../sweagent_lite_qwen38_27b_fp8_mtp3_medium_c8.qwen38-27b-fp8-mtp3-medium-c8-full300.json) and [`../sweagent_lite_qwen38_27b_fp8_sglang_mtp4_c15.qwen38-27b-fp8-sglang-mtp4-medium-budget4k-c15-full300.json`](../sweagent_lite_qwen38_27b_fp8_sglang_mtp4_c15.qwen38-27b-fp8-sglang-mtp4-medium-budget4k-c15-full300.json).

## Reproduction

Download the pinned checkpoint with:

```bash
bash tools/download_qwen38_27b_fp8.sh
```

Launch the retained server with:

```bash
bash tools/run_qwen38_27b_fp8_server.sh
```

Run the primary benchmark sequence with:

```bash
bash tools/benchmark_qwen38_27b_fp8.sh
```

The runtime cache is persisted under `/mnt/extended/gisenberg/models/.vllm-cache-qwen38` so later launches can reuse the compiled graph artifacts.

## Caveats

The dedicated Qwen3.8 image is a development build rather than a released vLLM tag because the gated-delta-net speculative-decoding fix is not yet in a stable release.
The image and checkpoint are pinned by immutable digest and revision for reproducibility.
vLLM warns that the checkpoint does not provide calibrated query and probability scales for FP8 KV attention and falls back to scale 1.0.
The exact 250K retrieval pass is encouraging, but it does not replace a BF16-KV quality A/B or a broader long-context benchmark.
The canonical artifacts are under [`../experiments/qwen38_27b_fp8/`](../experiments/qwen38_27b_fp8/).

References: [official FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).
