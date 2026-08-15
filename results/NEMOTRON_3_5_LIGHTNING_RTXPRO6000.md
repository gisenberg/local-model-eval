# NVIDIA Nemotron 3.5 Lightning NVFP4 on RTX Pro 6000

NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 runs entirely on one RTX Pro 6000 Blackwell 96 GB at a 262,144-token allocation.
The best balanced configuration is the official NVFP4 target with DSpark-3 through digest-pinned vLLM 0.27.1.
It reaches 455.61 tok/s in the isolated decode test, 1,777.59 tok/s at eight streams, 2,632.47 tok/s at sixteen streams, and passes an exact 250K-token retrieval probe.

## Result summary

| Metric | Result |
|---|---:|
| Target checkpoint | NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 |
| Target checkpoint size | 20.08 GiB |
| Target model load | 19.16 GiB VRAM |
| Retained draft configuration | DSpark-3 |
| Context allocation | 262,144 tokens |
| Reserved VRAM at idle | 91,346 MiB |
| Peak VRAM during 250K probe | 93,290 MiB |
| Isolated decode | **455.61 tok/s** |
| Eight-stream aggregate | **1,777.59 tok/s** |
| Sixteen-stream aggregate | **2,632.47 tok/s** |
| API and tool protocol | **7/7** |
| 250K retrieval | **Pass** |
| Native-thinking lightweight coding | **11/22** |
| Direct-answer lightweight coding | **13/22** |
| 4K capped-thinking lightweight coding | **8/22** |

The checkpoint's published maximum context is 1,048,576 tokens.
This deployment deliberately allocates 262,144 tokens because that covers the local 250K requirement while preserving generous concurrency and a safe physical-memory margin.

## Draft comparison

All configurations used the same target checkpoint, vLLM image, 262,144-token model length, FP8 KV cache, Humming MoE backend, FlashInfer attention and Mamba kernels, temperature 1.0, top-p 0.95, and no top-k override.
The isolated column is a three-run mean after one warmup, while the concurrency columns are means across two 512-token trials.

| Draft configuration | Isolated tok/s | 1 stream | 4 streams | 8 streams | 16 streams | 262K cache slots |
|---|---:|---:|---:|---:|---:|---:|
| No draft | 373.24 | 362.19 | 883.06 | 1,302.67 | 1,926.75 | 76.36 |
| **DSpark-3** | **455.61** | **493.15** | 954.97 | **1,777.59** | 2,632.47 | **65.20** |
| DSpark-7 | 496.75 | 453.84 | **1,156.47** | 1,606.26 | **2,643.14** | 57.60 |
| DFlash-3 | 333.58 | 350.19 | 735.72 | 1,335.20 | 1,846.36 | 38.46 |
| DFlash-7 | 402.27 | 393.77 | 897.03 | 1,349.23 | 1,907.66 | 35.57 |

DSpark-3 improves isolated decode by 22.1% and eight-stream aggregate throughput by 36.5% over the no-draft target.
DSpark-7 wins the isolated and four-stream measurements, but its longer draft reduces eight-stream throughput and consumes more cache capacity.
DSpark-3 is therefore the better default for concurrent agents, while DSpark-7 is worth considering for interactive or four-worker workloads.

DFlash does not pay for its overhead on this stack.
DFlash-3 is 10.6% slower than the target alone in isolated decode, and DFlash-7 provides only a 7.8% isolated gain while remaining essentially flat at eight and sixteen streams.
The DFlash-3 concurrency workload accepted about 1.2 to 1.4 target tokens per draft step, versus roughly 1.7 for DSpark-3.

## Context evidence

The retained DSpark-3 server reports a 17,091,120-token GPU cache and a theoretical 65.20-way concurrency at the full 262,144-token allocation.
The real retrieval request contained 249,985 raw tokens and 250,001 prompt tokens after the chat template.
It returned the exact six-digit needle from 75% depth in 21.806 seconds.
VRAM rose from 91,346 MiB reserved at idle to 93,290 MiB during the request, leaving 3,952 MiB physically free.

The artifact is [`../experiments/nemotron35_lightning_nvfp4/dspark3/long_context_250k.json`](../experiments/nemotron35_lightning_nvfp4/dspark3/long_context_250k.json).

## Protocol behavior

The DSpark-3 server passed all seven API checks.
It produced the exact basic response, returned a correct arithmetic answer with separated reasoning, emitted one correctly named tool call with exact arguments, and consumed the tool result without calling the tool again.

The artifact is [`../experiments/nemotron35_lightning_nvfp4/dspark3/api_smoke.json`](../experiments/nemotron35_lightning_nvfp4/dspark3/api_smoke.json).

## Lightweight quality

The local 22-point coding benchmark exposed a reasoning-control problem that matters operationally.
Native unbounded thinking scored 11/22 because the LRU task consumed all 16,384 completion tokens in reasoning and emitted no code.
The per-task native scores were 4/5, 1/5, 6/6, and 0/6.

vLLM 0.27.1 exposes `thinking_token_budget`, but DSpark uses its V2 model runner and logs that this request field is unsupported.
DFlash accepts the field through the V1 runner, but a 4,096-token cap scored only 8/22 and sometimes moved an unfinished reasoning trace into final content after truncation.
The cap is therefore not a safe production workaround.

The checkpoint's supported direct-answer template mode uses `chat_template_kwargs={"enable_thinking": false}`.
It eliminated reasoning spill and solved all six LRU tests, but the single stochastic pass scored only 13/22 overall.
These single-sample local scores are useful failure-mode probes, not substitutes for the checkpoint's broader published agentic evaluation.
NVIDIA reports 52.8 on SWE-bench Verified for the NVFP4 checkpoint, but a local full SWE-bench run has not yet been completed.

## Reproducible deployment

The target checkpoint is pinned to revision `0dcd680e5585c791728c83342b311d0a0026dbeb`.
The DSpark checkpoint is pinned to revision `d10c6ff40d6e69d1f92e407e027de3eafdb77645`.
The DFlash checkpoint is pinned to revision `7fc1f1ff4b82b917efbd0710df0872c2bb89caa5`.
The vLLM 0.27.1 amd64 image is pinned to manifest digest `sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2` and tagged locally as `local/vllm-nemotron35:v0.27.1`.

Download the pinned checkpoints with:

```bash
bash tools/download_nemotron35_lightning.sh
```

Launch the recommended server with:

```bash
bash tools/run_nemotron35_lightning_server.sh dspark 3
```

The launch configuration uses:

- `--max-model-len 262144`.
- `--gpu-memory-utilization 0.94`.
- `--max-num-seqs 16`.
- `--max-num-batched-tokens 16384`.
- FP8 KV cache with prefix caching.
- Humming NVFP4 MoE kernels.
- FlashInfer attention and Mamba kernels.
- Aligned FP16 Mamba state cache with stochastic rounding.
- Nemotron v3 reasoning parsing and Qwen3-Coder tool parsing.
- DSpark with three speculative tokens.

The server keeps its compiled vLLM and FlashInfer artifacts under `/mnt/extended/gisenberg/models/.vllm-cache-nemotron35` so subsequent launches reuse the expensive kernel compilation.

## Runtime caveats

vLLM reports that this RTX Pro 6000 workstation path lacks native dense FP4 execution and uses Marlin weight-only FP4 kernels for the dense NVFP4 layers.
The MoE path still uses Humming NVFP4 kernels.
The target FP8 attention checkpoint does not provide calibrated query and probability scaling factors, so vLLM falls back to uncalibrated values and warns that accuracy may be affected.
The draft checkpoints also warn that fused parallel projections have differing global NVFP4 scales, which may reduce draft accuracy and acceptance.

The canonical local artifacts are under [`../experiments/nemotron35_lightning_nvfp4/`](../experiments/nemotron35_lightning_nvfp4/).
