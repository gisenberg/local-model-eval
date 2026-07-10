# vLLM-Moet DeepSeek V4 Flash Smoke Test

Date: 2026-07-09 PDT

## Setup

- Runtime: `kacper-daftcode/vLLM-Moet` at commit `591250b`
- Docker image: `vllm-moet-sm120:v024`, built locally from `Dockerfile.sm120-v024`
- Image size after build: 33.9 GB
- Model: `deepseek-ai/DeepSeek-V4-Flash`, official FP8/NVFP4 Hugging Face checkpoint
- Model placement for test: `/home/gisenberg/models/deepseek-v4-flash` on the root NVMe volume
- Local model footprint before cleanup: 149 GB, 46 safetensors shards
- Host: local homelab Linux box, AMD EPYC 4585PX, 128 GB RAM, 64 GB swap
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM

The local Docker daemon initially lacked the NVIDIA container runtime. I installed
`nvidia-container-toolkit` 1.19.1, ran `nvidia-ctk runtime configure --runtime=docker`,
and reloaded Docker. `--runtime=nvidia` worked without restarting existing service
containers; `--gpus all` still appeared to need a full Docker restart.

## Claimed Path

The vLLM-Moet README claims a one-card DeepSeek V4 Flash path on PRO 6000 using:

- `VLLM_MOE_W2=1`
- `VLLM_MOE_W2_DELTA_GB=1`
- FP8 KV cache
- MTP with `deepseek_mtp`, 2 draft tokens
- 24K model context in the sample command

The same repo's `docs/v024-port.md` also claims a 512K one-card mode when the delta
pool is removed or auto-sized.

## Attempt 1

Config:

- `--max-model-len 24576`
- `--kv-cache-dtype fp8`
- `--block-size 256`
- `--gpu-memory-utilization 0.95`
- `--max-num-batched-tokens 1024`
- `--max-num-seqs 4`
- `--tokenizer-mode deepseek_v4`
- `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'`
- `VLLM_MOE_W2=1`
- `VLLM_MOE_W2_DELTA_GB=1`

Result:

- Server never reached `/v1/models`.
- Engine reached the DeepSeek V4 architecture path with FP8 KV and MTP enabled.
- It selected the FP4 expert path and the DeepGEMM MXFP4 MoE backend.
- The engine process was OOM-killed during load.
- GPU memory use was only around 10-13 GB at failure time.
- Kernel OOM log showed the engine process at about 101 GiB anonymous RSS.

## Attempt 2

Config was the same 24K/MTP command, but with the delta pool disabled:

- `VLLM_MOE_W2=1`
- `VLLM_MOE_W2_DELTA=0`
- `VLLM_MOE_W2_DELTA_GB=0`

Observed container log:

- vLLM version: 0.24.0
- Resolved main architecture: `DeepseekV4ForCausalLM`
- Resolved draft architecture: `DeepSeekV4MTPModel`
- Effective max model length: 24,576
- KV cache dtype: FP8
- Quantization: `deepseek_v4_fp8`
- Expert dtype resolved to `fp4`
- MoE backend: `DEEPGEMM_MXFP4`
- Lightning Indexer used FP8 indexer cache
- The APIServer warned that `VLLM_MOE_W2`, `VLLM_MOE_W2_DELTA`,
  `VLLM_MOE_W2_DELTA_GB`, and `VLLM_MOE_W2_CUBIT_DIR` were unknown vLLM
  environment variables. These variables are present in the vLLM-Moet patch/docs,
  so the warning is not enough to prove they were ignored, but it is worth recording.

Result:

- Server again never reached `/v1/models`.
- Docker state: `OOMKilled=true`, `ExitCode=1`.
- Finish time: `2026-07-10T03:52:23Z`.
- GPU memory use was around 13 GB at failure time.
- Kernel OOM log showed the engine process at about 117 GiB anonymous RSS.

## Cleanup

- The model snapshot at `/home/gisenberg/models/deepseek-v4-flash` was removed.
- Swap was restored to 0B used after the OOM runs.
- The `vllm-moet-sm120:v024` Docker image was left in place.
- The exited `vllm-moet-ds4-smoke` container was left in place as a log source.

## Read

This path did not produce a usable one-card DeepSeek V4 Flash server on our host.
The failure happened before any prompt or benchmark could run, so there is no quality
or throughput result to compare.

The concrete blocker was host RAM during model load/staging, not VRAM. The GPU was
mostly idle and far below capacity when the engine died, while the host OOM killer
terminated the vLLM engine at more than 100 GiB anonymous RSS. On this 128 GB RAM
plus 64 GB swap host, vLLM-Moet's DeepSeek V4 Flash path needs either substantially
lower host-side load pressure, more RAM, or a different loader configuration before
it can be benchmarked.
