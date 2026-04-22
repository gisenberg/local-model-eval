# NVFP4 vs FP8 vs BF16 — Qwen3.6-27B on RTX Pro 6000

Three-way comparison of Qwen3.6-27B at different precisions on the 4-bench coding suite, served via vLLM on RTX Pro 6000 Blackwell 96 GB.

## Results

| Variant | Weights | Best-of-3 | Avg | Tok/s | Size | Source |
|---|---|---|---|---|---|---|
| **BF16** (baseline) | `Qwen/Qwen3.6-27B` | 21/22 (95%) | 18.4/22 (84%) | 29 | 52 GB | HF release |
| **FP8** (vendor) | `Qwen/Qwen3.6-27B-FP8` | 21/22 (95%) | 20.0/22 (91%) | 47 | 29 GB | HF release |
| **NVFP4** (modelopt) | `mmangkad/Qwen3.6-27B-NVFP4` † | 22/22 (100%) | 18.6/22 (85%) | 46 | 29 GB | community, modelopt 0.42 |

† See [NVFP4 checkpoint note](#nvfp4-checkpoint-note) below — we self-quantized with modelopt 0.43 but the export didn't load in vLLM 0.19.1, so we swapped to a community checkpoint with the same methodology.

**Headline**: FP8 is the sweet spot. It matches BF16 on best-of-3, has the highest avg score, is 1.6× faster, and halves the memory footprint. NVFP4 gets the only 100% best-of-3 but loses ~1.5 points on the avg because it produced one zero run on A*; measured throughput was the same as FP8 (46 vs 47 tok/s) despite the smaller weight footprint, suggesting vLLM's NVFP4 path on Blackwell isn't yet hitting the native FP4 tensor cores through flashinfer-cutlass JIT.

## Per-benchmark

Best/avg out of 3 runs at T=0.3, max_tokens=15000, max-model-len=16384.

| Benchmark | BF16 best/avg | FP8 best/avg | NVFP4 best/avg |
|---|---|---|---|
| Expression Evaluator (5) | 5 / 4.0 | 5 / 4.3 | 5 / 4.3 |
| A* Pathfinding (6) | 5 / 4.7 | 5 / 5.0 | **6** / 3.3 * |
| LRU Cache w/ TTL (6) | 6 / 5.0 | 6 / 5.7 | **6 / 6.0** |
| String Processor (5) | 5 / 4.7 | 5 / 5.0 | 5 / 5.0 |
| **Total** | **21 / 18.4** | **21 / 20.0** | **22 / 18.6** |

\* NVFP4 A* run 1 produced no extractable code block (0/6). Run 2 was 4/6, run 3 was 6/6. Variance is real; the 100% best-of-3 is one lucky run, not a consistent quality lead.

## Methodology

- **Hardware**: RTX Pro 6000 Blackwell 96 GB, driver 580.126.09, CUDA 12.8 runtime.
- **Serving**: vLLM 0.19.1 with transformers built from main (required for `qwen3_5` arch).
- **Bench**: [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py) — identical 4-benchmark suite as `nvfp4_gemma31b_bench_v2.py`, 3 runs each, T=0.3. Raw outputs in [`experiments/nvfp4_qwen36_27b/`](../experiments/nvfp4_qwen36_27b/).
- **Quantization**: [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — NVIDIA Model Optimizer, `MAMBA_MOE_NVFP4_CONSERVATIVE_CFG` + additional `*linear_attn*` exclusion for Qwen3.5's gated-delta-net layers, calibrated on `abisee/cnn_dailymail` (128 samples, seq_len 2048) — matching NVIDIA's reference recipe for their Qwen3-NVFP4 releases.
- **Thinking**: Qwen3.6 is a reasoning model. The bench strips `<think>...</think>` before extracting code blocks (extraction regex would otherwise glue scratchwork fragments into invalid Python).

## NVFP4 checkpoint note

We produced our own NVFP4 checkpoint with `modelopt 0.43.0` using the config above. The resulting weights have the same `quantization_config.ignore` list (66 entries) as `mmangkad/Qwen3.6-27B-NVFP4` (67 entries, same pattern) — all `linear_attn` and some `self_attn` layers stay BF16, MLPs get NVFP4.

Our export loads into vLLM 0.19.1 but crashes during forward pass:

```
File ".../vllm/model_executor/models/qwen3_next.py", line 500, in forward
KeyError: 'weight_scale'
```

The `mmangkad` checkpoint (produced with `modelopt 0.42.0rc1.dev107`, nearly identical ignore list) loads and runs cleanly. The bench numbers above are from `mmangkad`'s checkpoint. The quality result — that a modelopt NVFP4 of this model can match BF16 on best-of-3 — still holds; we just didn't produce the checkpoint ourselves in this session.

The diagnosis is incomplete: it looks like modelopt 0.43 changed something in how quantized state is serialized for hybrid `linear_attn` + `self_attn` models that vLLM's `qwen3_next` loader doesn't handle yet. Not chased further in this session — pinning `nvidia-modelopt==0.42.*` would likely reproduce a working self-quantized checkpoint.

## Build notes

Getting modelopt + vLLM + flashinfer FP4 working end-to-end on a fresh Ubuntu 24.04 box required more than `pip install`:

1. `sudo apt install -y build-essential python3.12-dev` — Triton JIT needs `gcc` and `Python.h`.
2. `uv pip install ... nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl ninja` — flashinfer's FP4 kernels JIT-compile cutlass code that needs CUDA 13+ vector types (`ulonglong4_16a` etc.). The CUDA 12.x toolchain `apt` ships (`nvidia-cuda-toolkit`) won't compile it.
3. Overlay CUDA 12 libs (`curand`, `cublas`, `cufft`, `cudnn`, `cooperative_groups/`) into the CUDA 13 include tree — flashinfer expects a single `$CUDA_HOME` with everything, but the CUDA 13 PyPI split doesn't include those. Also symlink `libcudart.so` and `libcuda.so` stubs so `ld` resolves `-lcudart -lcuda`.
4. Uninstall `deepspeed` (modelopt pulls it as transitive; its import-time `installed_cuda_version()` raises `MissingCUDAException` if `CUDA_HOME` isn't set before anything else runs).
5. `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` (the default auto-pick works once the JIT chain is fixed; `cutlass` without flashinfer produces garbage output on SM 120, `marlin` errors on layer dims not divisible by 64).

Total wall-clock on first-time setup: ~15 min for the toolchain after the ~25 min pip install + 15 min model downloads. See `/home/gisenberg/venvs/qwen36-nvfp4/env.sh` for the final env, and `/home/gisenberg/venvs/qwen36-nvfp4/lib/python3.12/site-packages/nvidia/cu13/` for the header overlay.

## Why NVFP4 doesn't beat FP8 on throughput here

On paper Blackwell FP4 tensor cores should be ~2× FP8's throughput. We saw 46 vs 47 tok/s — effectively a tie. Two likely reasons:

- **Only MLPs are quantized.** Both the `linear_attn` (gated-delta-net) blocks and the `self_attn` blocks stay in BF16 per the MAMBA-MoE conservative config. For a hybrid architecture where attention + gating is a large fraction of FLOPs, halving MLP precision gives a smaller speedup than for a pure-MLP-heavy dense transformer.
- **flashinfer-cutlass JIT is not the fastest FP4 path on SM 120.** The `flashinfer-trtllm` backend would use pre-compiled TRT-LLM kernels (likely better tuned), but failed to compile in our env. The `cutlass` (non-flashinfer) backend loaded but produced garbage output — looks like a correctness bug in vLLM 0.19.1's native cutlass NVFP4 path for hybrid Qwen3.5.

A Qwen3.6-27B that kept every Linear quantized (no `linear_attn`/`self_attn` exclusions) would almost certainly decode faster, but the "quantize everything" attempt produced garbage — modelopt's default config doesn't survive the hybrid architecture's gated-delta-net projections.

## Files

- [`experiments/nvfp4_qwen36_27b/bf16/results.json`](../experiments/nvfp4_qwen36_27b/bf16/results.json)
- [`experiments/nvfp4_qwen36_27b/fp8/results.json`](../experiments/nvfp4_qwen36_27b/fp8/results.json)
- [`experiments/nvfp4_qwen36_27b/nvfp4/results.json`](../experiments/nvfp4_qwen36_27b/nvfp4/results.json)
- [`tools/quantize_nvfp4.py`](../tools/quantize_nvfp4.py) — reusable for other models
- [`tools/nvfp4_qwen36_27b_bench.py`](../tools/nvfp4_qwen36_27b_bench.py)
