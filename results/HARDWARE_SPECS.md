# Hardware Spec Sheet — Benchmarked Machines

This document captures the hardware we run benchmarks on, with a focus on the specs that actually matter for local LLM inference. Numbers come from official NVIDIA documentation, our own measurements, and the bandwidth/throughput data we've gathered through benchmarking.

## TL;DR

| | RTX 5090 (Workstation) | M4 Max MacBook Pro 16" | DGX Spark (GB10) |
|---|---|---|---|
| **Total memory** | 32 GB GDDR7 | 36 GB LPDDR5X (unified, **~30 GB usable on Metal**) | 128 GB LPDDR5x (unified) |
| **Memory bandwidth** | **1,792 GB/s** | **410 GB/s** | **273 GB/s** |
| **Max model (dense, Q4)** | ~31 GB at 53 tok/s | ~18 GB at **11.5 tok/s** (gemma 31B, turbo4 KV req'd) | ~70 GB but at <8 tok/s |
| **Max model (MoE A4B, Q6)** | 142 tok/s | **60 tok/s** (gemma 26B-A4B, 16K ctx max) | ~70 GB at 21 tok/s |
| **Max model (MoE A10B, Q4)** | doesn't fit | doesn't fit | ~70 GB at 21 tok/s |
| **Best for** | Dense models, max throughput, code generation | Portable inference, MoE A4B-class models | Huge MoE models, full context windows |
| **Bottleneck** | VRAM capacity | Metal working set (30 GB) | Memory bandwidth |

**The headline insight:** these three machines occupy distinct points on the capacity vs. bandwidth tradeoff curve. The 5090 has compute and bandwidth in abundance but is starved for capacity. The Spark has capacity in abundance but is starved for bandwidth. The M4 Max sits between them — moderate capacity, moderate bandwidth — and complements both with portability and an entirely different software stack (Metal/MLX). Almost no model class is "good on all three"; picking the right machine for a model is more important than picking the right model.

---

## NVIDIA RTX 5090 (Workstation)

### Specs

| Component | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5090 (GB202, "Blackwell") |
| **Architecture** | Blackwell, sm_120 |
| **CUDA cores** | 21,760 |
| **Tensor cores** | 680 (5th gen, FP4 native support) |
| **VRAM** | 32 GB GDDR7 |
| **Memory bus** | 512-bit |
| **Memory bandwidth** | **1,792 GB/s** |
| **Compute (FP16)** | ~104 TFLOPS |
| **Compute (FP8)** | ~419 TFLOPS |
| **Compute (FP4 dense)** | ~838 TFLOPS |
| **Compute (FP4 sparse)** | ~3,352 TFLOPS |
| **TDP** | 575 W |
| **Form factor** | Discrete GPU (PCIe 5.0 x16) |

### Host system

| Component | Value |
|---|---|
| **CPU** | (host system, not GPU-relevant for inference throughput) |
| **OS** | Windows 11 Pro |
| **CUDA toolkit** | 13.2 |
| **Inference stack** | llama.cpp (TurboQuant fork), LM Studio, vLLM (via WSL2) |

### Key constraints

- **VRAM ceiling at 32 GB.** Anything bigger spills to system RAM via PCIe and crashes throughput.
- Bandwidth is **never the bottleneck** for any model that fits — even a dense 31B model at Q4 reads ~17.5 GB/token, which would theoretically allow ~100 tok/s on this bandwidth alone (we measure 53 tok/s, compute-bound).
- **FP4 tensor cores are unique to Blackwell.** NVFP4-quantized models (like LilaRest's Gemma 31B NVFP4-turbo) can use the dedicated FP4 path through CUTLASS kernels, but only via vLLM with the cu130 wheel.

### What this machine is good at

- **Dense models up to ~31B at Q4.** Gemma 4 31B-IT is the canonical example: 53 tok/s, 17/17 quality, 22 GB total VRAM with TurboQuant turbo4 KV.
- **Anything compute-heavy.** FP4 kernels have ~3 PFLOPS of headroom — far more than the bandwidth can feed for a single user, so this machine excels at concurrent batched serving where compute matters.
- **High-quality code generation.** Our entire S-tier (Gemma 26B, Gemma 31B, fine-tuned Qwen 27Bs) runs at interactive speed with TurboQuant.

### What this machine is bad at

- **Anything that doesn't fit in 32 GB.** Models like Qwen 3.5 122B-A10B (72 GB weights) cannot be loaded — even though their per-token bandwidth would suit the GPU well.
- **Long contexts on dense models.** Gemma 4 31B with f16 KV maxes out at ~16K context before running out of VRAM. TurboQuant turbo4 extends this to ~58K but no further.
- **Models with expensive KV caches.** Dense transformers with global attention pay heavily for context length on a 32 GB budget.

---

## Apple MacBook Pro 16" (M4 Max, 2024)

### Specs

| Component | Value |
|---|---|
| **SoC** | Apple M4 Max (binned: 14C CPU / 32C GPU) |
| **Architecture** | Apple Silicon, ARM64 |
| **CPU** | 14 cores (10 performance + 4 efficiency) |
| **GPU** | 32-core Apple GPU |
| **Neural Engine** | 16-core |
| **Unified memory** | **36 GB LPDDR5X** |
| **Memory bandwidth** | **410 GB/s** |
| **Storage** | 1 TB SSD (configurable) |
| **Form factor** | 16" laptop |

Note: The "full" M4 Max (16C CPU / 40C GPU) ships with 48/64/128 GB and 546 GB/s bandwidth — a different chip binning. The 36 GB configuration uses the lower-bandwidth variant.

### Host system

| Component | Value |
|---|---|
| **OS** | macOS |
| **GPU API** | Metal Performance Shaders |
| **Inference stack** | llama.cpp (Metal backend), MLX framework, LM Studio, Ollama |

### Key constraints

- **Unified memory means no PCIe spill.** Like the Spark, weights and KV cache share a single coherent pool. Either it fits or it doesn't — no slow degradation from RAM offload.
- **Metal working set is 30 GB, not 36 GB.** macOS reserves the rest of unified memory for the system. `recommendedMaxWorkingSetSize ≈ 30150 MB` is the *hard* ceiling for `weights + KV cache + compute buffers`. Going over produces `kIOGPUCommandBufferCallbackErrorOutOfMemory`, not graceful degradation. So in practice the Mac has less effective memory than the spec sheet implies.
- **Bandwidth sits in the middle.** 410 GB/s is ~1.5x the Spark and ~4.4x slower than the 5090. This puts it in an interesting "Goldilocks zone" for mid-sized models.
- **No FP4 tensor cores.** Apple Silicon GPUs don't have NVIDIA-style tensor cores at all — Metal uses general-purpose compute units. NVFP4 models cannot run here. The practical quantization options are llama.cpp Q-formats (Q4_K_M, Q6_K, etc.) and MLX-native quants.
- **TurboQuant fork DOES build on Metal.** Earlier versions of this doc said otherwise — corrected after building `feature/turboquant-kv-cache` (commit 8590cbff9) on M4 Max with `cmake -DGGML_METAL=ON`. The Metal turbo3/turbo4 KV cache kernels work. **However, on this hardware turbo4 KV is *slower* than f16** (see measured numbers below) — the dequant compute overhead exceeds the KV bandwidth savings. Use turbo4 only when you need the memory savings, not for speed.

### Measured throughput on M4 Max 36 GB

Real numbers from `m4max_bench/`, llama.cpp turboquant fork build 8590cbff9, single-shot temp 0:

| Model | Quant | KV | Ctx | Tok/s | Code score | Notes |
|---|---|---|---|---|---|---|
| Nemotron 3 Nano 4B | Q4_K_M | f16 | 32K | **65** | 7/17 | Smallest, fastest |
| Gemma 4 26B-A4B (MoE, 4B active) | Q6_K | f16 | 16K | **60** | 15/17 | 32K f16 OOMs |
| Gemma 4 26B-A4B | Q4_K_M | f16 | 32K | 59 | 11/17 | Q4 quality drop is real |
| Gemma 4 26B-A4B | Q6_K | turbo4 | 16K | 46 | 16/17 | Turbo4 *slower* than f16 |
| Qwen 3.5 9B (dense) | Q4_K_M | f16 | 32K | 35 | 9/17 | Thinking off |
| Qwen 3.5 27B Opus-Distilled (dense) | Q4_K_M | f16 | 32K | 13 | 11/17 | Bandwidth-limited |
| Gemma 4 31B-IT (dense) | Q4_K_M | turbo4 | 16K | **11.5** | **17/17** | Requires turbo4 to fit |

**Reality check:** the bandwidth-math projections (deleted from this section) overestimated by 30-40% for everything ≥27B. A dense 27B at Q4 actually runs at 13 tok/s, not the projected 18. A dense 31B at Q4 (turbo4 KV) runs at 11.5 tok/s, not 16-18. The bandwidth ceiling is real, but kernel efficiency and KV-cache compute overhead eat more of the budget than the back-of-envelope formula assumed.

**Interactive zone:** Up to ~26B-A4B (MoE) and ~9B (dense) at Q4 with f16 KV. 27B+ dense models cross into "wait for it" territory at 11-13 tok/s.

### What this machine is good at

- **Mid-size MoE models on the go.** Gemma 4 26B-A4B at Q6_K runs at 60 tok/s with 15/17 quality on a laptop. That's a credible code-generation experience away from the desk.
- **Highest-quality coding model under 25 GB.** Gemma 4 31B-IT Q4_K_M turbo4 hits the same 17/17 score as on a 5090, just at 11.5 tok/s instead of 53. Slow but correct.
- **Long battery life inference.** ~30 W power draw under sustained load vs the 5090's 575 W. You can run a coding agent for hours on battery.
- **Workloads that need portability.** This is the only machine in the lineup you can put in a backpack.

### What this machine is bad at

- **Anything above ~25 GB of weights.** The Metal working set is ~30 GB, not 36. A 22 GB model (Gemma 4 26B-A4B Q6_K) leaves ~8 GB for KV+compute. A 25 GB model leaves none. Long contexts on bigger models are simply not possible at any KV format.
- **NVFP4 / NVIDIA-specific formats.** The model ecosystem is increasingly NVIDIA-flavored. AWQ, GPTQ, and NVFP4 have varying levels of Apple support; GGUF + MLX are the safe paths.
- **Maximum throughput.** A 5090 will always smoke this on raw tok/s for any model that fits both. The M4 Max trades throughput for portability and lower power.
- **MLX has a model-class-dependent story.** Tested `mlx_lm.server` 0.31.2 against `mlx-community/*` quants on the same prompts. MLX *wins* on dense ≥27B models (Qwen 3.5 27B Opus-Distilled MLX 4bit: 18.5 tok/s vs llama.cpp 13.0 — **+42%**, hitting ~99% of bandwidth). MLX *loses* on MoE models (Gemma 4 26B-A4B 6bit MLX: 33 tok/s vs llama.cpp Q6_K 60 — **-45%**, only ~49% bandwidth utilization). Likely MLX's MoE kernels don't exploit the sparse activation pattern as efficiently as llama.cpp's. For dense models, MLX is the best option; for MoE, llama.cpp wins.

### Status in this repo

Benchmarked April 2026. See [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) for the full per-model tier list with quality scores, and [`m4max_bench/`](../m4max_bench/) for the raw outputs.

Headline findings:
- Dense 27B+ runs hit 11-13 tok/s — bandwidth-bound, matches the lower end of the projection range.
- MoE 26B-A4B (4B active) runs at 60 tok/s — by far the best speed/quality combo on this machine.
- TurboQuant KV cache: builds and works on Metal, but is *slower* than f16 KV here (the dequant compute overhead exceeds the bandwidth savings on a bandwidth-constrained platform). Use turbo4 only when you need its capacity savings (e.g. Gemma 4 31B which can't run with f16 KV at all).
- The MLX comparison contradicted the previous "MLX is 10-30% faster" footnote — see the bullet above.

---

## NVIDIA DGX Spark (GB10 Grace Blackwell)

### Specs

| Component | Value |
|---|---|
| **SoC** | NVIDIA GB10 Grace Blackwell Superchip |
| **GPU die** | Blackwell, fifth-gen tensor cores, 6,144 CUDA cores, 192 tensor cores, FP4 native support |
| **CPU die** | 20 ARM cores (10× Cortex-X925 + 10× Cortex-A725), 3 nm TSMC |
| **Interconnect** | NVLink-C2C (CPU↔GPU coherent) |
| **Unified memory** | **128 GB LPDDR5x** (~120 GB usable) |
| **Memory speed** | 8,533 MT/s |
| **Memory bandwidth** | **273 GB/s** |
| **Compute (FP4)** | Up to ~1 PFLOP (sparse) |
| **Storage** | 4 TB NVMe |
| **Form factor** | Compact desktop workstation |

### Host system

| Component | Value |
|---|---|
| **OS** | Ubuntu (aarch64 Linux) |
| **CUDA toolkit** | 13.0 |
| **Inference stack** | llama.cpp (standard build), TurboQuant fork |

### Key constraints

- **Bandwidth is the dominant constraint, not capacity.** 273 GB/s sounds reasonable but is ~6.6x slower than the 5090's GDDR7. Per-token decoding is the rate-limiter for any model class except small-active MoE.
- **128 GB of unified memory** means weights and KV cache share the same pool. Loading huge models is not the issue — getting tokens out of them is.
- **No PCIe bottleneck.** The CPU and GPU share memory directly via NVLink-C2C, so there is no "spill to system RAM" problem like the 5090 has. Either it fits or it doesn't.

### Bandwidth-driven performance ceilings

For autoregressive decoding, every generated token requires reading the model's *active* parameters once. On a 273 GB/s pipe:

| Active params (Q4 weights) | Theoretical max | Practical (~80% util) |
|---|---|---|
| 3B | 135 tok/s | ~80–110 tok/s |
| 4B | 100 tok/s | ~60–80 tok/s |
| 10B | 45 tok/s | ~21–35 tok/s |
| 31B (dense) | 8 tok/s | **~6.7 tok/s** (measured) |
| 70B (dense) | ~4 tok/s | ~3 tok/s |

Anything above ~10B active parameters becomes painfully slow on this machine.

### What this machine is good at

- **MoE models with small active parameter counts.** Qwen 3.5 122B-A10B (10B active) hits 21 tok/s at A-tier code quality. Qwen3-Coder-Next 80B-A3B hits 50 tok/s.
- **Huge context windows.** With ~120 GB free for KV cache, you can run any model at its full trained context length without quantization.
- **Memory-hungry workloads** that don't generate tons of tokens: long-context reading comprehension, large-batch embedding generation, fine-tuning experiments up to 70B.

### What this machine is bad at

- **Dense large models.** Gemma 31B Q8_0 measured at 6.7 tok/s — three minutes to generate 5K characters of code. Loads fine, runs unusably.
- **Single-user interactive code generation on dense models.** Even when the model fits, the bandwidth math prevents interactive speeds.
- **Anything where the 5090's tensor cores would shine.** The Spark has FP4 tensor cores too, but bandwidth is the ceiling — you can't feed those tensor cores fast enough to use them.

---

## Side-by-Side Comparison

### Memory and bandwidth

| Metric | RTX 5090 | M4 Max 36 GB | DGX Spark |
|---|---|---|---|
| Capacity (raw) | 32 GB | 36 GB | 128 GB |
| Capacity (usable for GPU) | ~30 GB (CUDA) | **~30 GB (Metal working set)** | ~120 GB |
| Bandwidth | 1,792 GB/s | 410 GB/s | 273 GB/s |
| Bandwidth ratio (vs 5090) | 1.0x | 0.23x | 0.15x |
| Capacity ratio (vs 5090) | 1.0x | 1.13x raw / 1.0x usable | 4.0x |
| Largest dense Q4 model viable | ~31B (53 tok/s) | ~31B (**11.5 tok/s, turbo4 only**) | ~7B for interactive |
| Largest MoE A10B model viable | ~35B total | ~30B total | ~122B total |
| Power draw | 575 W | ~30 W | ~140 W |

### Measured throughput on the same model class

Where we have comparable runs:

| Model | RTX 5090 | M4 Max | DGX Spark | Notes |
|---|---|---|---|---|
| Gemma 4 31B-IT Q4_K_M (dense) | **53 tok/s** | **11.5 tok/s** (turbo4 req'd) | ~6.7 tok/s | All measured. M4 Max needs turbo4 KV to fit. |
| Qwen 3.5 35B-A3B Q4_K_M (MoE, 3B active) | **188 tok/s** | not tested | not tested | 5090 only — compute headroom matters here |
| Qwen 3.5 122B-A10B Q4 (MoE, 10B active) | **does not fit** | does not fit | 21 tok/s | Spark wins by capacity |
| Qwen3-Coder-Next 80B-A3B (MoE, 3B active) | does not fit | does not fit | 50 tok/s | Spark wins by capacity |
| Gemma 4 26B-A4B Q6_K (MoE, 4B active) | **142 tok/s** | **60 tok/s** (16K ctx max) | not tested | 5090 wins on bandwidth, M4 Max ctx-limited |
| Qwen 3.5 27B Opus-Distilled Q4_K_M (dense) | 64 tok/s | **13 tok/s** | not tested | Bandwidth ratio holds (~5x slower) |
| Qwen 3.5 9B Q4_K_M (dense, thinking off) | not in baseline | **35 tok/s** | not tested | M4 Max measured |
| Nemotron 3 Nano 4B Q4_K_M (dense) | not in baseline | **65 tok/s** | not tested | Smallest, fastest on Mac |

### Code quality (where comparable)

Quality is determined by the model, not the hardware — same GGUF file produces the same scores within sampling variance. The hardware just gates which models you can run.

| Model | Score | Where it's actually usable |
|---|---|---|
| Gemma 4 31B-IT Q4_K_M | 17/17 (100%) | **5090 only** (53 tok/s vs Spark's 6.7) |
| Gemma 4 26B-A4B Q6_K | 17/17 (100%) | 5090 (142 tok/s) |
| Qwen 3.5 122B-A10B Q4_K_M | 16/17 (94%) | **Spark only** (21 tok/s vs 5090 OOM) |
| Harmonic 27B Q4_K_M | 15/17 (88%) | 5090 only (Spark untested) |

---

## How to Pick a Machine for a Model

The decision tree:

```
Does the model fit in 32-36 GB at usable quant?
├── Yes → Do you need maximum throughput, or is portability/power important?
│         ├── Throughput > all → RTX 5090 (1,792 GB/s wins anything dense or MoE)
│         └── Portability/battery > peak speed → M4 Max (~30 W, runs anywhere)
└── No → Does it fit in 128 GB?
        ├── No → Neither/none
        └── Yes → Is it dense or MoE?
                ├── Dense → DGX Spark, but expect <10 tok/s if >20B
                └── MoE → DGX Spark, expect 20–80 tok/s depending on active params
```

### Worked examples

**"I want the highest-quality coding model that fits."**
- → RTX 5090 + Gemma 4 31B-IT Q4_K_M with TurboQuant turbo4 KV. 17/17 score, 53 tok/s, 22 GB VRAM, 58K context.

**"I want the same quality but on a laptop I can actually carry."**
- → M4 Max + Gemma 4 31B-IT Q4_K_M with the **TurboQuant fork's turbo4 KV** (which does build on Metal). Same 17/17 quality, but **11.5 tok/s** instead of 53. Without turbo4 the model literally won't load — 18 GB of weights + 14 GB of f16 KV at 16K context exceeds the 30 GB Metal working set.

**"I want to run the biggest model possible at interactive speed."**
- → DGX Spark + Qwen 3.5 122B-A10B Q4_K_M. 21 tok/s (just barely interactive), full 256K context, 16/17 quality.

**"I want max throughput for real-time use, model quality is secondary."**
- → RTX 5090 + Qwen 3.5 35B-A3B Q4_K_M. 188 tok/s (3x faster than anything else tested), 11/17 quality.

**"I want to load a 70B dense model and play with it."**
- → DGX Spark, but accept ~4 tok/s. Don't expect to use it for interactive coding. M4 Max can't fit it; 5090 can't fit it either.

**"I want the longest context window possible for any model."**
- → DGX Spark wins by default — 120 GB of KV cache headroom dwarfs the 5090's ~10 GB even with TurboQuant compression. M4 Max is the most context-constrained of the three: a 22 GB model leaves only ~8 GB for KV, and the working set ceiling is 30 GB total.

**"I'm in a hotel room and need a coding agent right now."**
- → M4 Max. The other two machines aren't going anywhere. This one runs Gemma 26B-A4B Q6_K at **60 tok/s** (measured) on battery, which is plenty for interactive use. Score: 15/17 single-shot.

---

## Why the Specs Lie

A few traps to avoid when reading marketing material:

1. **"128 GB of memory" doesn't mean "can run any 128 GB model fast."** Capacity ≠ bandwidth. Loading a 70B dense model on the Spark works; running it at interactive speed does not.

2. **"FP4 tensor cores" don't help if you're bandwidth-bound.** Both machines have them. The 5090 can actually feed them. The Spark can't, for any model that does sequential single-user decoding.

3. **"X TFLOPS of compute" is for batched workloads.** Single-user autoregressive decoding uses a tiny fraction of the available compute on any modern GPU — bandwidth is what matters for this workload class.

4. **"Up to 256K context" assumes you have memory for the KV cache.** A dense Gemma 31B's KV cache costs ~870 KB/token at f16, which means 256K context = 222 GB of KV cache. You will not get this on either machine without aggressive KV compression.

5. **Quantization levels affect both capacity AND bandwidth.** Q4 weights are 4x smaller than F16, which means 4x more tokens per second on the same memory bandwidth. This is a much bigger throughput win on the Spark than on the 5090.

6. **"M4 Max" is two different chips.** The 14C CPU / 32C GPU binning has 410 GB/s. The 16C CPU / 40C GPU binning has 546 GB/s — 33% more bandwidth and a different memory ceiling. Apple sells both under the same name. The 36 GB SKU is always the lower-bandwidth variant.

7. **Software ecosystems are not portable.** TurboQuant is x86 + CUDA. NVFP4 is NVIDIA only. MLX is Apple only. A model that runs perfectly on one machine may have no equivalent path on another, even if the math says it should fit. When evaluating a machine, evaluate the inference stack you can actually use on it.

---

## Source Notes

- RTX 5090 spec values: NVIDIA official product page and CUDA programming guide.
- M4 Max spec values: [Apple MacBook Pro 16" Tech Specs](https://support.apple.com/en-us/121554), [9to5Mac M4 Max coverage](https://9to5mac.com/2024/10/30/m4-max-chip-has-16-core-cpu-40-core-gpu-and-35-increase-in-memory-bandwidth/), [EveryMac M4 Max 14C/32C profile](https://everymac.com/systems/apple/macbook_pro/specs/macbook-pro-m4-max-14-core-cpu-32-core-gpu-16-2024-specs.html).
- DGX Spark spec values: [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [LMSYS DGX Spark Review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/), [Backend.ai analysis](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell).
- Throughput measurements: our own benchmarks, see [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md), [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md), and [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md).
