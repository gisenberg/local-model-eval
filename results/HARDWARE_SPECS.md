# Hardware Spec Sheet — Benchmarked Machines

This document captures the hardware we run benchmarks on, with a focus on the specs that actually matter for local LLM inference. Numbers come from official NVIDIA documentation, our own measurements, and the bandwidth/throughput data we've gathered through benchmarking.

## TL;DR

| | RTX 5090 (Workstation) | DGX Spark (GB10) |
|---|---|---|
| **Total memory** | 32 GB GDDR7 | 128 GB LPDDR5x (unified) |
| **Memory bandwidth** | **1,792 GB/s** | **273 GB/s** |
| **Max model (dense, Q4)** | ~31 GB at 53 tok/s | ~70 GB but at <8 tok/s |
| **Max model (MoE A10B, Q4)** | ~22 GB at 188 tok/s* | ~70 GB at 21 tok/s |
| **Best for** | Dense models, max throughput, code generation | Huge MoE models, full context windows |
| **Bottleneck** | VRAM capacity | Memory bandwidth |

*Qwen 35B-A3B on llama-server. Quality varies by inference engine.

**The headline insight:** these two machines look similar on paper (both Blackwell, both support FP4) but have **opposite tradeoffs**. The 5090 has compute and bandwidth in abundance but is starved for capacity. The Spark has capacity in abundance but is starved for bandwidth. Almost no model class is "good on both" — picking the right machine for a model is more important than picking the right model.

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

| Metric | RTX 5090 | DGX Spark | Ratio |
|---|---|---|---|
| Capacity | 32 GB | 128 GB | Spark 4x |
| Bandwidth | 1,792 GB/s | 273 GB/s | **5090 6.6x** |
| Bytes/token at 50 tok/s | 35.8 GB | 5.5 GB | — |
| Largest dense Q4 model viable | ~31B | ~7B for interactive use | — |
| Largest MoE A10B model viable | ~35B total | ~122B total | — |

### Measured throughput on the same model class

Where we have comparable runs:

| Model | RTX 5090 | DGX Spark | Notes |
|---|---|---|---|
| Gemma 4 31B-IT Q4_K_M (dense) | **53 tok/s** | ~6.7 tok/s | Bandwidth-bound on Spark |
| Qwen 3.5 35B-A3B Q4_K_M (MoE, 3B active) | **188 tok/s** | not benchmarked | Compute path on 5090 |
| Qwen 3.5 122B-A10B Q4 (MoE, 10B active) | **does not fit** | 21 tok/s | Spark wins by capacity |
| Qwen3-Coder-Next 80B-A3B (MoE, 3B active) | does not fit | 50 tok/s | Spark wins by capacity |

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
Does the model fit in 32 GB at usable quant?
├── Yes → RTX 5090 (better bandwidth = faster decoding)
└── No → Does it fit in 128 GB?
        ├── No → Neither machine
        └── Yes → Is it dense or MoE?
                ├── Dense → DGX Spark, but expect <10 tok/s if >20B
                └── MoE → DGX Spark, expect 20–80 tok/s depending on active params
```

### Worked examples

**"I want the highest-quality coding model that fits."**
- → RTX 5090 + Gemma 4 31B-IT Q4_K_M with TurboQuant turbo4 KV. 17/17 score, 53 tok/s, 22 GB VRAM, 58K context.

**"I want to run the biggest model possible at interactive speed."**
- → DGX Spark + Qwen 3.5 122B-A10B Q4_K_M. 21 tok/s (just barely interactive), full 256K context, 16/17 quality.

**"I want max throughput for real-time use, model quality is secondary."**
- → RTX 5090 + Qwen 3.5 35B-A3B Q4_K_M. 188 tok/s (3x faster than anything else tested), 11/17 quality.

**"I want to load a 70B dense model and play with it."**
- → DGX Spark, but accept ~4 tok/s. Don't expect to use it for interactive coding.

**"I want the longest context window possible for any model."**
- → DGX Spark wins by default — 120 GB of KV cache headroom dwarfs the 5090's ~10 GB even with TurboQuant compression.

---

## Why the Specs Lie

A few traps to avoid when reading marketing material:

1. **"128 GB of memory" doesn't mean "can run any 128 GB model fast."** Capacity ≠ bandwidth. Loading a 70B dense model on the Spark works; running it at interactive speed does not.

2. **"FP4 tensor cores" don't help if you're bandwidth-bound.** Both machines have them. The 5090 can actually feed them. The Spark can't, for any model that does sequential single-user decoding.

3. **"X TFLOPS of compute" is for batched workloads.** Single-user autoregressive decoding uses a tiny fraction of the available compute on any modern GPU — bandwidth is what matters for this workload class.

4. **"Up to 256K context" assumes you have memory for the KV cache.** A dense Gemma 31B's KV cache costs ~870 KB/token at f16, which means 256K context = 222 GB of KV cache. You will not get this on either machine without aggressive KV compression.

5. **Quantization levels affect both capacity AND bandwidth.** Q4 weights are 4x smaller than F16, which means 4x more tokens per second on the same memory bandwidth. This is a much bigger throughput win on the Spark than on the 5090.

---

## Source Notes

- RTX 5090 spec values: NVIDIA official product page and CUDA programming guide.
- DGX Spark spec values: [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [LMSYS DGX Spark Review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/), [Backend.ai analysis](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell).
- Throughput measurements: our own benchmarks, see [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) and [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md).
