# Results

This folder is the analysis and reference output for the local-model-eval project. The benchmarks themselves live in `experiments/`; the per-experiment scripts live in `tools/`. This folder is where the *findings* land.

## Where to start (by question)

| If you're asking... | Read this |
|---|---|
| "What hardware should I buy for local LLM inference?" | **[HARDWARE_SHORTLIST.md](HARDWARE_SHORTLIST.md)** — buyer's guide covering M4/M5 Max, M3 Ultra Studio, RTX 5090, RTX Pro 6000 Blackwell, and DGX Spark. Mix of our own measurements and cited third-party benchmarks. |
| "What are the technical specs of the machines you actually benchmarked?" | **[HARDWARE_SPECS.md](HARDWARE_SPECS.md)** — deep spec sheet for the 3 machines that came through our bench (5090, M4 Max 36 GB MBP, DGX Spark). |
| "Which model should I run on my [5090 / M4 Max / DGX Spark]?" | The matching `MODEL_RANKINGS_<platform>.md` file. Each is a self-contained tier list with quality scores, throughput, and quirks. |
| "How big a context window can I fit on [5090 / M4 Max]?" | The matching `CONTEXT_CAPACITY_<platform>.md` file. |
| "Should I use TurboQuant KV cache compression on [5090 / M4 Max]?" | The matching `TURBOQUANT_IMPACT_<platform>.md` file. The 5090 answer is "yes always"; the M4 Max answer is "only when forced." |
| "Should I use RotorQuant (the newer KV quantizer) on my hardware?" | **[ROTORQUANT_TLDR.md](ROTORQUANT_TLDR.md)** — cross-platform digest. Short answer: yes on M4 Max, yes on Spark for hybrid-attention models, mostly no on 5090. |

## Per-platform docs

The repo benchmarks three machines directly. Every platform-specific analysis follows the naming convention `<TYPE>_<PLATFORM>.md`.

### RTX 5090 (Workstation, 32 GB GDDR7, Windows + WSL2)
- [MODEL_RANKINGS_5090.md](MODEL_RANKINGS_5090.md) — tier list with quality scores, throughput, TTFT (corrected after a Windows `requests`/`urllib3` bug)
- [CONTEXT_CAPACITY_5090.md](CONTEXT_CAPACITY_5090.md) — KV cache spill cliff, full 256K context tests
- [TURBOQUANT_IMPACT_5090.md](TURBOQUANT_IMPACT_5090.md) — what TurboQuant unlocks: more context, sometimes faster
- [TURBO3_RESULTS_5090.md](TURBO3_RESULTS_5090.md) — full TurboQuant turbo3-vs-turbo4 sweep with thinking on/off
- [ROTORQUANT_5090.md](ROTORQUANT_5090.md) — rotorquant results on 3 Qwen-family 27B models: turbo4 wins on 2 of 3, Harmonic ties

### MacBook Pro M4 Max 36 GB (Apple Silicon, macOS, Metal)
- [MODEL_RANKINGS_M4MAX.md](MODEL_RANKINGS_M4MAX.md) — tier list, includes MLX vs llama.cpp comparison
- [CONTEXT_CAPACITY_M4MAX.md](CONTEXT_CAPACITY_M4MAX.md) — Metal working set ceiling, the `n_ubatch` finding, OOM-not-spill failure mode
- [TURBOQUANT_IMPACT_M4MAX.md](TURBOQUANT_IMPACT_M4MAX.md) — why TurboQuant turbo4 is *slower* than f16 on Apple Silicon
- [ROTORQUANT_HYPOTHESIS_M4MAX.md](ROTORQUANT_HYPOTHESIS_M4MAX.md) — pre-experiment prediction for rotorquant on Metal
- [ROTORQUANT_M4MAX.md](ROTORQUANT_M4MAX.md) — actual rotorquant results: K-only +19% vs f16, Gemma 4 models blocked by fork base

### NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified, Linux/aarch64)
- [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md) — bandwidth-bound model lineup, MoE-favorable
- [ROTORQUANT_SPARK.md](ROTORQUANT_SPARK.md) — rotorquant results (net-negative on Spark MoE, as predicted)

## Cross-platform docs

These docs span hardware rather than focusing on one machine:

- [HARDWARE_SPECS.md](HARDWARE_SPECS.md) — measured spec sheet + side-by-side throughput on the same models, for the 3 machines we benchmarked directly
- [HARDWARE_SHORTLIST.md](HARDWARE_SHORTLIST.md) — broader buyer's guide, including machines we cite third-party benchmarks for (M5 Max, M3 Ultra Studio, RTX Pro 6000 Blackwell)
- [ROTORQUANT_TLDR.md](ROTORQUANT_TLDR.md) — one-screen digest of the three per-platform rotorquant experiments (M4 Max: +19% on Metal, Spark: Qwen win + GLM broken, 5090: turbo4 mostly wins)

## Raw data

- [`raw/`](raw/) — raw benchmark JSONs and intermediate outputs from the LM Studio era. Per-experiment outputs from later runs live in [`../experiments/`](../experiments/).

## Naming convention

Files in this folder follow `<TYPE>_<PLATFORM>.md` where:

- `<TYPE>` is one of: `MODEL_RANKINGS`, `CONTEXT_CAPACITY`, `TURBOQUANT_IMPACT`, `TURBO3_RESULTS`, `ROTORQUANT`, `ROTORQUANT_HYPOTHESIS`
- `<PLATFORM>` is one of: `5090`, `M4MAX`, `SPARK`

Note: the `ROTORQUANT_<PLATFORM>.md` files hold post-experiment results (the folder name provides the "results" context), while `ROTORQUANT_HYPOTHESIS_<PLATFORM>.md` files hold the pre-experiment predictions. The pre-experiment docs for Spark and 5090 are currently still at the repo root; the M4 Max pair live in this folder for the cleaner naming.

Cross-platform docs (`HARDWARE_SPECS.md`, `HARDWARE_SHORTLIST.md`, this `README.md`) don't have a platform suffix.

Future per-platform additions (if we benchmark more hardware) should follow the same pattern: `MODEL_RANKINGS_M5MAX.md`, `CONTEXT_CAPACITY_PRO6000.md`, etc.

## Reading order if you're new to the repo

If you've never looked at this repo before and want to get up to speed:

1. **Start at the [top-level README](../README.md)** for the overall project framing and methodology.
2. **Read [HARDWARE_SHORTLIST.md](HARDWARE_SHORTLIST.md)** to see which hardware classes are on the table and why.
3. **Pick the platform that matches what you have** and read its `MODEL_RANKINGS_<platform>.md`.
4. **If you're hitting OOMs or want long contexts**, read the matching `CONTEXT_CAPACITY_<platform>.md`.
5. **If your model fits on multiple machines and you want to understand the speed/quality tradeoffs**, read [HARDWARE_SPECS.md](HARDWARE_SPECS.md).

The TurboQuant docs are deep dives, only worth reading if you're specifically interested in the KV cache compression story (5090: very useful; M4 Max: mostly not useful).
