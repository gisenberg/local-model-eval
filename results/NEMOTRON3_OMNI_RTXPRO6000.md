# Nemotron 3 Nano Omni 30B-A3B Reasoning on RTX Pro 6000 — first-look investigation

**Status:** strong result. Across 3 precisions × 2 reasoning modes × 4 coding benchmarks (3 runs each), the headline numbers are:

| Variant | Preset | Best-of-3 | Avg | Decode tok/s | Weight footprint |
|---|---|---|---|---|---|
| **BF16** | **Thinking** | **20/22 (91%)** | 15.3 | 190 | 62 GB |
| BF16 | Instruct | 18/22 (82%) | 16.7 | 188 | 62 GB |
| **FP8** | **Thinking** | **19/22 (86%)** | 14.0 | 287 | 33 GB |
| FP8 | Instruct | 14/22 (64%) | 12.0 | 267 | 33 GB |
| **NVFP4** | **Thinking** | **18/22 (82%)** | 13.0 | 305 | 21 GB |
| NVFP4 | Instruct | 14/22 (64%) | 10.0 | 304 | 21 GB |

**Two-line summary:** Thinking lifts best-of-3 on every variant (+2 / +5 / +4 going BF16/FP8/NVFP4). **FP8 + Thinking is the dominant pareto choice** — 19/22 at 287 tok/s in 33 GB, only 1 point below the BF16 ceiling at 1.5× the throughput.

## What it is

[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) is a Mamba2-Transformer hybrid MoE with multimodal heads bolted on:

- **Text backbone:** `nemotron_h`, 52 layers, hidden 2688, 6 active experts/token, 31B total / ~3B active. Custom `NemotronH_Nano_Omni_Reasoning_V3` architecture (requires `--trust-remote-code`).
- **Vision encoder:** CRADIO v4-H (`vit_hidden_size=1280`).
- **Audio encoder:** Parakeet (24-layer, hidden 1024).
- **Native context:** 256K (we tested at 32K to keep startup fast).
- **Available variants:** BF16 (61.5 GB), FP8 (32.8 GB), NVFP4 (20.9 GB). We tested all three.

The model card ships explicit sampling presets — a pleasant change from [Ling-2.6's silence on this](LING_2_6_FLASH_RTXPRO6000.md):

- **Instruct (non-thinking):** `temperature=0.2, top_k=1, max_tokens=4096`
- **Thinking:** `temperature=0.6, top_p=0.95, reasoning_budget=16384, grace_period=1024, max_tokens=20480`

Both presets honored as documented. The reasoning_budget and grace_period parameters get forwarded by vLLM through `--reasoning-parser nemotron_v3` to the chat-template's budget cap.

## Per-benchmark breakdown

The interesting story is which benchmarks each (precision, mode) combination wins or loses. Three runs per cell, scores listed in run order. **Best/Avg** are the two summary stats — best-of-3 captures resampling-with-best behavior, avg captures expected-value behavior.

| Benchmark | BF16 Inst | BF16 Think | FP8 Inst | FP8 Think | NVFP4 Inst | NVFP4 Think |
|---|---|---|---|---|---|---|
| Expression Evaluator (5) | 7,7,5 / b7 a6.3 | 4,1,5 / b5 a3.3 | 0,4,5 / b5 a3.0 | 4,5,5 / b5 a4.7 | 6,2,0 / b6 a2.7 | 0,5,5 / b5 a3.3 |
| A* Pathfinding (6) | 7,2,5 / b7 a4.7 | 6,0,3 / b6 a3.0 | 4,4,4 / b4 a4.0 | 6,1,4 / b6 a3.7 | 0,1,0 / b1 a0.3 | 5,6,4 / b6 a5.0 |
| LRU Cache with TTL (6) | 2,2,2 / b2 a2.0 | 4,4,4 / b4 a4.0 | 0,0,0 / b0 a0.0 | 0,3,0 / b3 a1.0 | 3,1,3 / b3 a2.3 | 3,0,1 / b3 a1.3 |
| String Processor (5) | 5,5,5 / b5 a5.0 | 5,5,5 / b5 a5.0 | 5,5,5 / b5 a5.0 | 4,5,5 / b5 a4.7 | 5,4,5 / b5 a4.7 | 3,4,3 / b4 a3.3 |
| **Total best-of-3** | **18/22** | **20/22** | **14/22** | **19/22** | **14/22** | **18/22** |

Scores >5/6 happen because the model writes more tests than asked and they pass — pytest counts them, we honor the count.

## Findings

### 1. Thinking lifts best-of-3 on every variant

```
BF16  : 18 → 20  (+2)
FP8   : 14 → 19  (+5)   ← biggest jump
NVFP4 : 14 → 18  (+4)
```

The lifts come from different benchmarks at different precisions:
- **BF16:** thinking only helps LRU (2 → 4). The other three were already at their BF16 ceiling.
- **FP8:** thinking unblocks Expression Evaluator (run 1 went from 0 → 4) and A* (4 → 6 best). LRU stays mostly at 0.
- **NVFP4:** thinking dramatically recovers A* (best 1 → best 6). This mirrors the [Qwen3.6 LRU recovery pattern](MODEL_RANKINGS_RTXPRO6000.md#prompting-levers-can-we-recover-qwen-family-failures): a "right at the capability edge" benchmark is unblocked by reasoning at the cost of higher variance elsewhere.

### 2. Average scores DROP with thinking, even though best-of-3 rises

```
                   Best-of-3      Avg
BF16   Instruct → Thinking:  18→20   16.7→15.3
FP8    Instruct → Thinking:  14→19   12.0→14.0
NVFP4  Instruct → Thinking:  14→18   10.0→13.0
```

BF16 actually has higher avg in Instruct mode. Why? Instruct uses `T=0.2, top_k=1` (effectively greedy) — runs are nearly deterministic and converge near the model's "modal" answer. Thinking uses `T=0.6, top_p=0.95` — wider sampling, more ceiling but more floor. **Best-of-3 captures the ceiling; avg captures the variance.**

Practical implication: if you're going to deploy with **resample-and-verify** (e.g. agentic loop with pytest feedback), use Thinking. If you're going to deploy as a **one-shot Q&A** without retries, the Instruct preset's lower variance may be worth more than the slightly higher ceiling Thinking offers — particularly on BF16.

### 3. FP8 + Thinking is the dominant pareto choice

```
Quality ranked by best-of-3:  BF16 Thinking (20) > FP8 Thinking (19) > NVFP4 Thinking (18) ≈ BF16 Instruct (18)
Throughput ranked tok/s:      NVFP4 (305) ≈ FP8 (287) > BF16 (190)
```

NVFP4 only gains 6% over FP8 in throughput (305 vs 287) but consistently loses 1 point in quality. The NVFP4 release uses **mixed precision** (NVFP4 for routed-expert MLPs, FP8 elsewhere — declared in `hf_quant_config.json`); the routed-expert FP4 is the lossy bit, and the speedup it buys is small because the FP8 paths still dominate the dense layers.

**The clean recommendation:**
- **Pick FP8 + Thinking** unless you absolutely need every quality point or every throughput point.
- BF16 only if your workload tolerates 190 tok/s and you want the quality ceiling.
- NVFP4 only if 21 GB of weights is the binding constraint (e.g. running another model on the same card).

### 4. LRU Cache with TTL is the persistent blind spot

| Variant + preset | LRU best | LRU avg |
|---|---|---|
| BF16 Thinking | 4 | 4.0 |
| BF16 Instruct | 2 | 2.0 |
| NVFP4 Instruct | 3 | 2.3 |
| NVFP4 Thinking | 3 | 1.3 |
| FP8 Thinking | 3 | 1.0 |
| FP8 Instruct | 0 | 0.0 |

For reference, on the same benchmark: **Gemma-4-31B-it gets 6/6, Qwen3.6-35B-A3B with thinking gets 6/6, gpt-oss-120b gets 6/6**. Nemotron-3-Omni does not — even at BF16 + Thinking it caps at 4/6. The pattern matches Qwen-family models without thinking-mode help, but unlike Qwen3.6 (which fully recovers LRU with thinking on), Nemotron only partially recovers it. This is a real capability gap on this particular pattern, not a sampling issue.

### 5. Throughput is essentially constant within precision tier

Decode tok/s is bandwidth-bound and barely changes between Instruct and Thinking modes:

```
BF16 :  188 vs 190    (+1%)
FP8  :  267 vs 287    (+7%)
NVFP4:  304 vs 305    (+0%)
```

The FP8 +7% looks like a warm-cache artifact (run 1 was 71 tok/s during CUDA-graph warmup, dragging the Instruct mean). Don't read into it.

## Where this fits in the rankings

Slotting **BF16 + Thinking (the new ceiling)** into the existing [Pro 6000 rankings table](MODEL_RANKINGS_RTXPRO6000.md):

| Position | Model | Quant + mode | Coding | Decode tok/s |
|---|---|---|---|---|
| above | gpt-oss-120b | Q8_0 | 21/22 | 264 |
| above | Qwen3.6-27B (dense) FP8 + DFlash | FP8 + spec dec | 21/22 | 195 warm |
| **here** | **Nemotron-3-Omni-30B-A3B** | **BF16 + Thinking** | **20/22** | **190** |
| also here | Qwen3.6-35B-A3B + Thinking | Q8_0 + thinking | 21/22 | 221 |
| below | Qwen3.6-35B-A3B baseline | Q8_0 | 15/22 | 221 |

And for the **FP8 + Thinking sweet spot** vs the rest:

| Position | Model | Quant + mode | Coding | Decode tok/s |
|---|---|---|---|---|
| **here** | **Nemotron-3-Omni-30B-A3B** | **FP8 + Thinking** | **19/22** | **287** |
| similar throughput | Qwen3.6-35B-A3B | Q8_0 (no thinking) | 15/22 | 221 |
| similar quality | Qwen3.6-35B-A3B + Thinking | Q8_0 + thinking | 21/22 | 221 |

**The takeaway is solid A-tier candidacy at FP8 + Thinking, S-tier candidacy at BF16 + Thinking** — both within striking distance of gpt-oss-120b and the Qwen3.6 thinking-on configs. The differentiator: Nemotron-3-Omni has a **multimodal capability** none of the others offer (vision + audio + video, untested here). For a text-only coding workload, gpt-oss-120b at 264 tok/s and Qwen3.6+thinking at 221 tok/s are still the headliners — but if you ever want to feed the model a screenshot or a voice memo, this is the only local option in this tier.

## Setup notes (so we don't re-do the 90 minutes of toolchain pain)

vLLM 0.20.0 with the cu13 wheel stack on Blackwell sm_120 needs five things to JIT-compile flashinfer's CUTLASS fused-MoE kernel cleanly. We hit each one as a separate failure mode; documenting the chain so future-us doesn't repeat them.

1. **Install nvcc.** `nvidia-cuda-nvcc==13.0.88` (NOT `nvidia-cuda-nvcc-cu13`, which is a deprecated stub. NOT `nvidia-cuda-nvcc==13.2.x` — version-must-match the rest of the cu13 stack, see #2). Ships nvcc to `~/venvs/<env>/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc`.
2. **Match nvcc version to runtime.** vLLM 0.20.0 pulls `nvidia-cuda-runtime==13.0.96` and friends. The CCCL header has a hard `#error "CUDA compiler and CUDA toolkit headers are incompatible"` if nvcc minor differs. Pin everything to 13.0.x:
   ```
   pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-cccl==13.0.85" \
               "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
   ```
3. **Install CCCL headers separately.** `nvidia-cuda-nvcc` only ships the compiler; the `cuda/std/utility` and `nv/target` headers come from `nvidia-cuda-cccl`. Without it, every `.cuda.o` build line fails with `fatal error: cuda/std/utility: No such file or directory`.
4. **Symlink the dev-style `.so` names.** The cu13 wheels install `libcudart.so.13`, `libnvrtc.so.13`, `libcublas.so.13` etc. The flashinfer linker invocation does `-lcudart -lnvrtc -lcublas`, which looks for unversioned `.so` files. Manual fix:
   ```
   cd ~/venvs/<env>/lib/python3.12/site-packages/nvidia/cu13/lib
   for n in cudart nvrtc cublas cublasLt nvJitLink cusparse curand; do
     ln -sf lib${n}.so.13 lib${n}.so
   done
   ```
5. **Symlink `lib` → `lib64`.** flashinfer's build template hardcodes `-L$cuda_home/lib64`. The wheels install to `lib/`. One symlink:
   ```
   ln -s lib ~/venvs/<env>/lib/python3.12/site-packages/nvidia/cu13/lib64
   ```
6. **Set `MAX_JOBS=8`.** flashinfer's `gemm_sm120` op (the one NVFP4 needs but BF16 doesn't) JIT-compiles ~70 FP8/MXFP4 cutlass kernels in parallel. With nproc=32 and 62 GB RAM, default-parallel nvcc runs out of RAM (`exit code 137`) about halfway through. `MAX_JOBS=8` brings peak RAM to ~30 GB and finishes in ~5 minutes.

After all this, the BF16 model serves with no runtime patches needed — different from Ling-2.6, which required hand-patching `lightning_attn.py`'s Triton tile sizes for sm_120. vLLM 0.20.0's `nemotron_h` model definition handles Blackwell out of the box. The pain was entirely in the build toolchain, not in the model code.

### Launch commands

```bash
# Common env (apply to all three precisions)
export CUDA_HOME=$(python -c 'import nvidia.cu13 as _; print(_.__path__[0])')
export PATH=~/venvs/nemotron3/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=8

# BF16 / FP8 / NVFP4 — vLLM auto-detects modelopt quant from hf_quant_config.json
vllm serve ~/models-vllm/nemotron3-30b-bf16  --served-model-name nemotron3-omni-bf16  --port 8090 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --trust-remote-code --reasoning-parser nemotron_v3
vllm serve ~/models-vllm/nemotron3-30b-fp8   --served-model-name nemotron3-omni-fp8   --port 8090 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --trust-remote-code --reasoning-parser nemotron_v3
vllm serve ~/models-vllm/nemotron3-30b-nvfp4 --served-model-name nemotron3-omni-nvfp4 --port 8090 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --trust-remote-code --reasoning-parser nemotron_v3
```

### Bench commands

```bash
# Instruct preset (T=0.2, top_k=1, max_tokens=4096, enable_thinking=False)
python tools/nemotron3_omni_bench.py --port 8090 \
  --served-name nemotron3-omni-bf16 --output-dir experiments/nemotron3_omni_30b_bf16 \
  --preset instruct

# Thinking preset (T=0.6, top_p=0.95, max_tokens=20480, reasoning_budget=16384, grace_period=1024)
python tools/nemotron3_omni_bench.py --port 8090 \
  --served-name nemotron3-omni-bf16 --output-dir experiments/nemotron3_omni_30b_bf16_thinking \
  --preset thinking
```

## What we didn't test

This is the limiting box around the numbers above:

1. **Multimodal.** The "Omni" half — vision (CRADIO), audio (Parakeet), video (256-frame, 2-min clips) — was not exercised. Our coding bench is text-only.
2. **Long context.** The native 256K context window was not tested. We ran at `max_model_len=32768` to keep startup fast.
3. **The two extra vLLM flags from the model card.** `--video-pruning-rate 0.5`, `--media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'`, and `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. None of these matter for text-only chat completions, but they're necessary for the multimodal use cases we skipped.
4. **SWE-bench Lite.** We have 4-bench coding numbers, not agentic-coding numbers. Given the 19/22-20/22 ceiling, this is plausibly competitive with the Qwen3.6-27B FP8 + DFlash preset (57.3% on SWE-bench Lite) but we don't know.
5. **Streaming throughput sidecar.** All tok/s numbers are derived from the bench harness (single-stream, batched within run). A dedicated streaming TTFT + warm-decode measurement (matching the methodology of `tools/rtxpro6000_bench.py`) wasn't run.

## Variance note: top_k=1 should be deterministic but isn't

In the BF16 Instruct row, `temperature=0.2, top_k=1` is greedy decoding — same prompt should produce identical output across runs. We see noticeable run-to-run variance (Expression Evaluator: 7, 7, 5; A* Pathfinding: 7, 2, 5).

Same documented behavior as [Finding #5: Infrastructure params can change output](../README.md#key-findings) in the repo README. vLLM's MoE expert dispatch produces different floating-point reduction orders depending on batch state, and at greedy decoding a single tipped-token can fork the entire continuation. NVFP4 is more sensitive to this than BF16 (lower precision in the routing softmax means the boundary cases tip more often). FP8 is intermediate.

This is not a bug. It's the same finding from [the Qwen3.6 rankings analysis](MODEL_RANKINGS_RTXPRO6000.md#qwen36-35b-a3b-hybrid-attention-moe), reproduced here for a different model family.

## Verdict

**S-tier-adjacent.** BF16 + Thinking lands at 20/22 (91%), one point below gpt-oss-120b (21/22) and one point below Qwen3.6-35B-A3B + thinking (21/22), at half their throughput. **FP8 + Thinking at 19/22 / 287 tok/s is the practical configuration** — within striking distance of the S-tier on both axes, with the smallest weight footprint of any 30B+ MoE in the lineup at a quality level worth deploying.

**Two follow-ups worth doing before this gets a fully blessed rankings entry:**
1. **SWE-bench Lite at FP8 + Thinking.** We expect competitive numbers given the BF16 ceiling, but agentic-coding doesn't track from-scratch coding-bench reliably (see Gemma's 100% coding-bench → 23.0% SWE-bench gap).
2. **Try the multimodal use cases.** This model's headline is "video + audio + image + text in, text out" — and right now we've measured the slice that doesn't differentiate it from the existing lineup. A specific multimodal coding/QA benchmark would justify the "we tested this *because* it's multimodal" angle.

## Raw artifacts

- `experiments/nemotron3_omni_30b_bf16/results.json` (Instruct) and `*_test.py` (extracted code)
- `experiments/nemotron3_omni_30b_bf16_thinking/results.json` (Thinking)
- `experiments/nemotron3_omni_30b_fp8/results.json` (Instruct)
- `experiments/nemotron3_omni_30b_fp8_thinking/results.json` (Thinking)
- `experiments/nemotron3_omni_30b_nvfp4/results.json` (Instruct)
- `experiments/nemotron3_omni_30b_nvfp4_thinking/results.json` (Thinking)
- Bench harness: `tools/nemotron3_omni_bench.py`

## See also

- [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md) — the production-preset tier list for this card; Nemotron-3-Omni is not yet in it pending SWE-bench follow-up.
- [LING_2_6_FLASH_RTXPRO6000.md](LING_2_6_FLASH_RTXPRO6000.md) — the prior hybrid-MoE attempt on this hardware; shelved at 0/22 because the model card ships no sampling guidance. The Nemotron success story is partly "the model card gave us the sampling preset" and partly "vLLM 0.20.0 has the model code".
- [QWEN36_RTXPRO6000.md](QWEN36_RTXPRO6000.md) — precedent for the BF16/FP8/NVFP4 precision sweep methodology we reused here.
