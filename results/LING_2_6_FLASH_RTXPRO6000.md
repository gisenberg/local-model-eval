# Ling-2.6-flash int4 on RTX Pro 6000 — investigation log

**Status:** shelved. Model serves on this hardware (with patches), but no sampling configuration we tried produces usable code on the existing 4-benchmark coding suite. Documenting what we tried so we don't repeat it. Revisit when inclusionAI publishes recommended sampling parameters or when one of the upstream bugs we hit gets fixed.

## Why we tried it

[`inclusionAI/Ling-2.6-flash`](https://huggingface.co/inclusionAI/Ling-2.6-flash) is Ant Group's 104B-total / **7.4B-active** sparse MoE — a sparser activation ratio than Qwen3.6-35B-A3B (which activates ~3B of 35B). int4 release is `inclusionAI/Ling-2.6-flash-int4`. On paper this is interesting on this hardware:

- Hybrid attention: 1:7 ratio of MLA (multi-head latent attention) to **Lightning Linear** layers. 32 layers total → 4 MLA + 28 linear-attention.
- 256 routed experts, 8 active per token, 1 shared expert. 31 of 32 layers are MoE; layer 0 is dense.
- MLA: `q_lora_rank: 1536`, `kv_lora_rank: 512`, `qk_head_dim: 192` (split 128 nope + 64 rope), `v_head_dim: 128`. DeepSeek-style.
- Native context 131K. `rope_scaling: null` (YaRN extension is opt-in via runtime args).
- int4 quant uses `compressed-tensors` `pack-quantized` format with `group_size: 32`. Only routed-expert MLPs are int4; attention, shared experts, lm_head, and the dense layer 0 stay BF16. (This is what made us optimistic — it sidesteps the NVFP4-style "does the kernel exist for this hybrid arch" issue we hit on Qwen3.6-27B.)
- Built-in MTP head (`num_nextn_predict_layers: 1`) — speculative-decoding ready in principle.

Estimated VRAM at 131K context: ~80 GB. Fits on the 96 GB Pro 6000 with margin.

## What we tested

- **vLLM nightly (`0.19.1rc1.dev378+g512765d52`)** with a one-line patch in vendored `lightning_attn.py` — the path we ended up trying first because it matches our existing toolchain.
- **antgroup/sglang fork (`ling_2_6` branch, head `fbeb8e3fe`)** with NEXTN MTP — the path the model card recommends, since "the official SGLang implementation of MTP contains a bug" per the upstream README.

Both paths get the model serving. Neither produces useful coding-bench output without sampling configuration the model card doesn't ship.

## vLLM nightly: required a Triton tile patch on Blackwell sm_120

vLLM nightly already registers `BailingMoeV2_5ForCausalLM` and the int4 weights load cleanly through `CompressedTensorsWNA16MarlinMoEMethod`. First forward pass crashes:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 131584, Hardware limit: 101376. Reducing block sizes or
`num_stages` may help.
```

Source: `vllm/model_executor/layers/lightning_attn.py`'s `_fwd_kv_parallel` Triton kernel. The kernel allocates `[D_FBLOCK, E_FBLOCK]` tiles with `NUM_FBLOCK = 1` hardcoded at line 454. For Ling's `head_dim=128` this works out to 128.5 KB of per-block shared memory. Hopper-class SMs have 228 KB available; Ampere data-center has 164 KB; **Blackwell sm_120 (Pro 6000 / RTX 5090) hard-caps at 99 KB.** Out of resources.

Workaround: change `NUM_FBLOCK = 1` to `NUM_FBLOCK = 2`. Halves the d/e tile size to fit. The kernel arithmetic already supports `NUM_FBLOCK > 1` (asserts on `d % NUM_FBLOCK == 0`). After the patch + clearing torch_compile/triton caches, vLLM serves cleanly.

This patch is in *vendored site-packages* — it gets clobbered by every `pip install --upgrade vllm` and won't help anyone else who hits this. **Should be filed upstream as a sm_120 autotune issue, not held as a private patch.** We've reverted it locally as part of this teardown.

vLLM does **not** load the model's MTP head: `bailing_moe_linear.py:1060` explicitly skips weights starting with `model.mtp.*`, and there's no `bailing_moe_v2_5_mtp.py` sibling module to back a draft worker. Every other vLLM-supported MoE family that has MTP (DeepSeek, Qwen3.5, Qwen3-Next, GLM4, Ernie, ExaONE-MoE, Nemotron-H, OpenPangu, Step3.5) has its own `*_mtp.py`. Bailing doesn't.

**Throughput, no spec dec, T=0 smoke test (binary-search prompt, 96 tokens, 3-run mean):** 108 tok/s warm decode.

## SGLang fork: serves with MTP, but slower and harder

The model card recommends SGLang with `--speculative-algorithm NEXTN`. The official package has a known MTP bug for Ling, so it points at antgroup's [`ling_2_6` branch fork](https://github.com/antgroup/sglang/tree/ling_2_6). Setup on Blackwell sm_120 was substantially heavier than vLLM:

1. **Fresh micromamba env** (the existing `vllm-nightly` env has incompatible dep pins).
2. **System dependencies installed via conda-forge:** Rust 1.94 (for `setuptools-rust`), `protobuf` (the build script for the in-process gRPC subsystem needs `protoc`).
3. **Header path workaround:** flashinfer's JIT-compiled C++ kernels need `cuda_fp16.h` and `cublasLt.h`. The conda-forge `cuda` env packages those at `<env>/targets/x86_64-linux/include/`, not `<env>/include/`. Required setting `CPATH` and `LIBRARY_PATH` to the targets subdirectory before launch.
4. **Hand-picking the attention backend:** the default `triton` backend has a bug for hybrid-attention models — `TritonAttnBackend.__init__` queries `model_runner.token_to_kv_pool.get_value_buffer(0)` to read `v_head_dim` at init time, but layer 0 in Ling is a Lightning Linear layer (full-attention layers are at indices 7, 15, 23, 31). `cutlass_mla` also fails — it inherits from `FlashinferMLABackend` but is missing the `cuda_graph_qo_indptr` attribute. `flashinfer` worked.
5. **Aggressive memory tightening for spec dec:** the draft model needs its own KV pool. We had to drop to `--max-model-len 32768`, `--mem-fraction-static 0.85`, `--max-running-requests 8`, `--max-mamba-cache-size 16` to leave enough headroom. Default settings OOM during draft-model KV pool init.
6. **Auxiliary env vars** that the [`voipmonitor/rtx6kpro` notes](https://github.com/voipmonitor/rtx6kpro/blob/master/inference-engines/sglang.md) flag for sm_120: `SGLANG_DISABLE_DEEP_GEMM=1`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `FLASHINFER_DISABLE_VERSION_CHECK=1`. (DeepGEMM hasn't been ported to sm_120; JIT-DeepGEMM hits the same wall.)

After all that, SGLang serves with NEXTN MTP. Numbers:

| Metric | Value |
|---|---|
| Warm decode (T=0, single-stream) | **57 tok/s** — half of vLLM |
| MTP `accept_rate` | **0.25** |
| MTP `accept_len` | **1.00** |

`accept_rate=0.25` with `accept_len=1.00` means the draft proposes 4 tokens per spec-dec round and only ~1 is accepted — the verifier discards 75% of draft work. This is "MTP paying full draft compute for no throughput win." The model card's recommended `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` is for a 4-GPU TP=4 deployment; single-GPU may need different values, but we didn't tune.

## The bench result that ended this experiment

Both engines hit **0/22** on the existing 4-benchmark coding suite (`tools/nvfp4_qwen36_27b_bench.py`, T=0.3, 3 runs, max_tokens=15000). Different failure modes:

- **SGLang at T=0.3:** outputs ~1200-2200 tokens of structurally complete code (recursive descent parser class + 5 pytest tests), with **logic bugs** that fail at runtime. Example: `parse_expression` delegates to `parse_term` with no `+`/`-` handling, so `"3 + 4"` parses 3 then raises ValueError on `+`. Or `parse_factor` calls `self.parse_power()` which doesn't exist (NameError). The model is producing real code but at the capability edge for this prompt+temperature combination.
- **vLLM at T=0.3 (with explicit `--chat-template chat_template.jinja`):** **catastrophic sampling collapse.** The 15K-token responses are **95,933 chars of `"I understand I understand I understand..."` followed by `"I believe I believe..."`**, until max_tokens. Verified by reading `experiments/ling_2_6_flash_int4_vllm_chattmpl/results.json` directly. This is a degenerate-loop sampler trap.

The vLLM collapse is from underspecified sampling: the bench harness only sets `temperature=0.3` and `max_tokens=15000`. vLLM defaults to `top_p=1.0`, `top_k=-1` (off), `repetition_penalty=1.0` (off). With no top_p/top_k filtering and no penalty, T=0.3 over Ling's distribution shape lets degenerate repeats accumulate. SGLang escapes this because **its sampling defaults differ** — we didn't fully characterize them but `sampling_defaults='model'` plus its built-in fallbacks suppress this failure mode.

At T=0 (the smoke tests we ran on both engines), the model produces clean idiomatic code on simple prompts ("Write a SHA-256 hash function" — 35 tokens, correct, finish_reason=stop). So the model is *capable*; the bench's sampling config is the problem.

## What was missing that would have made this work

1. **Sampling parameters from inclusionAI.** The HF model card has zero guidance on temperature/top_p/top_k/repetition_penalty. `generation_config.json` ships only token IDs (no defaults). The model card focuses on architecture, evals, and tool-use — it implicitly assumes you've configured sampling somewhere else. That somewhere else doesn't exist for this release.

2. **A bench profile that wasn't tuned for Qwen3.6 thinking.** Our existing 4-benchmark suite's `T=0.3, max_tokens=15000` was chosen for Qwen3.6-35B-A3B's thinking-mode trajectories. Ling defaults to `thinking_option = 'off'` (per the chat template). The bench wasn't a fair test even before the sampling-collapse issue.

3. **Upstream fixes for at least three of the things we patched around:**
   - vLLM `lightning_attn.py` Triton tile sizes that fit Blackwell sm_120's 99 KB shared memory budget (one-line workaround applied locally, then reverted).
   - vLLM bailing loader picking up `model.mtp.*` weights and exposing them via a `bailing_moe_v2_5_mtp.py` sibling module (mirroring the structure of `qwen3_next_mtp.py` etc.).
   - SGLang's `TritonAttnBackend.__init__` not assuming layer 0 is full-attention. `CutlassMLABackend` inheriting from `FlashinferMLABackend` without overriding the `cuda_graph_qo_indptr`-using methods.

## Verdict

Shelved. Disk reclaimed (~75 GB, weights + sglang fork + sglang-ling env). vLLM patch reverted. We can revisit when:

- inclusionAI publishes a `recommended_sampling.md` or updates `generation_config.json` with `top_p`/`repetition_penalty` defaults, or
- vLLM gains a `bailing_moe_v2_5_mtp.py` so we can compare meaningfully against the SGLang+MTP path, or
- We need a 7.4B-active dense-quality MoE specifically and have time to characterize the sampling space ourselves.

Until then, **Qwen3.6-27B FP8 + DFlash remains the dense-quality SWE-bench leader** on this hardware (57.3% SWE-bench Lite, 195 tok/s warm with spec dec), and **gpt-oss-120b Q8** is the throughput king (264 tok/s).

## Raw artifacts

- Bench output that triggered the shelving: `experiments/ling_2_6_flash_int4_vllm/results.json`, `experiments/ling_2_6_flash_int4_vllm_chattmpl/results.json`, `experiments/ling_2_6_flash_int4_sglang_mtp/results.json`. Kept for future re-analysis if anyone retries this model.
- The "I understand" loop is in the chat-template-fixed vLLM run's `lru_cache` first run.

## See also

- [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md) — the production-preset tier list for this card; Ling is not in it, by design.
- [QWEN36_RTXPRO6000.md](QWEN36_RTXPRO6000.md) — the Qwen3.6 deep dive whose 4-benchmark suite we reused (and which Ling failed against for sampling reasons).
