# DeepSeek V4 Flash GGUF on RTX Pro 6000

Tested 2026-06-30 on the RTX PRO 6000 Blackwell workstation.

## Candidate

The useful target for this host is the antirez q2 GGUF:

- Repo: <https://huggingface.co/antirez/deepseek-v4-gguf>
- File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
- Size: `86,720,111,200` bytes = 80.8 GiB
- Quant recipe: routed expert gate/up `IQ2_XXS`, routed expert down `Q2_K`, attention/shared/output mostly `Q8_0`, routers/indexer/compressor/HC kept higher precision.

The sokann near-lossless GGUF is useful as a runbook, but not as the target on this machine:

- Repo: <https://huggingface.co/sokann/DeepSeek-V4-Flash-GGUF>
- File: `DeepSeek-V4-Flash.gguf`
- Size: `156,378,344,544` bytes = about 146 GiB
- Card guidance: fits on 160 GiB system RAM plus 48 GiB VRAM. This host has 62 GiB RAM plus 63 GiB swap, so the 146 GiB file is the wrong fit.

## llama.cpp Build

DeepSeek V4 support landed in llama.cpp PR #24162:
<https://github.com/ggml-org/llama.cpp/pull/24162>

The existing CUDA build on this host was too old:

- Existing: `/home/gisenberg/llama-build/src/build/bin/llama-server`
- Version: `207 (bbeb89d)`, before DeepSeek V4 support

I built a separate worktree so existing benchmarks keep their old binary:

- Source: `/home/gisenberg/llama-build/src-deepseek-v4`
- Binary: `/home/gisenberg/llama-build/src-deepseek-v4/build/bin/llama-server`
- Version: `1021 (0eca4d490)`, tag `b9851`
- CUDA: `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`

## Run Config

The benchmark entry is `deepseek-v4-flash-iq2xxs` in `tools/rtxpro6000_bench.py`.

Download:

```bash
hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir /home/gisenberg/models/deepseek-v4-flash-iq2xxs \
  --max-workers 1
```

Throughput run:

```bash
LLAMA_BACKEND=cuda \
LLAMA_DIR=/home/gisenberg/llama-build/src-deepseek-v4/build/bin \
LLAMA_PORT=8091 \
python3 tools/rtxpro6000_bench.py deepseek-v4-flash-iq2xxs 32768
```

Effective server flags:

```bash
llama-server \
  -m /home/gisenberg/models/deepseek-v4-flash-iq2xxs/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --port 8091 --host 127.0.0.1 \
  -c 32768 -ngl auto -fa on -np 1 --no-mmap \
  --jinja -cram 0 --fit on -b 2048 -ub 2048
```

## Results

Throughput at 32K context:

| Metric | Value |
|---|---:|
| Load time | 12.0 s |
| VRAM after load | 92,168 MiB |
| VRAM headroom | ~5,719 MiB |
| Mean TTFT | 184 ms |
| Mean decode | 38.73 tok/s |
| Timed runs | 5 x 255 completion tokens |

Raw output:

- `experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-iq2xxs.json`

Coding suite:

| Benchmark | Raw | Capped | Notes |
|---|---:|---:|---|
| String Processor | 4/5 | 4/5 | One generated test has a wrong expected value: `"Python is fun"` has 3 vowels, not 4. |
| Expression Evaluator | 5/5 | 5/5 | Clean. |
| A* Pathfinding | 8/6 | 6/6 | Model generated two extra passing tests; cap to expected score for rankings. |
| LRU Cache with TTL | 0/6 | 0/6 | Implementation-only response; pytest collected zero tests. |
| Total | 17/22 raw | 15/22 capped | Use capped score for comparison. |

Raw output:

- `experiments/rtxpro6000_coding/deepseek-v4-flash-iq2xxs.json`
- `experiments/rtxpro6000_coding/deepseek-v4-flash-iq2xxs/*.md`

## SWE-bench Lite Partial Run

I aborted the full SWE-bench Lite run on 2026-07-01 after collecting enough signal. This is not a full 300-instance score.

Harness:

- SWE-agent `v1.1.0`, function-calling tools, 75 calls per instance.
- `num_workers=1`; the 80 GiB GGUF is too VRAM-tight for parallel slots on this card.
- Server started at 32K context first, then retried the 32K context failures at 64K.

Context findings:

| Config | Result |
|---|---|
| `-c 32768 -b 2048 -ub 2048` | Stable, ~92.2 GiB VRAM used, ~5.7 GiB headroom. Two SWE-agent instances hit client-side context limits. |
| `-c 65536 -b 2048 -ub 2048` | Loaded, but failed during the Django retry with CUDA OOM. |
| `-c 65536 -b 1024 -ub 512` | Stable enough for the context-failure retry and the resumed run, ~87.9 GiB VRAM used, ~9.3 GiB headroom. |

The 64K retry helped the low-context failures:

- `astropy__astropy-14365`: changed from context failure to submitted.
- `django__django-11283`: changed from context failure to submitted, then hit the per-instance call limit.

Official SWE-bench harness result for the merged partial snapshot:

| Metric | Count |
|---|---:|
| Total Lite instances | 300 |
| Predictions submitted before abort | 72 |
| Non-empty patches evaluated | 58 |
| Resolved | 42 |
| Unresolved | 16 |
| Empty patches | 14 |
| Harness errors | 0 |
| Resolved / evaluated non-empty patches | 72.4% |
| Resolved / submitted predictions | 58.3% |

Important caveat: the partial run is early-slice biased, mostly Astropy and Django. The submitted-patch quality is real enough to notice, but it should not be linearly compared to the full 300-instance Qwen/Gemma runs.

Raw output:

- `experiments/sweagent_lite_deepseek_full/preds.snapshot_with_64k_retries.json`
- `experiments/sweagent_lite_deepseek_context_retries_64k/preds.json`
- `experiments/sweagent_lite_deepseek_full/eval/sweagent_lite_deepseek_full.deepseek-v4-flash-iq2xxs-partial-72.json`

## vLLM-Moet Smoke Test

I also tried `kacper-daftcode/vLLM-Moet` at commit `591250b` on 2026-07-09 using the official
`deepseek-ai/DeepSeek-V4-Flash` FP8/NVFP4 checkpoint on the root NVMe volume.

Setup:

- Docker image: `vllm-moet-sm120:v024`, 33.9 GB, built locally from `Dockerfile.sm120-v024`.
- Model footprint: 149 GB, 46 safetensors shards.
- Host: RTX PRO 6000 Blackwell, 125 GiB RAM, 64 GiB swap.
- Runtime config: FP8 KV, `deepseek_v4` tokenizer mode, 24K context, MTP with two draft tokens.

Results:

| Attempt | MoE knobs | Result |
|---|---|---|
| Delta pool | `VLLM_MOE_W2=1`, `VLLM_MOE_W2_DELTA_GB=1` | Engine reached DeepSeek V4/MTP setup and the FP4/MXFP4 MoE path, then was OOM-killed during load. GPU use was only about 10-13 GB; host OOM log showed about 101 GiB anonymous RSS. |
| No delta pool | `VLLM_MOE_W2=1`, `VLLM_MOE_W2_DELTA=0`, `VLLM_MOE_W2_DELTA_GB=0` | Same pre-readiness failure. Docker recorded `OOMKilled=true`; GPU use was about 13 GB; host OOM log showed about 117 GiB anonymous RSS. |

The surviving container log also warned that the `VLLM_MOE_W2*` variables were unknown vLLM
environment variables. Those variables do exist in the vLLM-Moet patch/docs, so this warning is
not by itself proof that the knobs were ignored, but it is an operational risk to keep in mind.

Verdict: on this host, the vLLM-Moet path failed on host RAM/load staging before the server became
ready. This was not a VRAM-fit failure and produced no benchmarkable throughput or SWE-bench result.
The official checkpoint snapshot was removed after the failed smoke test.

Raw notes:

- `experiments/vllm_moet_deepseek_v4_flash/README.md`

## Read

This is runnable on the 96 GB card, and the partial SWE-bench submitted-patch quality was better than the small coding suite suggested. It is still not an obvious replacement for the current Pro 6000 winners.

- The fit is real at 32K, but tight: 92.2 GiB used leaves only ~5.6 GiB on a 97.9 GiB card.
- 64K can run if batch sizing is reduced, but the stable configuration trades away more throughput and still leaves no room for parallel SWE-agent workers.
- Throughput is similar to dense Gemma 4 31B Q8 territory, but with much higher VRAM use.
- Coding quality is not S-tier on this harness. Capped score is 15/22, with the same broad LRU weakness that shows up in several Qwen-family runs.
- SWE-bench Lite looked much better on the partial slice: 42/58 resolved among non-empty evaluated patches, 42/72 among all submitted predictions. The run was aborted because of wall-clock cost and operational fit, not because the model was failing the harness.
- The antirez card explicitly says these quants are specific for the DS4 inference engine and "may work with other inference engines or not." llama.cpp loads and runs this file, but quality should be treated as a real empirical question, not assumed from the base model.
- The sokann card's KLD comparison also flags the q2 quant as lossy versus the near-lossless 146 GiB baseline, despite "works amazingly well" qualitative behavior.
- The vLLM-Moet official-checkpoint route did not get past model load on this 128 GB RAM host; it appears to need a much lower host-staging footprint or more RAM before it can be evaluated here.

Practical verdict: keep it as a runnable curiosity / DeepSeek-specific behavior test. For daily Pro 6000 coding, Qwen3.6-27B FP8 + DFlash, Opus-distilled Qwen3.6-35B-A3B, stock Qwen3.6-35B-A3B, Gemma 4 31B Q8, and gpt-oss-120b Q8 remain better-supported choices in this repo.
