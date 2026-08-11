# DeepSeek V4 Flash 0731 DSpark on RTX PRO 6000

Tested on 2026-08-02 with the retained antirez IQ2_XXS importance-matrix main checkpoint and the 0731 DSpark sidecar.

## Outcome

DSpark is worth using on this host.

The recommended setting is `--spec-draft-n-max 3` at a fixed 262,144-token context.

It averaged 164.02 decode tok/s versus 78.49 tok/s without DSpark on the same llama.cpp build, a 2.09x throughput result and a 109.0% gain.

The full-context validation used the higher-memory n=5 configuration and successfully processed a 249,985-token API prompt while retaining 1,373 MiB of raw VRAM headroom at the observed peak.

The recommended n=3 configuration loaded with 22 MiB less VRAM than n=5, so it preserves at least as much context headroom under the tested fixed settings.

## Reproducibility

The DSpark implementation came from llama.cpp commit `bb4e0e1b3f6bb38960769a1c9bcd2081016154cd`, reported as build 10231.

The build is at `/home/gisenberg/llama-build/src-deepseek-v4-dspark-bb4e0e1/build/bin`.

The sidecar is `am17an/DeepseekV4-Flash-20260731-DSpark` at revision `9d79f20040120924bd2f7dc4f3a9f86c721b39f8`.

Its GGUF is 10,896,057,568 bytes and has SHA-256 `835d0fc5216b8a71111492c3f9e64add1d72345befa750610fdfae1011adf08f`.

The checksum matches the Hugging Face LFS object identifier.

The tested server command for the recommended configuration is:

```bash
rtk proxy env \
  LD_LIBRARY_PATH=/home/gisenberg/llama-build/src-deepseek-v4-dspark-bb4e0e1/build/bin:/home/gisenberg/.micromamba/envs/cuda/lib \
  /home/gisenberg/llama-build/src-deepseek-v4-dspark-bb4e0e1/build/bin/llama-server \
  -m /mnt/extended/gisenberg/models/deepseek-v4-flash-antirez-imatrix-0731-1cd7b564/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --spec-draft-model /mnt/extended/gisenberg/models/deepseek-v4-flash-0731-dspark-9d79f200/DeepseekV4-Flash-20260731-DSpark.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  --spec-draft-ngl all \
  --host 127.0.0.1 \
  --port 8091 \
  -c 262144 \
  -ngl all \
  -fa on \
  -np 1 \
  --jinja \
  --fit off \
  -b 1024 \
  -ub 128 \
  -ctk f16 \
  -ctv f16 \
  --threads 16 \
  --threads-batch 32 \
  --ctx-checkpoints 0 \
  --metrics \
  --reasoning-format deepseek \
  --reasoning off \
  --reasoning-budget 0
```

`--fit off` is intentional because it prevents the runtime from silently shrinking the requested context or changing offload decisions.

## Throughput sweep

Every row used one warmup followed by five timed 256-token generations with a 262,144-token slot.

| Configuration | Loaded VRAM | Mean TTFT | Mean decode | Gain over baseline |
|---|---:|---:|---:|---:|
| No DSpark | 85,113 MiB | 79.2 ms | 78.49 tok/s | - |
| DSpark n=2 | 95,607 MiB | 84.9 ms | 138.03 tok/s | 75.9% |
| DSpark n=3 | 95,619 MiB | 92.0 ms | 164.02 tok/s | 109.0% |
| DSpark n=4 | 95,631 MiB | 89.5 ms | 157.30 tok/s | 100.4% |
| DSpark n=5 | 95,641 MiB | 93.9 ms | 150.65 tok/s | 91.9% |

n=3 was the clear optimum for the standard prompt.

Its five timed results ranged from 163.54 to 164.50 tok/s.

The n=3 sidecar costs 10,506 MiB of additional loaded VRAM relative to the non-DSpark baseline.

## Long-context validation

The long-context test deliberately used n=5 because it was the highest-memory DSpark point in the sweep.

It placed the retrieval needle at 75% depth and returned the exact expected value `739184`.

| Metric | Result |
|---|---:|
| Allocated context | 262,144 tokens |
| Raw prompt tokens | 249,981 |
| API prompt tokens | 249,985 |
| Retrieval | Pass |
| End-to-end time | 630.269 s |
| Average prompt evaluation | 396.86 tok/s |
| Peak observed total GPU use | 95,869 MiB |
| Raw VRAM free at peak | 1,373 MiB |

DSpark did not materially change prompt ingestion speed compared with the prior 395.19 tok/s baseline, which is expected because speculative decoding accelerates generation rather than prompt evaluation.

The three-token post-fill response measured 57.34 tok/s, but that sample is too short to treat as a stable decode benchmark.

## Quality and protocol

The recommended n=3 configuration scored 16/22 on the lightweight coding suite while retaining the full 262,144-token allocation.

| Task | Score |
|---|---:|
| String processor | 5/5 |
| Expression evaluator | 5/5 |
| A-star pathfinding | 6/6 |
| TTL/LRU cache | 0/6 |
| Total | 16/22 |

This matches the quant's previous best native total and is one point above its previous reasoning-off total.

An immediate second n=3 run reproduced the same 16/22 score, the same per-task split, and byte-identical responses for all four prompts.

The repeated result establishes deterministic reproducibility under this configuration, although the one-point comparison with the older reasoning-off run still should not be interpreted as a broad statistical quality improvement.

The n=3 server also passed all 7 API and tool-contract checks.

It produced the exact basic response, returned the correct arithmetic answer, suppressed reasoning content, emitted one correctly named tool call with exact arguments, and consumed the tool result without another tool call.

## Artifacts

- [Run manifest](../experiments/deepseek_v4_flash_antirez_imatrix_0731_dspark_250k/run_manifest.json)
- [Long-context result](../experiments/deepseek_v4_flash_antirez_imatrix_0731_dspark_250k/long_context_250k.json)
- [n=3 API smoke](../experiments/deepseek_v4_flash_antirez_imatrix_0731_dspark_250k/api_smoke_n3.json)
- [Baseline throughput](../experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731-no-think.json)
- [n=2 throughput](../experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731-dspark-n2.json)
- [n=3 throughput](../experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731-dspark-n3.json)
- [n=4 throughput](../experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731-dspark-n4.json)
- [n=5 throughput](../experiments/rtxpro6000_bench_cuda/deepseek-v4-flash-antirez-imatrix-0731-dspark-n5.json)
- [n=3 first lightweight coding result](../experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731-dspark-n3-run1.json)
- [n=3 lightweight coding result](../experiments/rtxpro6000_coding/deepseek-v4-flash-antirez-imatrix-0731-dspark-n3.json)

The benchmark harness keys are `deepseek-v4-flash-antirez-imatrix-0731-dspark-n2` through `deepseek-v4-flash-antirez-imatrix-0731-dspark-n5`.

Set `CODING_CTX=262144` when running the lightweight coding harness to preserve the same full-context allocation.
