# Muse Glimmer 30B FP8 + DFlash on RTX Pro 6000

Muse Glimmer 30B FP8 runs entirely on the RTX Pro 6000 Blackwell 96 GB through a patched, digest-pinned vLLM DFlash image.
The validated deployment exposes a 131,072-token context allocation, supports 16 sequences, and uses the checkpoint's native 15-token Muse assistant.

## Result summary

| Metric | Result |
|---|---:|
| VRAM reserved | 93,976 MiB |
| Context allocation | 131,072 tokens |
| Lightweight coding | **22/22** |
| Mean single-stream decode | **120.12 tok/s** |
| Median single-stream decode | **123.95 tok/s** |
| Mean TTFT | **84 ms** |
| Eight-stream aggregate decode | **626.22 tok/s** |
| Sixteen-stream aggregate decode | **726.55 tok/s** |
| SWE-bench Lite | **129/300, 43.0%** |
| Non-empty patch resolution | **129/209, 61.7%** |
| Harness errors | **0** |

The full SWE-bench Lite score ranks sixth among the nine deployments evaluated on the shared host and harness.
It narrowly exceeds NVIDIA Nemotron 3 Puzzle 75B FP8 + MTP-3 at 128/300 and DeepSeek V4 Flash EXL3 2.04 bpw at 121/300.
It trails Qwen3-Coder-Next FP8 at 136/300 and the four Qwen3.6 deployments above that.

## DFlash effect

The matched no-DFlash baseline also scored 22/22, so the lightweight test found no quality regression from speculative decoding.
DFlash increased mean single-stream decode from 48.98 to 120.12 tok/s, a 2.45x gain.
The four coding tasks completed in 158.29 seconds with DFlash versus 514.79 seconds without it, a 3.25x wall-clock improvement.

The 15-token assistant accepted about 15% of proposed tokens in the synthetic concurrency sweep.
Even with that modest token acceptance rate, batching scaled aggregate throughput from 121.28 tok/s at one stream to 626.22 tok/s at eight and 726.55 tok/s at sixteen.
Eight workers provided the better agent-run balance because per-agent throughput remained 78.28 tok/s, compared with 45.41 tok/s at sixteen.

## SWE-bench Lite run

The full 300-instance generation phase ran from 2026-08-10 20:32:16 PT to 2026-08-11 01:36:03 PT, or 5h03m47s.
Official evaluation finished at 02:03:42 PT, bringing total elapsed time to about 5h31m.

| Outcome | Count |
|---|---:|
| Resolved | 129 |
| Unresolved non-empty | 80 |
| Empty patch | 91 |
| Submitted predictions | 300 |
| Harness errors | 0 |

The 61.7% resolution rate among non-empty patches is substantially stronger than the 43.0% end-to-end score.
Patch coverage is therefore the main quality constraint: 91 instances, or 30.3%, ended without a patch.

The canonical artifacts are:

- [`../experiments/sweagent_lite_muse_glimmer_30b_fp8_dflash15_c8/preds.json`](../experiments/sweagent_lite_muse_glimmer_30b_fp8_dflash15_c8/preds.json)
- [`../experiments/sweagent_lite_muse_glimmer_30b_fp8_dflash15_c8/eval/muse-glimmer-30b-fp8-dflash15-c8-full300.json`](../experiments/sweagent_lite_muse_glimmer_30b_fp8_dflash15_c8/eval/muse-glimmer-30b-fp8-dflash15-c8-full300.json)
- [`../experiments/muse_glimmer_30b_fp8_dflash15/results.json`](../experiments/muse_glimmer_30b_fp8_dflash15/results.json)
- [`../experiments/muse_glimmer_30b_fp8_dflash15/concurrency.json`](../experiments/muse_glimmer_30b_fp8_dflash15/concurrency.json)
- [`../experiments/muse_glimmer_30b_fp8_no_dflash/results.json`](../experiments/muse_glimmer_30b_fp8_no_dflash/results.json)

## Reproducible deployment

The derived image starts from the dedicated Muse Glimmer vLLM image by immutable digest.
Its build-time patch adds guarded native assistant registration and compatibility fixes for Muse configuration fields, exported assistant weight names, and the direct language-model wrapper.
Each replacement verifies its expected upstream source before modifying it, so a changed base image fails visibly instead of silently receiving a stale patch.

Build and launch:

```bash
docker build -t local/vllm-muse-glimmer:dflash-native containers/muse-glimmer-dflash
bash containers/muse-glimmer-dflash/run_server.sh
```

The default server configuration is:

- FP8 W8A8 target checkpoint and Muse Glimmer assistant fully resident on one GPU.
- `--gpu-memory-utilization 0.97`.
- `--max-model-len 131072`.
- `--max-num-seqs 16`.
- Muse Glimmer tool-call and reasoning parsers.
- DFlash with 15 speculative tokens.

The 131K allocation is deliberate for this single-card deployment.
At 93,976 MiB reserved, the configuration leaves too little VRAM to extend the KV allocation to 250K without reducing concurrency, changing cache precision, or spilling state outside the GPU.

Run the full benchmark with:

```bash
bash tools/run_swebench_lite_muse_glimmer_fp8_dflash.sh
```

The script verifies API readiness, runs generation and tool-call protocol checks, launches SWE-agent with eight workers, validates all 300 predictions, and invokes the official SWE-bench harness.
