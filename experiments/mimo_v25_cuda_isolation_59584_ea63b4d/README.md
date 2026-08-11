# MiMo V2.5 long-context CUDA isolation

This directory records the July 30, 2026 isolation matrix for the repeated CUDA crash seen during the MiMo V2.5 SWE-bench Lite retry.

## Fixed inputs

| Component | Value |
|---|---|
| Model | MiMo V2.5 `UD-IQ2_XXS`, revision `f7aff7868d5f79da58b505f84626d7a807393c37` |
| llama.cpp | `ea63b4d32ea1b66bdbe369be7f9443f6c00f8b31` |
| Driver and GSP | NVIDIA open kernel module `595.84` |
| Kernel | `7.0.0-28-generic` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB |
| Context | One 262,144-token slot |
| Workload | Shared 100K-token prefix followed by ten sequential generations of up to 4,096 tokens |

## Results

| Variant | Result |
|---|---|
| `fa_on_ub512` | Failed with CUDA launch timeout and Xid 8 after 6 complete responses |
| `fa_on_ub128` | Failed with CUDA launch timeout and Xid 8 during request 4 |
| `fa_off_f16kv_ub512` | Failed with CUDA OOM during prefill, no Xid |
| `fa_off_f16kv_ub128` | Passed 10/10 and generated 40,960 completion tokens |
| `fa_on_f16kv_ub128` | Passed 10/10 and generated 38,391 completion tokens |

The two `fa_off_ub*` directories are preserved startup checks that attempted Q8 V cache without flash attention.
llama.cpp rejected them because V-cache quantization requires flash attention.

The decisive control is `fa_on_f16kv_ub128`.
It keeps flash attention enabled and changes only K/V cache precision relative to `fa_on_ub128`.
Its clean result isolates the observed watchdog to Q8 KV handling under flash attention, or their interaction.

## Artifact layout

Each completed variant contains the exact `serve_cmd.txt`, streamed API `probe.json`, `server.log`, `kernel.log`, one-second `gpu_telemetry.csv`, and post-run `nvidia_smi_after.txt`.
The top-level `run.log` records the matrix launcher output.
The reusable drivers are [`../../tools/mimo_v25_cuda_decode_probe.py`](../../tools/mimo_v25_cuda_decode_probe.py) and [`../../tools/run_mimo_v25_cuda_isolation.sh`](../../tools/run_mimo_v25_cuda_isolation.sh).
