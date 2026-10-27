# Qwen3.8-Flash-Next FP4 on RTX Pro 6000

Tested 2026-10-26 on the RTX PRO 6000 Blackwell workstation.

## Outcome

`nvidia/Qwen3.8-Flash-Next-FP4-Local` serves one agent at an 851,968-token window on the single 96 GB card.
It passes exact needle retrieval at 845,001 prompt tokens with a 96.0-second cold prefill, about 8,800 prompt tok/s.
Decode at that depth measured 158.24 tok/s with MTP-3, and short-context decode averaged 283.41 tok/s.
The lightweight coding suite scored 16/22 on the first pass at medium effort.
The only miss was LRU, which spent the whole 32K output budget reasoning without looping; one of two top_k 20 reruns then scored 6/6.
The API smoke test passed 7/7, including tool calls and preserved reasoning.
It became the opencode default on 2026-10-26, replacing the two-agent Qwen3.8-27B 512K route, which remains routable.

## Pinned inputs

| Component | Pin |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Weights | 21 safetensors shards, 95.17 GiB, at `/mnt/extended/models/qwen38-flash-next-fp4-local-b9ce18a6` |
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026` (`745feb916ee8`), vLLM `0.1.dev21830+g8244d1c44` |
| Serve script | `../opencode-config/hosts/rtxpro6000/serve-qwen38-flash-next-vllm.sh` |
| llama-swap route | `qwen38-flash-next-fp4-850k` |

## Architecture and memory

The model is a 125B-total, 6B-active MoE with Gated DeltaNet and Qwen Sparse Attention (QSA) layers in a 3:1 ratio.
Only 12 of 48 layers keep a KV cache, with 2 KV heads of dimension 256, so FP8 KV costs about 12 KiB per token.
The 51B-parameter n-gram (PLE) table stays in host RAM through `VLLM_PLE_CPU_OFFLOAD=1`.
Model loading took 71.1 GiB of VRAM, and the engine reported a 1,189,703-token FP8 KV pool, 1.40x the 851,968-token window.
Resident VRAM was 95,769 MiB with the card otherwise idle, and host RAM had about 45 GiB available afterward.

## Serving configuration

The script follows the model card's RTX PRO 6000 recipe: b12x MoE, linear, and GDN kernels, `modelopt_mixed`, FP8 KV, and MTP-3 with probabilistic draft sampling.
It differs in four ways.
It uses `--language-model-only` because opencode sends text only.
It sets `--max-num-seqs 1` for one full-window agent.
It applies Qwen's official static YaRN with factor 3.25 over the native 262,144 tokens, which gives exactly 851,968.
It passes `max_model_len` inside the speculative config.

The last change is required.
vLLM does not apply YaRN factors when it derives a length limit, so the MTP draft inherited a 262,144-token limit.
The b12x QSA kernel then failed during startup profiling with `QSA sequence length 851968 exceeds the configured limit 262144`.
The failed attempt is kept in `experiments/qwen38_flash_next_fp4_850k/server_attempt1_qsa_limit.log`.

## Results

| Check | Result |
|---|---|
| API smoke | 7/7 |
| Short decode, 512 tokens, medium effort | 283.41 tok/s mean, 0.040 s mean TTFT |
| Lightweight coding, medium, 32K output | 16/22 first pass (string 5/5, expression 5/5, A* 6/6, LRU 0/6) |
| LRU reruns, top_k 20 | 6/6 and 0/6 |
| Needle, 845,001 prompt tokens, depth 0.6 | PASS, 96.03 s cold TTFT |
| Decode at 845K depth, 1,024 tokens | 158.24 tok/s, 0.33 s TTFT on the cached prefix |
| Cold start through llama-swap | 311 s with the checkpoint in page cache; about 9 minutes uncached from the SATA SSD |

The thinking-off check returned no reasoning tokens.

## Prefill

Each cold row is a distinct prompt with no cached prefix.
Each append row adds about 25K new tokens on top of the cold prompt that was just cached, which is the cost an agent pays per turn at that depth.

| Cold prompt tokens | Cold TTFT | Cold average | Cached depth | Appended tokens | Append TTFT | Marginal rate |
|---:|---:|---:|---:|---:|---:|---:|
| 8,209 | 0.62 s | 13,254 tok/s | - | - | - | - |
| 32,772 | 2.52 s | 12,987 tok/s | 24,076 | 25,064 | 2.09 s | 11,993 tok/s |
| 131,068 | 10.69 s | 12,265 tok/s | 120,380 | 27,069 | 2.48 s | 10,913 tok/s |
| 262,161 | 23.03 s | 11,385 tok/s | 252,798 | 25,746 | 2.72 s | 9,448 tok/s |
| 524,304 | 52.73 s | 9,943 tok/s | 517,634 | 23,041 | 2.99 s | 7,717 tok/s |
| 786,427 | 88.52 s | 8,884 tok/s | 776,451 | 26,356 | 4.00 s | 6,591 tok/s |
| 845,001 | 96.03 s | 8,799 tok/s | - | - | - | - |

The marginal rate falls only about 1.8x from 24K to 776K of depth, because QSA attends to a bounded token budget per query and three of every four layers are linear attention.
For comparison, a fit to the Qwen3.8-27B 1M-route runs gives about 920 tok/s at 375K and 710 tok/s at 500K, and Nanbeige4.3-33B measured 148 tok/s at 375K.
The cache reused each prefix only up to its last hybrid-state checkpoint, 7K to 11K tokens before the end of the cached prompt, so the append rows count those recomputed tokens as new.
The sweep was run with `tools/prefill_depth_sweep.py`, and its output is in `experiments/qwen38_flash_next_fp4_850k/prefill_sweep.json`.

## Comparison

The Qwen3.8-27B FP8 512K route keeps two full 512K agents resident, but a cold 500K prefill takes about nine minutes there.
Flash-Next prefilled 845K in 96 seconds, but it is sized for one full-window agent.
Its first-pass lightweight score is below the 27B's medium-effort 18/22, and both reach 22/22 only on reruns.
SWE-bench Lite has not been run on Flash-Next.

## Artifacts

All artifacts are in `experiments/qwen38_flash_next_fp4_850k/`.
They include `server.log`, `api_smoke.json`, `lightweight/`, `lru_rerun_topk20_1/`, `lru_rerun_topk20_2/`, and `needle_1x845k.json`.
