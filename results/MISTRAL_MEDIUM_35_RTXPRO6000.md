# Mistral Medium 3.5 128B Q4 on RTX Pro 6000 — first-look

**Status:** tested, **not competitive on this hardware**. 20 tok/s decode against the existing field's 195–264 tok/s, mid-tier coding score (15/22 = 68% on Q4_K_M, 11/22 = 50% on UD-Q4_K_XL). 13× slower than gpt-oss-120b at lower coding quality. Filed as a "we measured it so we don't have to wonder" reference; not added to the rankings doc.

## What it is

[`unsloth/Mistral-Medium-3.5-128B-GGUF`](https://huggingface.co/unsloth/Mistral-Medium-3.5-128B-GGUF) — Mistral's open-weight 125B **dense** model (not MoE), `mistral3` architecture, 88 layers, 256K native context, modified-MIT license. Vision support via `mmproj-*.gguf` (untested). The unsloth release fixes a YaRN `mscale_all_dim` bug from the initial Mistral upload — corresponding llama.cpp fix is [`048a490`](https://github.com/ggml-org/llama.cpp/commit/048a490) "convert: Mistral format yarn apply_scale support" (2026-04 ish), so a build older than that won't apply YaRN scaling correctly. We rebuilt llama.cpp at master `bbeb89d` (b9026) before testing.

We pulled the two Q4 variants the unsloth card highlights as recommended:

| Quant | File size (3-shard) | Footprint at 65K ctx |
|---|---|---|
| Q4_K_M | 74.9 GB | 94,076 MB VRAM |
| UD-Q4_K_XL ("Unsloth Dynamic") | 75.7 GB | 94,830 MB VRAM |

UD-Q4_K_XL is unsloth's dynamic quantization variant — slightly heavier but marketed as higher fidelity. We tested both head-to-head.

## Hardware fit

```
weights (Q4_K_M)        : 75 GB
KV cache @ 65K ctx (fp16): 19 GB           ←  88 layers × n_embd_k_gqa=1024 × 2(K+V) × 2 bytes × tokens
runtime + buffers       : ~0 GB
                          -------
                          94 GB    of 96 GB available
```

Both quants fit at 65K context with ~2 GB of VRAM headroom. **At native 256K context the model would not fit** — the KV cache alone would need ~75 GB. For full-context use cases on this card, you'd need to either drop to a smaller quant (Q3_K_M = 60.6 GB) or compress the KV cache.

## Throughput

5 timed runs each, streaming, T=0, 256-token completions, CUDA backend (llama.cpp `bbeb89d`).

| Quant | TTFT | Decode tok/s | VRAM @ 65K |
|---|---|---|---|
| Q4_K_M | 156 ms | **20.35** | 94.1 GB |
| UD-Q4_K_XL | 158 ms | **20.16** | 94.8 GB |

UD is ~1% slower (more bits → more bandwidth per token), within run-to-run noise. Both runs are extremely stable: 5 timed decodes hit within ±0.02 tok/s of each other.

**Bandwidth utilization:** 75 GB × 20.35 tok/s ÷ 1792 GB/s = **85%** of the card's quoted memory bandwidth. This is a *good* utilization number — the card is being fed efficiently — it just doesn't matter, because the model is too big to ever go fast on a single Pro 6000. **Pure dense 125B at any 4-bit quant is bandwidth-locked at ~20 tok/s on this card**, no matter how clever the kernel is.

For comparison on the same hardware:
| Model | Decode tok/s | Coding |
|---|---|---|
| gpt-oss-120b Q8_0 (sparse MoE) | 264 | 21/22 |
| Qwen3.6-35B-A3B Q8_0 (sparse MoE) | 221 | 15/22 |
| Qwen3-Coder-Next Q6_K | 196 | 15/22 |
| Gemma-4-31B-it Q8_0 (dense, smaller) | 44 | 22/22 |
| **Mistral Medium 3.5 Q4_K_M (dense, 125B)** | **20** | **15/22** |
| **Mistral Medium 3.5 UD-Q4_K_XL** | **20** | **11/22** |

The throughput penalty is structural: dense × 125B params means every token reads ~75 GB of weights, and the card has 1792 GB/s. Twenty is the speed-of-light number here.

## Coding score

4-benchmark suite (String 5 + ExprEval 5 + A* 6 + LRU 6), single run, T=0, ctx=32K, max_tokens=16384.

| Benchmark | Q4_K_M | UD-Q4_K_XL |
|---|---|---|
| String Processor | 5/5 | 5/5 |
| Expression Evaluator | 4/5 | **1/5** |
| A* Pathfinding | 4/6 | 4/6 |
| LRU Cache with TTL | 2/6 | 1/6 |
| **Total** | **15/22 (68%)** | **11/22 (50%)** |

UD-Q4_K_XL underperforms Q4_K_M by 4 points — opposite of what its "dynamic = higher fidelity" framing suggests. With a single run per cell we can't claim this is a reliable regression versus run-to-run variance, but it's at least not the speedup-free quality gain unsloth's framing implies.

### Failure modes (read carefully — most are model-authored buggy *tests*, not buggy impls)

The harness runs the model's own pytest tests against the model's own implementation, so a failing test can mean **either** a bug in the impl **or** a bug in the test. Breakdown of the failures:

- **Q4_K_M Expression Evaluator (1/5 fail)** — `test_parentheses` asserts `((8/2)+3)*2 == 16.0`. The model returned 14.0, which is correct (`((4)+3)*2 = 14`). The model's *test* is wrong; the *impl* is right. Score is harsh here.
- **Both quants A* (2/6 fail)** — `test_simple_path_uniform_grid` and `test_weighted_grid_prefers_lower_cost` both assert `total_cost == 4` for a 5-cell path on a uniform-cost grid (sum of 5 cells of cost 1 = 5). Same story: impl is correct, test is wrong about what "total_cost" should equal when including the start cell.
- **UD-Q4_K_XL Expression Evaluator (4/5 fail)** — this one is a real impl bug: every multi-token expression hits `ValueError: Unexpected end of expression`, including `"3 + 4"` and `"-3"`. The recursive-descent parser is mis-wired; a `_parse_term` invocation calls `_consume()` before peeking. Real regression vs Q4_K_M.
- **Both quants LRU Cache (2/6 and 1/6)** — the model wrote a `time.monotonic` mock with a fixed-length `side_effect` list that runs out partway through the test, raising `StopIteration`. The cache impl itself looks broadly OK; the test-fixture design is wrong. Same blind-spot pattern as every other [non-Gemma model in our lineup](MODEL_RANKINGS_RTXPRO6000.md) — only Gemma 4 31B and gpt-oss-120b reliably pass LRU.

**Real-impl-bug summary:** UD-Q4_K_XL is the only configuration with a *clearly* broken implementation (the parser). The other ~6 failures are split between the well-known LRU-mock blind spot (4 tests) and buggy-test issues that the impl handles correctly (3 tests).

If we re-ran 3× and took best-of-3 (the methodology used for [Nemotron-3-Omni](NEMOTRON3_OMNI_RTXPRO6000.md)), we'd likely see the buggy-test failures clear up on at least one run, lifting headline numbers by 2–4 points. We didn't bother because the throughput floor is the disqualifying constraint.

## Setup notes

Nothing exotic. The flow:

1. **Pull both shards.** Each Q4 variant lives in a subfolder (`Q4_K_M/`, `UD-Q4_K_XL/`) and is sharded as `*-00001-of-00003.gguf` (7 MB header), `*-00002-of-00003.gguf` (49.9 GB), `*-00003-of-00003.gguf` (~25 GB).
   ```
   hf download unsloth/Mistral-Medium-3.5-128B-GGUF \
     --include "Q4_K_M/*"     --local-dir ~/models/mistral-medium-3.5-q4km
   hf download unsloth/Mistral-Medium-3.5-128B-GGUF \
     --include "UD-Q4_K_XL/*" --local-dir ~/models/mistral-medium-3.5-udq4kxl
   ```
2. **Rebuild llama.cpp at ≥ b9026.** The `mistral3` arch enum is older but the YaRN `mscale_all_dim` fix lives at `048a490`. Older builds will load and *seem* fine on short prompts but apply the wrong rope scaling at long context. With `bbeb89d` (b9026) and ctx=65536 we observed normal behavior.
3. **Bench.** Reuses `tools/rtxpro6000_bench.py` and `tools/rtxpro6000_coding_bench.py`. Both now honor `LLAMA_PORT` so they don't collide with `llama-swap` on 8080:
   ```
   LLAMA_BACKEND=cuda LLAMA_PORT=8088 \
     python3 tools/rtxpro6000_bench.py mistral-medium-3.5-q4km
   LLAMA_BACKEND=cuda LLAMA_PORT=8088 \
     python3 tools/rtxpro6000_coding_bench.py mistral-medium-3.5-q4km
   ```

## Verdict

**Not competitive on this card** for the workloads this repo measures:

- **Throughput:** 20 tok/s. That's slower than the M4 Max running 30B-class dense models. The Pro 6000's 1792 GB/s + 96 GB sells well for *fitting* large models, but at 75 GB of weights you've already given up the throughput half of "fits *and* serves fast." gpt-oss-120b (sparse MoE) sits at 264 tok/s in the same VRAM envelope.
- **Coding quality:** 15/22 best of two attempts. Same tier as Qwen3.6-35B-A3B Q8 (15/22) which is 11× faster, and four points behind Nemotron-3-Omni FP8+thinking (19/22) which is 14× faster.
- **Quant cliff:** UD-Q4_K_XL's "dynamic" framing did not deliver here; its parser regressed badly relative to vanilla Q4_K_M. Single-run sample so this could be variance, but it's enough to wave us off promoting the dynamic quant as the obvious pick.

**Where this *would* matter:** if you specifically need an open-weight European-licensed dense 125B for long-context document work or vision-language input, this is a viable option on a 96 GB card *if* you can tolerate 20 tok/s and stay under ~65K context. Our bench suite doesn't measure either of those — long-context retrieval and vision-grounded coding are out of scope here — so this verdict only applies to the from-scratch coding-quality use case. If multimodal becomes interesting we'd revisit at Q4_K_M with `mmproj-F16.gguf` loaded.

**Not added to [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md).** Same disposition as [Qwen3.5-122B-A10B-NVFP4](https://github.com/gisenberg/local-model-eval/commit/c243ca5) — measured, filed, not promoted to the production tier list because faster + higher-quality alternatives already occupy the slot.

## Raw artifacts

- `experiments/rtxpro6000_bench_cuda/mistral-medium-3.5-q4km.json`
- `experiments/rtxpro6000_bench_cuda/mistral-medium-3.5-udq4kxl.json`
- `experiments/rtxpro6000_coding/mistral-medium-3.5-q4km.json` + `mistral-medium-3.5-q4km/*.md` (per-bench raw model output)
- `experiments/rtxpro6000_coding/mistral-medium-3.5-udq4kxl.json` + `mistral-medium-3.5-udq4kxl/*.md`
- Bench scripts (modified for LLAMA_PORT env override): `tools/rtxpro6000_bench.py`, `tools/rtxpro6000_coding_bench.py`

## See also

- [MODEL_RANKINGS_RTXPRO6000.md](MODEL_RANKINGS_RTXPRO6000.md) — the production-preset tier list; Mistral Medium 3.5 is not in it, by design.
- [NEMOTRON3_OMNI_RTXPRO6000.md](NEMOTRON3_OMNI_RTXPRO6000.md) — same hardware, dense+MoE multimodal alternative (30B-A3B), 19/22 at 287 tok/s in 33 GB. The model class to compare Mistral Medium 3.5 *against* if "I want a multimodal locally-served model" is the requirement.
- [LING_2_6_FLASH_RTXPRO6000.md](LING_2_6_FLASH_RTXPRO6000.md) — prior "tried it, shelved it" investigation; that one was sampling-config issues, this one is a structural throughput floor.
