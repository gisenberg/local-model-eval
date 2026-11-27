# Inkling-2-Mini on RTX Pro 6000

Tested 2026-11-20 on the RTX PRO 6000 Blackwell workstation, the day the model was released.
The image was rebuilt from vLLM main on 2026-11-22, see [Upstream main rebuild](#upstream-main-rebuild).
On 2026-11-24 the batched-token budget dropped to 2,048 and the image moved to main plus two open fixes, see [Live tuning](#live-tuning-on-2026-11-24).
On 2026-11-27 the memory fraction rose to 0.97 and the image moved to main plus a conv-state fix, which made room for three full-window agents, see [Conv-state fix](#conv-state-fix-on-2026-11-27).

## Outcome

`thinkingmachines/Inkling-2-Mini` serves as the opencode daily driver through vLLM with FP8 KV and DSpark speculative decoding.
One 3,516,686-token FP8 pool is shared by up to 16 concurrent requests at up to 1,000,000 tokens each, so three full-window agents fit at once with about 517K tokens left for subagents.
It scored 22/22 on the first lightweight pass at the model's default high effort, with every task ending on `stop`.
The API smoke test passed 7/7, including tool calls, a tool follow-up with string arguments, and replayed reasoning.
Three concurrent 985K-token needles and four concurrent 775K-token needles passed on the deployed image.
On the deployed image, single-stream decode measured 321 tok/s, and aggregate decode reached 1,148 tok/s at 8 requests and 1,854 tok/s at 16, with the agent lanes active.
It replaced Qwen3.8-Flash-Next as the opencode default on 2026-11-20.

SGLang was evaluated first and rejected.
Its RTX PRO 6000 attention path has no FP8 KV, and its hybrid radix cache dropped every agent's prefix once four or more long agents shared the card.

## Pinned inputs

| Component | Pin |
|---|---|
| Model | `thinkingmachines/Inkling-2-Mini`, revision `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` |
| Weights | One 19.53 GB safetensors shard plus a 636 MB DSpark head, at `/home/gisenberg/models-nvme/inkling-2-mini-1fb4134b` on the T705 NVMe, SHA-256 verified against the hub |
| Image | `local/vllm:inkling2-mini-ae5ded6f-pr69181-pr69288-pr70334-sm120`, vLLM main `ae5ded6f` plus vllm-project/vllm#69181, #69288, and #70334 (merged tree `e7ddacfe`), built for SM120 |
| Previous images | `local/vllm:inkling2-mini-65ada0c4-pr69297-pr69452-sm120` (main `65ada0c4` plus #69297 and #69452), `local/vllm:inkling2-mini-118d12f7-sm120` (vLLM main `118d12f7`), and `local/vllm:inkling2-mini-4c616217-sm120` (vllm-project/vllm#69028) |
| Build script | `../opencode-config/hosts/rtxpro6000/build-inkling2-mini-vllm.sh` |
| Serve script | `../opencode-config/hosts/rtxpro6000/serve-inkling2-mini-vllm.sh` |
| llama-swap route | `inkling-2-mini` |

The checkpoint is already mixed NVFP4 and MXFP8 as released, so no separate NVFP4 quantization was needed.

## Architecture and memory

The model has 24 weight layers, and layers 6 to 17 run three times with shared weights, for 48 layer visits per token.
Only layers 5, 11, 17, and 23 use global attention, and layers 11 and 17 sit inside the loop, so each token keeps full-length KV for 8 layer visits.
The other 40 visits use a 1,024-token sliding window.
Each global visit has 8 KV heads of dimension 128, so the full-length cache costs 16 KiB per token in FP8 and 32 KiB in BF16.

| Runtime | KV dtype | Speculation | Max requests | Pool tokens |
|---|---|---|---:|---:|
| vLLM | FP8 | none | 8 | 2,392,996 |
| vLLM | FP8 | DSpark-7 (BF16 draft KV) | 8 | 2,040,228 |
| vLLM | FP8 | DSpark-7 (BF16 draft KV) | 16 | 2,029,187 |
| vLLM, 1,000,000 window, memory 0.94 | FP8 | DSpark-7 (BF16 draft KV) | 16 | 1,991,020 |
| vLLM, 1,000,000 window, memory 0.95, PR build | FP8 | DSpark-7 (BF16 draft KV) | 16 | 2,020,260 |
| vLLM main build, 1,000,000 window, memory 0.95, 8,192 batched tokens | FP8 | DSpark-7 (BF16 draft KV) | 16 | 2,020,200 |
| Same, 4,096 batched tokens | FP8 | DSpark-7 (BF16 draft KV) | 16 | 2,577,440 |
| Same, 2,048 batched tokens | FP8 | DSpark-7 (BF16 draft KV) | 16 | 2,983,188 |
| Same, memory 0.97 | FP8 | DSpark-7 (BF16 draft KV) | 16 | 3,067,471 |
| Main plus #69181, memory 0.97, 8,192 batched tokens | FP8 | DSpark-7 (BF16 draft KV) | 16 | 3,264,138 |
| Same, 2,048 batched tokens (deployed) | FP8 | DSpark-7 (BF16 draft KV) | 16 | 3,516,686 |
| SGLang | BF16 | DSpark-8, SWA ratio 0.10, static 0.94 | 4 | 1,157,248 |
| SGLang | BF16 | DSpark-8, SWA ratio 0.05, static 0.94 | 4 | 1,335,296 |

vLLM reserved 67.5 GiB for KV at a 0.94 memory fraction and 68.5 GiB at 0.95.
The PR build's pool measured 2,020,260 tokens on three consecutive launches with the card otherwise idle, and the main build's measures 2,020,200.
vLLM sizes its reservation as a fraction of total memory, so another resident GPU job makes it refuse to start rather than shrink the pool.
The DSpark draft keeps its own KV in BF16 on SM12x, which costs about 350K tokens of pool.

## Serving configuration

The serve script follows the vLLM recipe: TP1, the `inkling` reasoning parser, and DSpark with 7 speculative tokens and probabilistic draft sampling.
It adds the `inkling` tool parser, FP8 KV, `--enable-prompt-tokens-details`, a 1,000,000-token window, 16 sequences, 2,048 batched tokens, and a 0.97 memory fraction.
The window is 1,000,000 rather than the native 1,048,576 because the window was sized when two native windows needed more than the pool held, and three still need 3.15M.

The published `vllm/vllm-openai:inkling-2-mini` image (`c2dc759f5984`) compiles vLLM's CUDA ops for SM80, SM89, SM90, and SM100 only.
Inkling's target kernels are Triton or CuTe and run on SM120, so the image serves without speculation.
The DSpark draft uses the generic FlashAttention backend and `_C` ops, and it failed at startup with `CUDA error: no kernel image is available for execution on the device`.
Rebuilding the same pull request with `torch_cuda_arch_list='12.0'` produced 59 SM120 kernels in `_C` and fixed DSpark.
The single-architecture build took about 75 minutes with 6 jobs.

This vLLM build clamps `max_tokens` to the remaining window instead of rejecting the request, so opencode's `limit.input` is the full window.
Reasoning effort is the top-level OpenAI `reasoning_effort` field, with named levels from `none` to `max` or a number from 0.0 to 0.99, and the default is 0.9.
Prior reasoning is replayed from either `reasoning_content` or `reasoning`, and opencode 1.18.34 sends `reasoning_content`.
`reasoning_effort: "none"` still produces a short thought, which matches the open SGLang bug sgl-project/sglang#43145 for the same model.
Image input works through the same route.

## Results

| Check | Result |
|---|---|
| API smoke | 7/7 |
| Lightweight coding, high effort, 32K output | 22/22 first pass (string 5/5, expression 5/5, A* 6/6, LRU 6/6) |
| Same suite, 8 concurrent requests | 5/8 runs full; two LRU runs reached the 32K cap mid-reasoning and one run wrote a wrong expected value in its own test |
| Short decode, 512 tokens | 283 tok/s mean, 0.045 s TTFT |
| Warm aggregate decode, 1,024 tokens, PR build | 327 (c1), 489 (c2), 695 (c4), 1,188 (c8), 1,730 (c16) tok/s |
| Warm aggregate decode, 1,024 tokens, main build | 344 (c1), 696 (c4), 1,240 (c8), 1,460 (c12), 1,924 (c16) tok/s |
| Needles, two concurrent 1.04M-token prompts, FP8 KV | both PASS; 565 and 868 s cold TTFT; follow-ups cached 1,039,936 tokens |
| Decode at 1.04M depth, no speculation, two concurrent | 46 and 52 tok/s |
| Needles, two concurrent 985K-token prompts, deployed DSpark route at a 1,000,000 window | both PASS; 427 and 802 s cold TTFT; follow-ups cached 984,960 tokens; decode at depth 50 and 60 tok/s |
| Cold start through llama-swap | 45 s with warm compile caches; about 230 s on the first launch |
| Eight concurrent `opencode run` sessions | 8/8 finished with passing tests in 43 to 68 s, no 429s or timeouts |

The eight-way LRU outputs were coherent edge-case reasoning that ran out of the bench's 32K budget, not degenerate text.
The wrong test expected `"Vnqlc"` for "World" shifted by -1, while the implementation correctly returned `"Vnqkc"`.
The first request at each new batch shape pays about 2 seconds of Triton compilation, which is then cached on the NVMe runtime cache.

## Upstream main rebuild

On 2026-11-22 the image was rebuilt from vLLM main at `118d12f7`.
That commit adds the merged Inkling port (vllm-project/vllm#69028) and two SM12x speedups.
vllm-project/vllm#69021 keeps the TMA k-group partials path on SM12x, and vllm-project/vllm#69168 tunes small-batch GEMMs for SM120.
The build has 60 SM120 kernels in `_C`, and DSpark starts without patches.
The fix for the original SM120 DSpark failure, vllm-project/vllm#69096, is still open, and the local SM120 build avoids it.

Both images were measured back to back on the same day with the card otherwise idle.
Each row is the mean of two trials of 1,024 output tokens after a warm-up pass at every batch shape.

| Concurrent requests | PR build (`4c616217`) | Main build (`118d12f7`) |
|---:|---:|---:|
| 1 | 293 (290 to 296) | 344 (316 to 372) |
| 4 | 624 (543 to 705) | 696 (660 to 731) |
| 8 | 1,190 (1,112 to 1,267) | 1,240 (1,208 to 1,273) |
| 12 | 1,451 (1,390 to 1,511) | 1,460 (1,386 to 1,535) |
| 16 | 1,911 (1,831 to 1,991) | 1,924 (1,851 to 1,996) |

The single-stream gain is outside trial noise, and at 8 or more requests the two builds overlap.
Mean DSpark acceptance length was 3.4 to 3.9 tokens on both builds.
The PR build measured 327 tok/s single and 1,730 tok/s at 16 on 2026-11-20, so day-to-day variation is about as large as the multi-request difference.

Teacher-forced prompt logprobs over 16,586 positions from four fixed texts were bit-identical between the builds, with a perplexity of 2.969 on both.
The probe is `tools/prompt_logprob_compare.py`.
It exercises the prefill path, so the decode-path kernels were checked with repeated coding passes instead.
Four concurrent lightweight passes scored 16, 21, 22, and 22 of 22 on the main build and 16, 21, 22, and 22 on the PR build.
Each build's 16 was one LRU run that reached the 32K output cap mid-reasoning, and each 21 was one wrong expected value in the model's own test.

On the deployed route the main build passed the API smoke test 7/7 and both concurrent 985K-token needles.
Cold TTFT was 430 and 804 s, follow-ups cached 984,960 tokens, and decode at depth rose to 67 and 69 tok/s from 50 and 60.
Cold start through llama-swap was 51 s.

A community recipe for this card runs DSpark with 4 draft tokens, which it reports as about 7% faster per step at one request.
Upstream vLLM rejects that with `Inkling2 DSpark requires num_speculative_tokens = 7 (the checkpoint's block size - 1)`, and the recipe patches the model code to allow it.
The deployment keeps 7 rather than carry a model-code patch.

## Live tuning on 2026-11-24

Two opencode agent lanes were using the route throughout, so every restart waited until vLLM reported no running and no waiting requests on two reads 5 seconds apart.
The three restarts waited 3, 9, and 3 minutes for idle and took 183, 55, and 122 seconds, the longer ones because a new batch shape or image compiles its kernels on first launch.
Each restart dropped the lanes' prefix cache.

### Batched-token budget

vLLM reserves short-convolution history in proportion to the batched-token budget, as vllm-project/vllm#69280 and #69444 describe, so a smaller budget leaves more of the same memory for KV.
All three settings ran on the `118d12f7` image.

| `--max-num-batched-tokens` | Pool tokens | Cold 32K / 131K / 262K | Append 16K at 262K depth |
|---:|---:|---|---:|
| 8,192 | 2,020,200 | 2.2 / 13.7 / 39.4 s | 4.9 s |
| 4,096 | 2,577,440 | 73.7 / 14.4 / 41.5 s | 4.9 s |
| 2,048 | 2,983,188 | 2.5 / 14.9 / 43.6 s | 5.0 s |

The 4,096 row's 32K prompt was the first request at that batch shape and includes kernel compilation.
The 2,048 row is a warm rerun with fresh prompts, and its 524K cold prompt took 142.5 s against 124 to 137 s at 8,192 on earlier runs.
So 2,048 buys 48% more pool for 8 to 14% slower cold prefill, with appends to a cached prefix unchanged.
Decode could not be compared cleanly because the lanes' traffic dominated every run.

Prompt logprobs changed slightly with the budget, with 96.0% top-1 agreement between 8,192 and 2,048 and perplexity 3.000 against 2.996.
At a fixed budget they were bit-identical across restarts and under load.
At 8,192 each short test text fits in one chunk, while at 2,048 later chunks read earlier tokens through the FP8 KV cache, as every long agent context already does.

### Main plus two open fixes

The image moved to vLLM main `65ada0c4` with two open pull requests merged by `VLLM_PRS` in the build script.
vllm-project/vllm#69297 stops the Inkling tool parser from re-parsing the whole argument on every streamed delta.
vllm-project/vllm#69452 stops Triton recompiles of the prefill attention kernels after a cold start.
The host had about 14 GB of free RAM next to the win-ci VM, so the build ran with 2 jobs and 1 nvcc thread and never dropped below 12 GB free.

| Check | `118d12f7` | `65ada0c4` plus fixes |
|---|---|---|
| Long streamed `write_file` call | 28 KB valid, longest delta gap 11.1 s, `/health` worst 10.6 s | 37 KB valid, longest gap 0.77 s, `/health` worst 0.02 s |
| Lightweight coding, four concurrent passes | 16, 21, 22, 22 of 22 (2026-11-22) | 22, 22, 22, 22 of 22 |
| API smoke | 7/7 | 7/7 |
| Single-stream decode, trials with no other request running | 327 and 281 tok/s | 313 and 248 tok/s |
| Mean DSpark acceptance length | 3.66 | 4.09 |
| Two concurrent 985K needles | pass | pass, follow-ups cached 984,960 tokens, decode at depth 39 and 49 tok/s with the lanes active |

Prompt logprobs at 2,048 agreed on 99.5% of top-1 tokens with the previous image, and perplexity moved from 2.9960 to 2.9965.
Single-stream trials that overlapped one lane request fell to 120 to 195 tok/s on the new image, which is why the trials above count only solo runs.
The tool-stream probe is `tools/long_tool_stream_probe.py`.

## Conv-state fix on 2026-11-27

vllm-project/vllm#69181 backs only the short-convolution rows a later step can read, instead of reserving rows for every in-flight token.
It is the single-GPU, SM12x form of the idea in #69444.
The image also merges #69288, which stops the MoE kernels recompiling for nearly every new prompt length on GPUs without TRT-LLM-gen kernels, and #70334, which retiles the SM120 decode kernels and adds programmatic dependent launch.
#69297 and #69452 from the previous image had merged into main by then.
The build ran with 2 jobs and 1 nvcc thread next to the win-ci VM.

Before the new image, the memory fraction rose from 0.95 to 0.97 on the previous image, which grew the pool from 2,983,188 to 3,067,471 tokens at 2,048 batched tokens.
The API smoke test passed 7/7 at 0.97, and the card had about 4.5 GiB unused under a 16-request decode load.

| Check | 2,048 batched tokens | 8,192 batched tokens |
|---|---|---|
| Pool tokens | 3,516,686 | 3,264,138 |
| Cold 32K / 131K / 262K | 2.59 / 17.48 / 43.22 s | 2.12 / 13.60 / 38.83 s |
| Append 16K at 32K / 131K / 262K depth | 1.67 / 3.11 / 4.92 s | 1.36 / 47.09 / 4.30 s |
| Aggregate decode at 1 / 4 / 8 / 16 requests | 321 / 738 / 1,148 / 1,854 tok/s | 363 / 719 / 1,128 / 1,943 tok/s |
| Prompt logprobs against the previous image at the same budget | 99.7% top-1 agreement, mean difference 0.010 | 100% top-1 agreement, maximum difference 1e-6 |
| API smoke | 7/7 | 7/7 |

The 8,192 append at 131K depth was the first request at a new batch shape and includes kernel compilation.
The pool still shrinks with the batched-token budget, but by 7% between 2,048 and 8,192 rather than the 32% before the fix.
2,048 was kept for the larger pool, since cold prefill is the only cost and agents mostly append to a cached prefix.

The 8,192 logprobs were compared with the `118d12f7` capture at the same budget and are bit-identical to rounding.
At 2,048 the agreement matches the noise between two captures on the previous image, 99.5% and a mean difference of 0.010.
Two of the four reference texts had been edited since the earlier capture, so `tools/prompt_logprob_compare.py` now skips texts whose tokens differ and lists them, and both comparisons cover the remaining 6,069 positions.

At 8,192 the long streamed `write_file` call produced 30 KB of valid arguments with a longest delta gap of 0.18 s, and `/health` answered within 0.03 s.
Four concurrent lightweight passes scored 22, 22, 22, and 21 of 22, and the miss was one `caesar_cipher` assertion at temperature 1.0.

Three concurrent 985K-token needles and four concurrent 775K-token needles, each 95% of a quarter of the pool, all passed retrieval at 8,192.
The win-ci agent lanes sent 82 requests during the needle runs, so these show the pool under real load rather than its clean capacity.
In the three-way run the lanes' contexts pushed the total past the pool, so two of the three follow-ups lost their prefix and re-prefilled 985K tokens, with no preemptions.
In the four-way run every follow-up hit the cache at 775,168 tokens, but KV usage reached 97.9%, vLLM preempted 51 times, and decode at depth fell to 6 to 7 tok/s per request.
Filling the pool to within a few percent while other agents are active is therefore not a usable operating point, and agent splits should leave room for subagents and the lanes.

## Prefill

Each cold row is a distinct prompt with no cached prefix.
Each append row adds about 16K new tokens to a cached prompt of the given depth.
vLLM caches at 16K-token boundaries, so the append rows include up to 16K tokens of recomputation.

| Cold prompt tokens | Cold TTFT | Cached depth | Append TTFT |
|---:|---:|---:|---:|
| 8,196 | 0.46 s | - | - |
| 32,778 | 4.28 s | 32,768 | 1.37 s |
| 131,086 | 13.11 s | 131,072 | 2.58 s |
| 262,155 | 38.22 s | 262,144 | 4.22 s |
| 524,302 | 124.48 s | 524,288 | 7.65 s |
| 786,447 | 258.32 s | 786,432 | 11.03 s |
| 1,004,693 | 406.5 s | - | - |

The marginal rate falls roughly in proportion to depth because the 8 global visits attend to the whole prefix.
Qwen3.8-Flash-Next appended at 786K depth in 4.0 seconds, so it remains the faster choice for a single very deep agent.

## Context window and agent permutations

The pool is shared at run time, so a single route covers every split of it.
A request may use up to 1,000,000 tokens, and up to 16 requests run at once.
Agents stay fast as long as the sum of their live contexts fits in the 3.52M-token pool.
The retention tests below ran at the earlier 2.02M pool and 8,192 batched tokens.

| Agents resident at once | Context each | Pool used | Notes |
|---:|---:|---:|---|
| 1 | 1,000,000 | 1.00M | leaves about 2.52M for subagents |
| 2 | 1,000,000 | 2.00M | about 1.52M left for subagents |
| 3 | 1,000,000 | 3.00M | the deployed configuration, about 517K left for subagents |
| 4 | 835K | 3.34M | about 180K left |
| 6 | 556K | 3.34M | |
| 8 | 415K | 3.32M | |
| 16 | 208K | 3.33M | the configured request ceiling |

The splits past three agents leave about 5% of the pool free, because the four-way needle run above preempted heavily with only 2% free.

When live contexts exceed the pool, vLLM evicts the least recently used prefix and that agent pays a cold prefill on its next turn.

## Prefix-cache retention

Each test sent a stale prompt first, then several long sessions, then two rounds of continuation turns per session.
The total exceeded the pool, so the cache had to evict the stale prompt and keep every active session.

| Runtime | Mix | Continuation hit ratio |
|---|---|---|
| vLLM, FP8, no speculation | stale 520K, then 2 x 1.0M | 1.000 |
| vLLM, FP8, no speculation | stale 520K, then 4 x 523K | 1.000 |
| vLLM, FP8, no speculation | stale 400K, then 8 x 260K | 1.000 |
| vLLM, FP8, DSpark | stale 520K, then 2 x 907K | 1.000 |
| vLLM, FP8, DSpark | stale 520K, then 4 x 481K | 1.000 |
| vLLM, FP8, DSpark | stale 400K, then 8 x 241K | 0.999 minimum |
| SGLang, SWA ratio 0.05 | stale 704K, then 2 x 523K | 0.000 |
| SGLang, SWA ratio 0.09 | stale 300K, then 2 x 522K | 1.000 |
| SGLang, SWA ratio 0.09 | stale 200K, then 4 x 278K | 0.000 |
| SGLang, SWA ratio 0.09 | stale 300K, then 8 x 130K | 0.000 |

The probe is `tools/prefix_cache_retention_probe.py`.

## Why SGLang was rejected

SGLang's verified RTX PRO 6000 recipe (`lmsysorg/sglang:inkling-2-mini`, digest `efac853b`) served correctly.
It scored 22/22 lightweight and 7/7 on the API smoke test, decoded at 316 tok/s single and 1,054 tok/s at eight requests, and retrieved a 1.04M-token needle.
Raising `--max-running-requests` past the recipe's cap of 4 worked.

Its FA4 attention on SM120 rejects FP8 KV with `FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe`, so the pool stays BF16.

Its hybrid radix cache stores one sliding window per prefill chunk, as `--hybrid-cache-checkpoints-level` documents for models with sliding-window layers.
A cached prefix therefore holds about 14% of its length in the separate SWA pool at the default 8K chunk size.
Each new long prefill evicts the oldest windows, including the tail window another agent needs for its next turn.
Once that tail is gone the agent's whole prefix is unreachable, its next turn re-prefills, and that prefill evicts the next agent.
Holding every agent's windows needs an SWA pool of about 14% of the total cached tokens, and each SWA token costs about 3.6 full-length tokens of memory.
A 16K chunk halves the retention but ran out of memory at a 0.94 static fraction.
The usable multi-agent pool would have been about 1.05M BF16 tokens, roughly half of vLLM's.

## Artifacts

All artifacts are in `experiments/inkling2_mini_rtxpro6000/`.
The main rebuild A/B, quality checks, and route validation are in `vllm_main_ab/`, and the 2026-11-24 live tuning and build validation are in `live_ab/`.
The 2026-11-27 memory-fraction check is in `live_ab/gpu097/` and the conv-state build validation in `live_ab/build_ae5ded6f_sconv/`.
The vLLM DSpark runs are in `vllm_sm120_dspark/`, the vLLM no-speculation runs in `vllm_fp8_nospec/`, and the retention tests in `vllm_cache_worstcase/` and `cache_worstcase/`.
The SGLang runs are at the top level of the directory and in `evict_ab/`, and the image build summary, including the SM120 kernel check, is in `vllm_build/build_summary.txt`.
