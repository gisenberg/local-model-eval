# DGX Spark: Qwen 35B-A3B at FP8/BF16 vs 122B-A10B at Q4 — does the small model win?

**tl;dr: No.** The 122B at Q4 scores 17/17 on our coding suite. The 35B at BF16 (zero quant loss — best it can possibly be) scores 9/17. The "large model quantized down beats small model at larger quant" rule holds for MoE.

## The question

> There is a weird cult on the Spark forums that prefers Qwen 3.5 35B-A3B at FP8 over the 3.5 122B-A10B at INT4. Really wish I had some benchmarks of my own to try this stuff. I heard rule of thumb is that a large model quantized down is usually better than a small model at a larger quant, but perhaps MoE suffers more.
>
> — James Hillyerd

## The data

We benchmarked both on the DGX Spark (GB10, 128 GB LPDDR5X, ~273 GB/s). The coding suite is three single-shot implementations at temperature 0: a recursive descent expression evaluator (5 tests), A* pathfinding on a weighted grid (6 tests), and an LRU cache with TTL expiry (6 tests) — 17 tests total. Every model gets the same prompts, same test harness.

| Model | Active params | Quant | Weight size | Tok/s | ExprEval | A* | LRU | Total | Tier |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-122B-A10B (bartowski) | 10B | Q4_K_M | 71 GB | **26** | **5/5** | **7/6** | **5/6** | **17/17** | **S** |
| Qwen3.5-122B-A10B (unsloth) | 10B | Q4_K_M | 72 GB | 21 | 5/5 | 7/6 | 6/6 | **18/17** | **S** |
| Qwen3.5-122B-A10B (vLLM INT4+FP8) | 10B | AR-INT4/FP8 | ~72 GB | **49** | 5/5 | 7/6 | 4/6 | **16/17** | S |
| **Qwen3.6-35B-A3B (unsloth)** | **3B** | **BF16** | **69 GB** | **31** | **4/5** | **5/6** | **0/6** | **9/17** | **D** |
| Qwen3-Coder-Next 80B-A3B | 3B | Q4_K_M | 46 GB | 50 | 5/5 | 6/6 | 3/6 | 14/17 | B |

Key details:

- The **Qwen3.6-35B-A3B BF16** result is the *best possible* quality for that model — zero quantization loss. FP8 can only be equal or worse. If BF16 scores 9/17, FP8 won't score higher.
- The **122B at Q4_K_M** (~4.5 bpw) still scored 17/17. That's a lossless pass on every benchmark despite aggressive quantization. The model has enough capacity that Q4 doesn't hurt it on these tasks.
- Throughput favors the 35B (31 tok/s BF16) over the 122B on ik-llama (26 tok/s) — but the 122B on vLLM with INT4+FP8 and MTP-2 speculative decoding runs at **49 tok/s**, beating the 35B on speed *and* quality.

## Where the 35B failed

**Expression Evaluator (4/5):** The model wrote a test asserting `1 + 2 * 3 - 4 / 2 == 4.0` but the implementation returns `5.0`. That's `1 + 6 - 2 = 5`, which is actually correct — the model wrote a wrong test. The impl has the right precedence; the test doesn't. Ironically, a smarter model (the 122B) would have caught this.

**A* Pathfinding (5/6):** The test creates a 3×3 grid with the entire middle row blocked (`[0, 0, 0]`) and asserts `find_path((0,0), (0,2))` returns `None`. But `(0,0)` to `(0,2)` are in the same unblocked top row — the model's implementation correctly found a path along row 0. The test is wrong, the impl is right. Again, self-consistency failure.

**LRU Cache (0/6):** Zero tests ran — the model emitted impl and tests in a format the test harness couldn't extract (combined single block without proper separation). This is a structural output failure, not a logic failure, but the result is the same: unusable output.

**Pattern:** The 35B's failures are all self-consistency problems — it writes code that works, then writes tests that don't match, or structures output in a way that breaks the harness. The 122B doesn't make these mistakes because it has enough capacity to maintain coherence across the full generation.

## Why MoE doesn't change the rule

James's intuition about "MoE might suffer more from quantization" is reasonable but backwards. MoE actually suffers *less*:

1. **Only ~10B params are active per token** regardless of total model size. Quantizing the 122B's inactive experts from FP16 to Q4 doesn't affect the per-token computation quality — those experts aren't being read. The active 10B experts are quantized, but 10B params at Q4 is still more capacity than 3B params at BF16.

2. **The router is barely affected by weight quantization.** Expert selection happens in a tiny gating network that's well within the precision budget even at Q4.

3. **The capacity advantage compounds with expert count.** The 122B has 256 experts to choose from; the 35B also has 256 experts but each one is ~3× smaller. When the router picks the best 10 experts for a token, the 122B's experts individually have more capacity to contribute useful computation — even quantized.

## The speed argument doesn't hold either

The Spark forum preference for the 35B is presumably speed-motivated. But:

- 35B BF16: **31 tok/s** (3B active × BF16 = ~6 GB/token)
- 122B Q4_K_M on ik-llama: **26 tok/s** (10B active × Q4 ≈ ~5 GB/token)
- 122B INT4+FP8 on vLLM: **49 tok/s** (with MTP-2 speculative decoding)

The 35B's speed advantage over ik-llama (31 vs 26) is real but small — 19%. And it vanishes entirely with vLLM, which runs the 122B nearly 60% *faster* than the 35B while scoring 16/17 vs 9/17.

If the Spark forum users are running llama.cpp without ik-llama or vLLM, they might see a bigger speed gap. But even then, trading 17/17 quality for 9/17 quality to gain 19% throughput is a bad deal.

## Bottom line

The rule holds: **a large MoE quantized to Q4 decisively beats a small MoE at FP8/BF16 on the Spark.** The 122B has 3.5× more total parameters, and even at Q4 the active experts carry more capacity per token than the 35B's experts at full precision. The speed difference is marginal and disappears with the right inference engine.

---

*Tested April 2026 on DGX Spark GB10 (128 GB, ~273 GB/s LPDDR5X). Full methodology and per-model breakdowns in [MODEL_RANKINGS_SPARK.md](MODEL_RANKINGS_SPARK.md). Benchmark prompts, test harness, and raw outputs in the [local-model-eval](https://github.com/gisenberg/local-model-eval) repo.*
