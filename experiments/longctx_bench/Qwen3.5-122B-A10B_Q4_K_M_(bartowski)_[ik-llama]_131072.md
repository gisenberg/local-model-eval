# Qwen3.5-122B-A10B Q4_K_M (bartowski) [ik-llama] — ctx 131072

Prompt tokens: 94147
Completion tokens: 451
Elapsed: 193.19s
Tok/s: 2.33
Score: 6/6

## Output

```diff
--- a/tools/spark_bench.py
+++ b/tools/spark_bench.py
@@ -1030,6 +1030,10 @@ def main():
         "--kv", choices=list(KV_CONFIGS.keys()), default=None,
         help="Override the model's default KV cache config (e.g. 'asym-q8' for f16K/q8V)",
     )
+    parser.add_argument(
+        "--max-tokens", type=int, default=None,
+        help="Override the per-model max_tokens default for generation",
+    )
     args = parser.parse_args()
 
     os.makedirs(args.output_dir, exist_ok=True)
@@ -1045,6 +1049,10 @@ def main():
         # Per-model overrides
         chat_template = model_cfg.get("chat_template")
         max_tokens = model_cfg.get("max_tokens", 16384)
+        if args.max_tokens is not None:
+            # CLI flag overrides per-model config
+            max_tokens = args.max_tokens
+
         request_timeout = model_cfg.get("request_timeout", 600)
 
         if args.kv:
@@ -1123,7 +1131,7 @@ def main():
                 print(" ready")
 
                 result = run_throughput_benchmark(
-                    args.port, model_name, kv_label,
+                    args.port, model_name, kv_label, max_tokens=max_tokens,
                     request_timeout=request_timeout,
                 )
                 all_results.append(result)
@@ -1131,7 +1139,7 @@ def main():
                 if not args.throughput_only:
                     bench_results = run_code_benchmarks(
                         args.port, model_name, kv_label, args.output_dir,
-                        max_tokens=max_tokens, request_timeout=request_timeout,
+                        max_tokens=max_tokens, request_timeout=request_timeout,
                     )
                     all_results.extend(bench_results)
 
```
