# Qwen3.5-122B-A10B Q4_K_M (bartowski) [ik-llama] — ctx 32768

Prompt tokens: 23670
Completion tokens: 448
Elapsed: 56.17s
Tok/s: 7.98
Score: 6/6

## Output

```diff
--- a/tools/spark_bench.py
+++ b/tools/spark_bench.py
@@ -630,6 +630,10 @@ def main():
         "--kv", choices=list(KV_CONFIGS.keys()), default=None,
         help="Override the model's default KV cache config (e.g. 'asym-q8' for f16K/q8V)",
     )
+    parser.add_argument(
+        "--max-tokens", type=int, default=None,
+        help="Override the per-model max_tokens default from the command line",
+    )
     args = parser.parse_args()
 
     os.makedirs(args.output_dir, exist_ok=True)
@@ -643,6 +647,10 @@ def main():
         chat_template = model_cfg.get("chat_template")
         max_tokens = model_cfg.get("max_tokens", 16384)
         request_timeout = model_cfg.get("request_timeout", 600)
+
+        # CLI override takes precedence over per-model config
+        if args.max_tokens is not None:
+            max_tokens = args.max_tokens
 
         if args.kv:
             kv_types = [args.kv]
@@ -723,7 +731,7 @@ def main():
                 print(" ready")
 
                 result = run_throughput_benchmark(
-                    args.port, model_name, kv_label,
+                    args.port, model_name, kv_label, max_tokens=max_tokens,
                     request_timeout=request_timeout,
                 )
                 all_results.append(result)
@@ -731,7 +739,7 @@ def main():
                 if not args.throughput_only:
                     bench_results = run_code_benchmarks(
                         args.port, model_name, kv_label, args.output_dir,
-                        max_tokens=max_tokens, request_timeout=request_timeout,
+                        max_tokens=max_tokens, request_timeout=request_timeout
                     )
                     all_results.extend(bench_results)
```
