# Qwen3.5-122B-A10B Q4_K_M (bartowski) [ik-llama] — ctx 65536

Prompt tokens: 47177
Completion tokens: 390
Elapsed: 95.35s
Tok/s: 4.09
Score: 5/6

## Output

```diff
--- a/tools/spark_bench.py
+++ b/tools/spark_bench.py
@@ -1070,6 +1070,10 @@ def main():
         "--kv", choices=list(KV_CONFIGS.keys()), default=None,
         help="Override the model's default KV cache config (e.g. 'asym-q8' for f16K/q8V)",
     )
+    parser.add_argument(
+        "--max-tokens", type=int, default=None,
+        help="Override the per-model max_tokens default (default: 16384 or model-specific)",
+    )
     args = parser.parse_args()
 
     os.makedirs(args.output_dir, exist_ok=True)
@@ -1085,6 +1089,10 @@ def main():
         # Per-model overrides
         chat_template = model_cfg.get("chat_template")
         max_tokens = model_cfg.get("max_tokens", 16384)
+        if args.max_tokens is not None:
+            max_tokens = args.max_tokens
+        else:
+            max_tokens = model_cfg.get("max_tokens", 16384)
         request_timeout = model_cfg.get("request_timeout", 600)
 
         if args.kv:
@@ -1165,7 +1173,7 @@ def main():
                 if not args.throughput_only:
                     bench_results = run_code_benchmarks(
                         args.port, model_name, kv_label, args.output_dir,
-                        max_tokens=max_tokens, request_timeout=request_timeout,
+                        max_tokens=max_tokens, request_timeout=request_timeout
                     )
                     all_results.extend(bench_results)
 
```
