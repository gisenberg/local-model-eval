#!/usr/bin/env python3
"""
Coding benchmark for Qwen3.6-27B served via vLLM, run across three precision
variants: BF16 base, NVFP4 (our modelopt quant), and the vendor FP8 release.

Matches the methodology of tools/nvfp4_gemma31b_bench_v2.py so results plug
into the same cross-model comparison: 4 coding tasks, 3 runs each, T=0.3.

Assumes `vllm serve` is already running on --port serving the target checkpoint
under --served-name. Launch vLLM separately so this script can be re-run against
different variants without disk I/O churn. Example serve commands:

  # BF16
  vllm serve ~/models-vllm/qwen36-27b-bf16-hf \\
      --served-model-name qwen36-27b-bf16 --port 8090 \\
      --max-model-len 16384 --gpu-memory-utilization 0.90

  # NVFP4
  vllm serve ~/models-vllm/qwen36-27b-nvfp4 \\
      --served-model-name qwen36-27b-nvfp4 --port 8090 \\
      --quantization modelopt_fp4 \\
      --max-model-len 16384 --gpu-memory-utilization 0.90

  # Vendor FP8
  vllm serve ~/models-vllm/qwen36-27b-fp8 \\
      --served-model-name qwen36-27b-fp8 --port 8090 \\
      --max-model-len 16384 --gpu-memory-utilization 0.90
"""

import argparse, json, os, re, subprocess, sys, time
import requests

TEMP = 0.3
RUNS = 3
# Qwen3.6 is a thinking model — reasoning traces easily exceed 8192. Matches
# rtxpro6000_coding_bench.py's budget for Qwen3.6-35B-A3B and leaves ~1.4K for
# the prompt within the 16384 max-model-len window.
MAX_TOKENS = 15000

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator", "expected": 5,
        "prompt": "Build a mathematical expression evaluator in Python. Requirements:\n1. Support +, -, *, / with correct operator precedence\n2. Support parentheses for grouping\n3. Support unary minus (e.g., '-3', '-(2+1)')\n4. Support floating point numbers (e.g., '3.14')\n5. Raise ValueError for: mismatched parentheses, division by zero, invalid tokens, empty expressions\n6. Implement as class ExpressionEvaluator with evaluate(expr: str) -> float\n7. Use recursive descent parser — no eval() or ast.literal_eval()\n8. Include type hints and docstrings\n9. Write 5 pytest tests",
    },
    "astar": {
        "name": "A* Pathfinding", "expected": 6,
        "prompt": "Implement A* pathfinding on a weighted 2D grid in Python.\n1. Class AStarGrid with find_path(start, end) -> Optional[List[Tuple[int,int]]]\n2. 4-directional, Manhattan heuristic, heapq, walls (0), weighted cells\n3. Handle: start==end, walls, out-of-bounds (ValueError)\n4. Path must be optimal. Include type hints and docstrings\n5. Write 6 pytest tests",
    },
    "lru_cache": {
        "name": "LRU Cache with TTL", "expected": 6,
        "prompt": "Implement LRU cache with TTL in Python.\n1. Class TTLCache(capacity, default_ttl)\n2. get(key), put(key, value, ttl=None), delete(key), size()\n3. O(1) avg time. Doubly-linked list + hash map, no OrderedDict\n4. time.monotonic(), lazy cleanup. Type hints and docstrings\n5. Write 6 pytest tests using unittest.mock.patch on time.monotonic",
    },
    "string_processor": {
        "name": "String Processor", "expected": 5,
        "prompt": "Write class StringProcessor with:\n1. reverse_words(s) -> str\n2. count_vowels(s) -> int (case-insensitive)\n3. is_palindrome(s) -> bool (ignore case, spaces, punctuation)\n4. caesar_cipher(s, shift) -> str (a-z/A-Z only, support negative)\n5. most_common_word(s) -> Optional[str] (case-insensitive, first if tied)\nInclude type hints, docstrings, and 5 pytest tests.",
    },
}


def run_inference(prompt, port, model):
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMP,
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    d = resp.json()
    msg = d["choices"][0]["message"]
    u = d.get("usage", {})
    comp = u.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        "tokens": comp,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(comp / elapsed if elapsed > 0 else 0, 1),
        "finish_reason": d["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content, test_file):
    # Qwen3.6 is a reasoning model — strip <think>...</think> before extracting code blocks,
    # otherwise we collect scratchwork snippets and glue them into invalid Python.
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0}
    combined = "\n\n".join(b.strip() for b in blocks)
    for cls in ["ExpressionEvaluator", "AStarGrid", "TTLCache", "StringProcessor", "BST", "Node"]:
        combined = re.sub(rf"from \w+ import {cls}\b.*\n", "", combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        out = r.stdout + r.stderr
        return {
            "passed": len(re.findall(r" PASSED", out)),
            "failed": len(re.findall(r" FAILED", out)),
            "errors": len(re.findall(r" ERROR", out)),
        }
    except Exception:
        return {"passed": 0, "failed": 0, "errors": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--served-name", required=True, help="served-model-name registered with vllm serve")
    ap.add_argument("--output-dir", required=True, help="e.g. experiments/nvfp4_qwen36_27b/nvfp4")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Qwen3.6-27B coding bench ({args.served_name}) ===")
    print(f"Temp={TEMP}, {RUNS} runs, port={args.port}\n")

    all_results = {}
    for bk, bench in BENCHMARKS.items():
        runs = []
        for run in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
            try:
                result = run_inference(bench["prompt"], args.port, args.served_name)
                tf = f"{args.output_dir}/{bk}_run{run}_test.py"
                tr = extract_and_test(result["content"], tf)
                print(f" {result['tok_per_sec']:.0f} tok/s | {tr['passed']}/{bench['expected']} | {result['tokens']} tok")
                runs.append({**result, **tr})
            except Exception as e:
                print(f" ERROR: {e}")
                runs.append({"error": str(e)})

        valid = [r for r in runs if "passed" in r]
        if valid:
            best = max(r["passed"] for r in valid)
            avg = sum(r["passed"] for r in valid) / len(valid)
            all_results[bk] = {"best": best, "avg": round(avg, 1), "expected": bench["expected"], "runs": runs}
            print(f"  -> Best: {best}/{bench['expected']}, Avg: {avg:.1f}/{bench['expected']}\n")
        else:
            all_results[bk] = {"runs": runs, "error": "all runs failed"}

    total_best = sum(min(v.get("best", 0), v.get("expected", 0)) for v in all_results.values())
    total_avg = sum(min(v.get("avg", 0), v.get("expected", 0)) for v in all_results.values())
    total_exp = sum(v.get("expected", 0) for v in all_results.values())
    print(
        f"\nTOTAL: Best-of-{RUNS} = {total_best}/{total_exp} "
        f"({total_best/total_exp*100:.0f}%), Avg = {total_avg:.1f}/{total_exp}"
    )

    all_results["_meta"] = {"served_name": args.served_name, "port": args.port, "temp": TEMP, "runs": RUNS}
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
