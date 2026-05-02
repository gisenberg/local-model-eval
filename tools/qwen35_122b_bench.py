#!/usr/bin/env python3
"""
Coding benchmark for RedHatAI/Qwen3.5-122B-A10B-NVFP4 served via vLLM on the
RTX Pro 6000.

Sampling matches the model card's published evaluation config:
  temperature=1.0, top_p=0.95, top_k=20, min_p=0.0
  presence_penalty=1.5, repetition_penalty=1.0

Same 4-task coding suite / 3 runs / pytest scoring as the other vLLM-served
benches in this repo.

Assumes vLLM is already running on --port (default 8090) with the model
exposed as `qwen35-122b-nvfp4`. Example serve command (from the dflash-pr40898
venv with cu13 toolchain):

  vllm serve ~/models-vllm/qwen35-122b-nvfp4 \\
      --served-model-name qwen35-122b-nvfp4 \\
      --port 8090 --max-model-len 32768 \\
      --gpu-memory-utilization 0.85 \\
      --enable-prefix-caching --trust-remote-code \\
      --reasoning-parser qwen3 --language-model-only \\
      --moe-backend flashinfer_cutlass \\
      --enforce-eager
"""

import argparse, json, os, re, subprocess, sys, time
import requests

RUNS = 3

PRESET = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 8192,
}

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
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **PRESET,
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=body,
        timeout=1500,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    d = resp.json()
    msg = d["choices"][0]["message"]
    u = d.get("usage", {})
    comp = u.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", "") or msg.get("reasoning", ""),
        "tokens": comp,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(comp / elapsed if elapsed > 0 else 0, 1),
        "finish_reason": d["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content, test_file):
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
    ap.add_argument("--served-name", default="qwen35-122b-nvfp4")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Qwen3.5-122B-A10B-NVFP4 coding bench ({args.served_name}) ===")
    print(f"Preset {PRESET}, {RUNS} runs, port={args.port}\n")

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

    all_results["_meta"] = {
        "served_name": args.served_name,
        "port": args.port,
        "preset": PRESET,
        "runs": RUNS,
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
