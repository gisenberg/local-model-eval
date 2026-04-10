#!/usr/bin/env python3
"""
TriAttention proof-of-concept: run coding benchmarks against Qwen3-8B
served via vLLM with TriAttention token eviction enabled.

Assumes vLLM is running on port 8090 with TriAttention plugin.
"""

import json
import os
import re
import subprocess
import sys
import time
import requests

PORT = 8090
MODEL = "Qwen/Qwen3-8B"
TEMP = 0.3
RUNS = 3
OUTPUT_DIR = "experiments/triattention_bench"

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator",
        "expected": 5,
        "prompt": (
            "Build a mathematical expression evaluator in Python. Requirements:\n"
            "1. Support +, -, *, / with correct operator precedence\n"
            "2. Support parentheses for grouping\n"
            "3. Support unary minus (e.g., '-3', '-(2+1)')\n"
            "4. Support floating point numbers (e.g., '3.14')\n"
            "5. Raise ValueError with a descriptive message for: mismatched parentheses, "
            "division by zero, invalid tokens, empty expressions\n"
            "6. Implement as a class called ExpressionEvaluator with an evaluate(expr: str) -> float method\n"
            "7. Use a recursive descent parser — do NOT use eval() or ast.literal_eval()\n"
            "8. Include type hints throughout and a brief docstring on each method\n"
            "9. Write 5 pytest tests covering: basic arithmetic, precedence, parentheses, "
            "unary minus, and error cases"
        ),
    },
    "astar": {
        "name": "A* Pathfinding",
        "expected": 6,
        "prompt": (
            "Implement A* pathfinding on a weighted 2D grid in Python. Requirements:\n\n"
            "1. Class AStarGrid with __init__(self, grid: List[List[int]]) where grid values "
            "represent movement cost (0 = impassable wall, positive int = cost to enter that cell)\n"
            "2. find_path(start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]] "
            "- return shortest path as list of (row, col) coordinates from start to end inclusive, "
            "or None if no path exists\n"
            "3. Support 4-directional movement (up, down, left, right) - no diagonals\n"
            "4. Use Manhattan distance as the heuristic\n"
            "5. Handle edge cases: start == end (return [start]), start or end is a wall (return None), "
            "start or end out of bounds (raise ValueError)\n"
            "6. The path must be optimal (minimum total cost)\n"
            "7. Use a min-heap (heapq) for the open set\n"
            "8. Include type hints throughout and a brief docstring on each method\n"
            "9. Write 6 pytest tests covering: simple path on uniform grid, path around obstacles, "
            "weighted grid (path prefers lower-cost cells), no path exists (fully blocked), "
            "start equals end, and invalid coordinates."
        ),
    },
    "lru_cache": {
        "name": "LRU Cache with TTL",
        "expected": 6,
        "prompt": (
            "Implement an LRU (Least Recently Used) cache in Python with time-based expiration. Requirements:\n\n"
            "1. Class TTLCache with __init__(self, capacity: int, default_ttl: float)\n"
            "2. get(key: str) -> Optional[Any] - return value if exists and not expired, else None. "
            "Accessing a key makes it most-recently-used.\n"
            "3. put(key: str, value: Any, ttl: Optional[float] = None) - insert/update. If at capacity, "
            "evict the least-recently-used non-expired item.\n"
            "4. delete(key: str) -> bool - remove key, return True if it existed\n"
            "5. size() -> int - return count of non-expired items\n"
            "6. All operations must be O(1) average time. Use a doubly-linked list + hash map internally "
            "- do NOT use OrderedDict.\n"
            "7. Use time.monotonic() for time tracking\n"
            "8. Include type hints and docstrings\n"
            "9. Write 6 pytest tests using unittest.mock.patch to mock time.monotonic."
        ),
    },
}


def run_inference(prompt):
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "temperature": TEMP,
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    comp = usage.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", ""),
        "tokens": comp,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(comp / elapsed if elapsed > 0 else 0, 1),
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "error": "no code blocks"}
    combined = "\n\n".join(b.strip() for b in blocks)
    for pat in [
        r'from \w+ import ExpressionEvaluator',
        r'from \w+ import AStarGrid',
        r'from \w+ import TTLCache',
    ]:
        combined = re.sub(pat, '', combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        return {
            "passed": len(re.findall(r' PASSED', output)),
            "failed": len(re.findall(r' FAILED', output)),
            "errors": len(re.findall(r' ERROR', output)),
        }
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 0, "error": str(e)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== TriAttention + Qwen3-8B Coding Benchmark ===")
    print(f"Config: KV budget=4096, divide_length=128, window=128, temp={TEMP}, {RUNS} runs")
    print()

    all_results = {}
    for bk, bench in BENCHMARKS.items():
        run_results = []
        for run in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
            result = run_inference(bench["prompt"])
            tf = f"{OUTPUT_DIR}/qwen3_8b_triattention_{bk}_run{run}_test.py"
            tr = extract_and_test(result["content"], tf)
            think = len(result.get("reasoning", "") or "")
            print(
                f" {result['tok_per_sec']:.0f} tok/s | "
                f"{tr['passed']}/{bench['expected']} pass | "
                f"{result['tokens']} tok | think:{think}c"
            )
            run_results.append({**result, **tr, "thinking_chars": think})

        best = max(r["passed"] for r in run_results)
        avg = sum(r["passed"] for r in run_results) / len(run_results)
        all_results[bk] = {
            "best": best,
            "avg": round(avg, 1),
            "expected": bench["expected"],
            "runs": run_results,
        }
        print(f"  -> Best: {best}/{bench['expected']}, Avg: {avg:.1f}/{bench['expected']}")
        print()

    total_best = sum(min(v["best"], v["expected"]) for v in all_results.values())
    total_exp = sum(v["expected"] for v in all_results.values())
    total_avg = sum(min(v["avg"], v["expected"]) for v in all_results.values())
    print(f"TOTAL: Best-of-{RUNS} = {total_best}/{total_exp}, Avg = {total_avg:.1f}/{total_exp}")

    with open(f"{OUTPUT_DIR}/triattention_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/triattention_results.json")


if __name__ == "__main__":
    main()
