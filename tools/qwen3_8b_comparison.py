#!/usr/bin/env python3
"""
Three-way comparison on Qwen3-8B:
1. llama-server + f16 KV (baseline)
2. llama-server + turbo4 KV (TurboQuant)
3. vLLM + TriAttention (already done — 14/17 best-of-3)

Runs benchmarks at temp 0.3, 3 runs each, thinking off.
"""

import json
import os
import re
import subprocess
import sys
import time
import requests

LLAMA_SERVER = "T:/git/TheTom/llama-cpp-turboquant/build/bin/Release/llama-server.exe"
MODEL_PATH = os.path.expanduser("~/.lmstudio/models/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
PORT = 8080
TEMP = 0.3
RUNS = 3
MAX_TOKENS = 16384
OUTPUT_DIR = "experiments/qwen3_8b_comparison"

CONFIGS = [
    {"key": "f16", "ctk": "f16", "ctv": "f16", "label": "f16 KV (baseline)"},
    {"key": "turbo4", "ctk": "turbo4", "ctv": "turbo4", "label": "turbo4 KV (TurboQuant)"},
]

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator", "expected": 5,
        "prompt": (
            "Build a mathematical expression evaluator in Python. Requirements:\n"
            "1. Support +, -, *, / with correct operator precedence\n"
            "2. Support parentheses for grouping\n"
            "3. Support unary minus (e.g., '-3', '-(2+1)')\n"
            "4. Support floating point numbers (e.g., '3.14')\n"
            "5. Raise ValueError for: mismatched parentheses, division by zero, invalid tokens, empty expressions\n"
            "6. Implement as class ExpressionEvaluator with evaluate(expr: str) -> float\n"
            "7. Use recursive descent parser — no eval() or ast.literal_eval()\n"
            "8. Include type hints and docstrings\n"
            "9. Write 5 pytest tests"
        ),
    },
    "astar": {
        "name": "A* Pathfinding", "expected": 6,
        "prompt": (
            "Implement A* pathfinding on a weighted 2D grid in Python. Requirements:\n\n"
            "1. Class AStarGrid with __init__(self, grid: List[List[int]]) where grid values "
            "represent movement cost (0 = impassable wall, positive int = cost to enter that cell)\n"
            "2. find_path(start, end) -> Optional[List[Tuple[int, int]]] — shortest path or None\n"
            "3. 4-directional movement, Manhattan heuristic, heapq open set\n"
            "4. Handle: start==end, walls, out-of-bounds (ValueError)\n"
            "5. Path must be optimal (minimum total cost)\n"
            "6. Include type hints and docstrings\n"
            "7. Write 6 pytest tests"
        ),
    },
    "lru_cache": {
        "name": "LRU Cache with TTL", "expected": 6,
        "prompt": (
            "Implement an LRU cache with TTL in Python. Requirements:\n\n"
            "1. Class TTLCache with __init__(capacity, default_ttl)\n"
            "2. get(key) -> Optional[Any], put(key, value, ttl=None), delete(key) -> bool, size() -> int\n"
            "3. O(1) average time. Doubly-linked list + hash map, no OrderedDict\n"
            "4. time.monotonic() for time tracking, lazy cleanup on access\n"
            "5. Include type hints and docstrings\n"
            "6. Write 6 pytest tests using unittest.mock.patch on time.monotonic"
        ),
    },
}


def start_server(ctk, ctv):
    cmd = [
        LLAMA_SERVER, "-m", MODEL_PATH,
        "--port", str(PORT), "-c", "32768", "-ngl", "99",
        "-fa", "on", "-ctk", ctk, "-ctv", ctv, "-np", "1", "-rea", "off",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def wait_for_server(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


def run_inference(prompt):
    t0 = time.perf_counter()
    resp = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json={
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": TEMP,
    }, timeout=600)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    comp = usage.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        "tokens": comp,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(comp / elapsed if elapsed > 0 else 0, 1),
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0}
    combined = "\n\n".join(b.strip() for b in blocks)
    for p in [r'from \w+ import ExpressionEvaluator', r'from \w+ import AStarGrid', r'from \w+ import TTLCache']:
        combined = re.sub(p, '', combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                           capture_output=True, text=True, timeout=30)
        out = r.stdout + r.stderr
        return {"passed": len(re.findall(r' PASSED', out)), "failed": len(re.findall(r' FAILED', out)),
                "errors": len(re.findall(r' ERROR', out))}
    except Exception:
        return {"passed": 0, "failed": 0, "errors": 0}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg['label']}")
        print(f"{'='*70}")

        proc = start_server(cfg["ctk"], cfg["ctv"])
        print("  Starting server...", end="", flush=True)
        if not wait_for_server():
            print(" TIMEOUT")
            stop_server(proc)
            continue
        print(" ready")

        config_results = {}
        for bk, bench in BENCHMARKS.items():
            runs = []
            for run in range(1, RUNS + 1):
                print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
                result = run_inference(bench["prompt"])
                tf = f"{OUTPUT_DIR}/{cfg['key']}_{bk}_run{run}_test.py"
                tr = extract_and_test(result["content"], tf)
                print(f" {result['tok_per_sec']:.0f} tok/s | {tr['passed']}/{bench['expected']} pass | {result['tokens']} tok")
                runs.append({**result, **tr})

            best = max(r["passed"] for r in runs)
            avg = sum(r["passed"] for r in runs) / len(runs)
            config_results[bk] = {"best": best, "avg": round(avg, 1), "expected": bench["expected"], "runs": runs}
            print(f"  -> Best: {best}/{bench['expected']}, Avg: {avg:.1f}/{bench['expected']}")

        total_best = sum(min(v["best"], v["expected"]) for v in config_results.values())
        total_avg = sum(min(v["avg"], v["expected"]) for v in config_results.values())
        total_exp = sum(v["expected"] for v in config_results.values())
        print(f"\n  TOTAL: Best-of-{RUNS} = {total_best}/{total_exp}, Avg = {total_avg:.1f}/{total_exp}")

        all_results[cfg["key"]] = config_results
        stop_server(proc)
        time.sleep(3)

    # Add TriAttention results from previous run
    tri_path = "triattention_bench/triattention_results.json"
    if os.path.exists(tri_path):
        with open(tri_path) as f:
            all_results["triattention"] = json.load(f)
        print(f"\n  Loaded TriAttention results from {tri_path}")

    with open(f"{OUTPUT_DIR}/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Comparison table
    print(f"\n{'='*90}")
    print("QWEN3-8B THREE-WAY COMPARISON (temp=0.3, best-of-3)")
    print(f"{'='*90}")
    print(f"{'Config':30s} | {'ExprEval':>10s} | {'A*':>10s} | {'LRU':>10s} | {'Total':>12s}")
    print("-" * 90)
    labels = {"f16": "llama.cpp f16 (baseline)", "turbo4": "llama.cpp turbo4", "triattention": "vLLM + TriAttention"}
    for key, label in labels.items():
        r = all_results.get(key, {})
        if not r:
            continue
        parts = []
        total_b = 0
        total_e = 0
        for bk in ["expression_evaluator", "astar", "lru_cache"]:
            b = r.get(bk, {})
            bp = min(b.get("best", 0), b.get("expected", 6))
            exp = b.get("expected", 6)
            total_b += bp
            total_e += exp
            parts.append(f"{bp}/{exp}")
        print(f"{label:30s} | {parts[0]:>10s} | {parts[1]:>10s} | {parts[2]:>10s} | {total_b}/{total_e}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
