#!/usr/bin/env python3
"""
Expanded benchmark suite: original 3 benchmarks + 3 new variants, temp 0.3 with 3 runs.

New benchmarks:
- string_processor: Easy baseline (every model should pass)
- expr_eval_impl_only: Implementation-only (tests provided by harness)
- debug_fix: Find and fix 3 bugs in a BST (tests provided by harness)

Temperature 0.3 with 3 runs per benchmark to average out variance and reduce
sensitivity to FP accumulation order differences.
"""

import json
import os
import re
import subprocess
import sys
import time
import requests

LLAMA_SERVER = "T:/git/TheTom/llama-cpp-turboquant/build/bin/Release/llama-server.exe"
MODELS_DIR = os.path.expanduser("~/.lmstudio/models")
PORT = 8080
OUTPUT_DIR = "expanded_bench"
TEMPERATURE = 0.3
RUNS_PER_BENCHMARK = 3
MAX_TOKENS = 16384

MODELS = [
    {
        "key": "gemma26b-q6k",
        "name": "Gemma 4 26B-A4B Q6_K",
        "path": f"{MODELS_DIR}/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "gemma31b-q4km",
        "name": "Gemma 4 31B-IT Q4_K_M",
        "path": f"{MODELS_DIR}/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "qwopus27b-q6k",
        "name": "Qwopus 3.5 27B-v3 Q6_K",
        "path": f"{MODELS_DIR}/Jackrong/Qwopus3.5-27B-v3-GGUF/Qwopus3.5-27B-v3-Q6_K.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "qwen27b-opus-q4km",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M",
        "path": f"{MODELS_DIR}/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "harmonic27b-q8",
        "name": "Harmonic 27B Q8_0",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q8_0.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "harmonic27b-q4km",
        "name": "Harmonic 27B Q4_K_M",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
    {
        "key": "qwen35b-q4km",
        "name": "Qwen 3.5 35B-A3B Q4_K_M",
        "path": f"{MODELS_DIR}/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "context_length": 32768,
    },
]

# --- Provided test suites for implementation-only and debug benchmarks ---

EXPR_EVAL_PROVIDED_TESTS = '''
import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")
'''

DEBUG_FIX_PROVIDED_TESTS = '''
import pytest

@pytest.fixture
def bst():
    tree = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        tree.insert(v)
    return tree

def test_search_existing(bst):
    assert bst.search(3) is True
    assert bst.search(7) is True
    assert bst.search(1) is True

def test_search_missing(bst):
    assert bst.search(2) is False
    assert bst.search(9) is False

def test_inorder(bst):
    assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]

def test_insert_and_search():
    tree = BST()
    tree.insert(10)
    tree.insert(5)
    tree.insert(15)
    assert tree.search(10) is True
    assert tree.search(5) is True
    assert tree.search(15) is True
    assert tree.search(20) is False
    assert tree.inorder() == [5, 10, 15]
'''

BUGGY_BST = '''class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(self.root, value)

    def _insert(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(node.left, value)
        else:
            if node.left is None:
                node.right = Node(value)
            else:
                self._insert(node.right, value)

    def search(self, value):
        return self._search(self.root, value)

    def _search(self, node, value):
        if node is None:
            return False
        if value == node.value:
            return True
        if value < node.value:
            return self._search(node.left, value)
        return self._search(node.left, value)

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node.value)
        self._inorder(node.left, result)'''

# --- Benchmark definitions ---

BENCHMARKS = {
    # Original 3 (model writes implementation + tests)
    "expression_evaluator": {
        "name": "Expression Evaluator",
        "expected_tests": 5,
        "mode": "self_test",
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
        "expected_tests": 6,
        "mode": "self_test",
        "prompt": (
            "Implement A* pathfinding on a weighted 2D grid in Python. Requirements:\n\n"
            "1. Class AStarGrid with __init__(self, grid: List[List[int]]) where grid values "
            "represent movement cost (0 = impassable wall, positive int = cost to enter that cell)\n"
            "2. find_path(start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]] "
            "— return shortest path as list of (row, col) coordinates from start to end inclusive, "
            "or None if no path exists\n"
            "3. Support 4-directional movement (up, down, left, right) — no diagonals\n"
            "4. Use Manhattan distance as the heuristic\n"
            "5. Handle edge cases: start == end (return [start]), start or end is a wall (return None), "
            "start or end out of bounds (raise ValueError)\n"
            "6. The path must be optimal (minimum total cost)\n"
            "7. Use a min-heap (heapq) for the open set\n"
            "8. Include type hints throughout and a brief docstring on each method\n"
            "9. Write 6 pytest tests covering: simple path on uniform grid, path around obstacles, "
            "weighted grid (path prefers lower-cost cells), no path exists (fully blocked), "
            "start equals end, and invalid coordinates. Assert both path validity and optimality (total cost)."
        ),
    },
    "lru_cache": {
        "name": "LRU Cache with TTL",
        "expected_tests": 6,
        "mode": "self_test",
        "prompt": (
            "Implement an LRU (Least Recently Used) cache in Python with time-based expiration. Requirements:\n\n"
            "1. Class TTLCache with __init__(self, capacity: int, default_ttl: float) where capacity "
            "is max items and default_ttl is seconds until expiry\n"
            "2. get(key: str) -> Optional[Any] — return value if exists and not expired, else None. "
            "Accessing a key makes it most-recently-used.\n"
            "3. put(key: str, value: Any, ttl: Optional[float] = None) — insert/update. If at capacity, "
            "evict the least-recently-used non-expired item. If all items are expired, clear them all first. "
            "Custom ttl overrides default.\n"
            "4. delete(key: str) -> bool — remove key, return True if it existed\n"
            "5. size() -> int — return count of non-expired items (lazy cleanup: expired items removed on access)\n"
            "6. All operations must be O(1) average time. Use a doubly-linked list + hash map internally "
            "— do NOT use OrderedDict.\n"
            "7. Use time.monotonic() for time tracking\n"
            "8. Include type hints throughout and a brief docstring on each method\n"
            "9. Write 6 pytest tests covering: basic get/put, capacity eviction (LRU order), TTL expiry, "
            "custom per-key TTL, delete, and size with mixed expired/valid items. "
            "Use unittest.mock.patch to mock time.monotonic for deterministic time control in tests "
            "— do NOT use time.sleep."
        ),
    },
    # New: easy baseline
    "string_processor": {
        "name": "String Processor",
        "expected_tests": 5,
        "mode": "self_test",
        "prompt": (
            "Write a Python class called StringProcessor with the following methods:\n\n"
            "1. reverse_words(s: str) -> str — reverse the order of words in a string (not the characters). "
            "Multiple spaces between words should become a single space. Leading/trailing spaces removed.\n"
            "2. count_vowels(s: str) -> int — count vowels (a, e, i, o, u, case-insensitive) in the string\n"
            "3. is_palindrome(s: str) -> bool — check if the string is a palindrome, ignoring case, spaces, and punctuation\n"
            "4. caesar_cipher(s: str, shift: int) -> str — apply Caesar cipher with given shift. Only shift a-z and A-Z, "
            "leave other characters unchanged. Support negative shifts.\n"
            "5. most_common_word(s: str) -> Optional[str] — return the most frequently occurring word (case-insensitive). "
            "If tied, return the one that appears first. Return None for empty strings.\n\n"
            "Include type hints and a brief docstring on each method.\n"
            "Write 5 pytest tests covering each method."
        ),
    },
    # New: implementation-only (harness provides tests)
    "expr_eval_impl": {
        "name": "Expr Eval (impl only)",
        "expected_tests": 5,
        "mode": "provided_tests",
        "provided_tests": EXPR_EVAL_PROVIDED_TESTS,
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
            "8. Include type hints throughout and a brief docstring on each method\n\n"
            "Do NOT write tests — only the implementation."
        ),
    },
    # New: debug and fix
    "debug_fix": {
        "name": "Debug BST Fix",
        "expected_tests": 4,
        "mode": "provided_tests",
        "provided_tests": DEBUG_FIX_PROVIDED_TESTS,
        "prompt": (
            "The following Python code implements a binary search tree with insert, search, and in-order traversal. "
            "It has 3 bugs. Find and fix all bugs. Return ONLY the corrected code — do not explain the bugs.\n\n"
            f"```python\n{BUGGY_BST}\n```"
        ),
    },
}


def start_server(model_cfg):
    cmd = [
        LLAMA_SERVER,
        "-m", model_cfg["path"],
        "--port", str(PORT),
        "-c", str(model_cfg["context_length"]),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", model_cfg["ctk"],
        "-ctv", model_cfg["ctv"],
        "-np", "1",
        "-rea", "off",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return proc


def wait_for_server(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def run_inference(prompt, temperature=0, max_tokens=MAX_TOKENS):
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=1200,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", ""),
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
        "elapsed_s": round(elapsed, 2),
        "tokens": comp_tokens,
        "tok_per_sec": round(comp_tokens / elapsed if elapsed > 0 else 0, 1),
    }


def extract_and_test(content, test_file, provided_tests=None):
    """Extract code, optionally append provided tests, run pytest."""
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "error": "no code blocks found"}

    combined = "\n\n".join(b.strip() for b in blocks)
    for pattern in [
        r'from \w+ import ExpressionEvaluator',
        r'from \w+ import AStarGrid',
        r'from \w+ import TTLCache',
        r'from \w+ import StringProcessor',
        r'from \w+ import BST',
        r'from \w+ import Node',
    ]:
        combined = re.sub(pattern, '', combined)

    if provided_tests:
        combined = combined + "\n\n" + provided_tests.strip()

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        passed = len(re.findall(r' PASSED', output))
        failed = len(re.findall(r' FAILED', output))
        errors = len(re.findall(r' ERROR', output))
        return {"passed": passed, "failed": failed, "errors": errors,
                "total": passed + failed + errors, "output": output}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "error": "pytest timeout"}
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "error": str(e)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for model in MODELS:
        model_key = model["key"]
        model_name = model["name"]

        if not os.path.isfile(model["path"]):
            print(f"\nSKIP: {model_name} — not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        proc = start_server(model)
        print("  Starting server...", end="", flush=True)
        if not wait_for_server():
            print(" TIMEOUT")
            stop_server(proc)
            all_results.append({"model": model_name, "model_key": model_key, "error": "server start timeout"})
            continue
        print(" ready")

        time.sleep(2)
        vram_mb = get_vram_mb()
        print(f"  VRAM: {vram_mb} MB" if vram_mb else "  VRAM: N/A")

        model_result = {
            "model": model_name,
            "model_key": model_key,
            "vram_mb": vram_mb,
            "temperature": TEMPERATURE,
            "runs": RUNS_PER_BENCHMARK,
            "benchmarks": {},
        }

        for bench_key, bench in BENCHMARKS.items():
            bench_name = bench["name"]
            provided_tests = bench.get("provided_tests")
            run_results = []

            for run_num in range(1, RUNS_PER_BENCHMARK + 1):
                label = f"  [{bench_name}] Run {run_num}/{RUNS_PER_BENCHMARK}"
                print(f"{label}...", end="", flush=True)
                try:
                    result = run_inference(bench["prompt"], temperature=TEMPERATURE)
                    test_file = f"{OUTPUT_DIR}/{model_key}_{bench_key}_run{run_num}_test.py"
                    test_result = extract_and_test(result["content"], test_file, provided_tests)
                    print(
                        f" {result['tok_per_sec']:.0f} tok/s | "
                        f"{result['tokens']} tok | "
                        f"{test_result['passed']}/{bench['expected_tests']} pass | "
                        f"{result['elapsed_s']:.0f}s"
                    )
                    run_results.append({
                        "run": run_num,
                        "tok_per_sec": result["tok_per_sec"],
                        "tokens": result["tokens"],
                        "elapsed_s": result["elapsed_s"],
                        "finish_reason": result["finish_reason"],
                        "tests_passed": test_result["passed"],
                        "tests_failed": test_result["failed"],
                        "tests_errors": test_result["errors"],
                    })
                except Exception as e:
                    print(f" ERROR: {e}")
                    run_results.append({"run": run_num, "error": str(e)})

            # Aggregate runs
            valid = [r for r in run_results if "error" not in r]
            if valid:
                avg_passed = sum(r["tests_passed"] for r in valid) / len(valid)
                best_passed = max(r["tests_passed"] for r in valid)
                avg_tps = sum(r["tok_per_sec"] for r in valid) / len(valid)
                model_result["benchmarks"][bench_key] = {
                    "name": bench_name,
                    "expected": bench["expected_tests"],
                    "avg_passed": round(avg_passed, 1),
                    "best_passed": best_passed,
                    "avg_tok_per_sec": round(avg_tps, 1),
                    "runs": run_results,
                }
            else:
                model_result["benchmarks"][bench_key] = {"name": bench_name, "error": "all runs failed"}

        all_results.append(model_result)
        print(f"\n  Stopping server...", end="", flush=True)
        stop_server(proc)
        print(" done")
        time.sleep(3)

    # Save
    with open(f"{OUTPUT_DIR}/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    bench_keys = list(BENCHMARKS.keys())
    bench_short = ["ExprEval", "A*Path", "LRU", "StrProc", "ImplOnly", "Debug"]

    print(f"\n{'=' * 140}")
    print(f"EXPANDED BENCHMARK SUMMARY (temp={TEMPERATURE}, {RUNS_PER_BENCHMARK} runs, best-of-{RUNS_PER_BENCHMARK})")
    print(f"{'=' * 140}")
    header = f"{'Model':40s} | {'VRAM':>6s}"
    for s in bench_short:
        header += f" | {s:>8s}"
    header += f" | {'Total':>10s}"
    print(header)
    print("-" * 140)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:40s} | ERROR: {r['error']}")
            continue
        vram = f"{r['vram_mb']}MB" if r.get("vram_mb") else "?"
        row = f"{r['model']:40s} | {vram:>6s}"
        total_best = 0
        total_expected = 0
        for bk in bench_keys:
            b = r["benchmarks"].get(bk, {})
            if "error" in b:
                row += f" | {'ERR':>8s}"
            else:
                bp = b.get("best_passed", 0)
                exp = b.get("expected", 0)
                total_best += bp
                total_expected += exp
                row += f" | {f'{bp}/{exp}':>8s}"
        pct = f"({total_best/total_expected*100:.0f}%)" if total_expected > 0 else ""
        row += f" | {f'{total_best}/{total_expected}':>5s} {pct}"
        print(row)
    print(f"{'=' * 140}")
    print(f"\nResults saved to {OUTPUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
