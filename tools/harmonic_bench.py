#!/usr/bin/env python3
"""
TurboQuant turbo4 benchmark with thinking DISABLED (-rea off).

turbo4 is recommended by turboquant_plus for "quality-critical applications
(code, reasoning, instruction-following)" due to lower PPL impact (+0.23%
vs turbo3's +1.06%). Combined with thinking off, this tests the best-quality
KV compression without reasoning loop risk.

Also validates the turboquant_plus finding that Q4_K_M weights + symmetric
turbo is the risky combination — Q6_K models should outperform Q4_K_M.
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
OUTPUT_DIR = "experiments/harmonic_bench"

MODELS = [
    {
        "key": "harmonic27b-q8-turbo4-nothink",
        "name": "Harmonic 27B Q8_0",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q8_0.gguf",
        "native_ctx": 262144,
        "ctk": "turbo4", "ctv": "turbo4",
        "context_length": 32768,
        "label": "turbo4/turbo4 -rea off",
    },
    {
        "key": "harmonic27b-q4km-turbo4-nothink",
        "name": "Harmonic 27B Q4_K_M",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q4_K_M.gguf",
        "native_ctx": 262144,
        "ctk": "turbo4", "ctv": "turbo4",
        "context_length": 32768,
        "label": "turbo4/turbo4 -rea off",
    },
]

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator",
        "expected_tests": 5,
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
}

THROUGHPUT_PROMPT = (
    "Write a Python function that takes a list of integers and returns the longest "
    "increasing subsequence. Include type hints, handle edge cases, and add a brief "
    "docstring. Then write 3 unit tests using pytest."
)

MAX_TOKENS = 16384  # 16K generation budget — sufficient with reasoning-budget cap


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
    print(f"  Config: -ctk {model_cfg['ctk']} -ctv {model_cfg['ctv']}"
          f" -c {model_cfg['context_length']} -rea off")
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


def run_inference(prompt, max_tokens=MAX_TOKENS):
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
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
        "usage": usage,
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
        "elapsed_s": round(elapsed, 2),
        "tokens": comp_tokens,
        "tok_per_sec": round(comp_tokens / elapsed if elapsed > 0 else 0, 1),
    }


def run_streaming(prompt, max_tokens=2048):
    start = time.perf_counter()
    first_token_at = None
    thinking_chunks = 0
    content_chunks = 0
    server_usage = None

    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True, timeout=600,
    )
    resp.raise_for_status()

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "usage" in chunk and chunk["usage"]:
            server_usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("reasoning_content"):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            thinking_chunks += 1
        if delta.get("content"):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            content_chunks += 1

    end = time.perf_counter()
    ttft = (first_token_at - start) if first_token_at else None
    gen_time = (end - first_token_at) if first_token_at else (end - start)
    token_count = thinking_chunks + content_chunks
    if server_usage and server_usage.get("completion_tokens"):
        token_count = server_usage["completion_tokens"]
    tps = token_count / gen_time if gen_time > 0 else 0

    return {
        "tokens_per_second": round(tps, 2),
        "token_count": token_count,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(end - start, 3),
        "thinking_tokens": thinking_chunks,
        "content_tokens": content_chunks,
    }


def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "error": "no code blocks found"}

    combined = "\n\n".join(b.strip() for b in blocks)
    for pattern in [
        r'from \w+ import ExpressionEvaluator',
        r'from \w+ import AStarGrid',
        r'from \w+ import TTLCache',
    ]:
        combined = re.sub(pattern, '', combined)

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
        return {"passed": passed, "failed": failed, "errors": errors, "total": passed + failed + errors, "output": output}
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
        print(f"MODEL: {model_name} [{model['label']}]")
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
        print(f"  VRAM usage: {vram_mb} MB" if vram_mb else "  VRAM: N/A")

        model_result = {
            "model": model_name,
            "model_key": model_key,
            "kv_config": model["label"],
            "context_length": model["context_length"],
            "reasoning": "off",
            "native_ctx": model["native_ctx"],
            "vram_mb": vram_mb,
            "benchmarks": {},
        }

        # Throughput
        print(f"\n  [Throughput] Streaming...", end="", flush=True)
        try:
            tp = run_streaming(THROUGHPUT_PROMPT, max_tokens=2048)
            print(f" {tp['tokens_per_second']:.1f} tok/s | {tp['token_count']} tok | TTFT {tp['ttft_s']}s")
            model_result["throughput"] = tp
        except Exception as e:
            print(f" ERROR: {e}")
            model_result["throughput"] = {"error": str(e)}

        # Coding benchmarks
        for bench_key, bench in BENCHMARKS.items():
            print(f"\n  [{bench['name']}] Generating...", end="", flush=True)
            try:
                result = run_inference(bench["prompt"])
                print(
                    f" {result['tok_per_sec']:.1f} tok/s | "
                    f"{result['tokens']} tok | {result['elapsed_s']:.1f}s | "
                    f"finish={result['finish_reason']}"
                )

                safe = f"{model_key}_{bench_key}"
                with open(f"{OUTPUT_DIR}/{safe}.json", "w", encoding="utf-8") as f:
                    json.dump({"model": model_name, "benchmark": bench_key, **result}, f, indent=2, ensure_ascii=False)

                test_file = f"{OUTPUT_DIR}/{safe}_test.py"
                test_result = extract_and_test(result["content"], test_file)
                print(
                    f"  Tests: {test_result['passed']} passed, "
                    f"{test_result['failed']} failed, {test_result['errors']} errors "
                    f"(of {bench['expected_tests']} expected)"
                )

                model_result["benchmarks"][bench_key] = {
                    "tok_per_sec": result["tok_per_sec"],
                    "tokens": result["tokens"],
                    "elapsed_s": result["elapsed_s"],
                    "finish_reason": result["finish_reason"],
                    "thinking_chars": len(result["reasoning"]),
                    "content_chars": len(result["content"]),
                    "tests_passed": test_result["passed"],
                    "tests_failed": test_result["failed"],
                    "tests_errors": test_result["errors"],
                    "tests_expected": bench["expected_tests"],
                }
            except Exception as e:
                print(f" ERROR: {e}")
                model_result["benchmarks"][bench_key] = {"error": str(e)}

        all_results.append(model_result)
        print(f"\n  Stopping server...", end="", flush=True)
        stop_server(proc)
        print(" done")
        time.sleep(3)

    # Save results
    with open(f"{OUTPUT_DIR}/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 130}")
    print("OPTIMAL TURBOQUANT BENCHMARK SUMMARY")
    print(f"{'=' * 130}")
    print(
        f"{'Model':35s} | {'KV Config':14s} | {'VRAM':>6s} | {'Tok/s':>6s} | {'TTFT':>6s} | "
        f"{'ExprEval':>8s} | {'A*Path':>8s} | {'LRUCache':>8s} | {'Total':>10s}"
    )
    print("-" * 130)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:35s} | ERROR: {r['error']}")
            continue
        vram = f"{r['vram_mb']}MB" if r.get("vram_mb") else "?"
        tps = f"{r['throughput']['tokens_per_second']:.1f}" if "tokens_per_second" in r.get("throughput", {}) else "?"
        ttft = f"{r['throughput']['ttft_s']:.2f}s" if r.get("throughput", {}).get("ttft_s") else "?"

        bench_strs = []
        total_passed = 0
        total_expected = 0
        for bk in ["expression_evaluator", "astar", "lru_cache"]:
            b = r["benchmarks"].get(bk, {})
            if "error" in b:
                bench_strs.append("ERR")
            else:
                p = b.get("tests_passed", 0)
                e = b.get("tests_expected", 0)
                total_passed += p
                total_expected += e
                bench_strs.append(f"{p}/{e}")

        total_str = f"{total_passed}/{total_expected}" if total_expected > 0 else "?"
        pct = f"({total_passed/total_expected*100:.0f}%)" if total_expected > 0 else ""

        print(
            f"{r['model']:35s} | {r['kv_config']:14s} | {vram:>6s} | {tps:>6s} | {ttft:>6s} | "
            f"{bench_strs[0]:>8s} | {bench_strs[1]:>8s} | {bench_strs[2]:>8s} | "
            f"{total_str:>5s} {pct}"
        )
    print(f"{'=' * 130}")
    print(f"\nResults saved to {OUTPUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
