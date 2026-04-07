#!/usr/bin/env python3
"""
TurboQuant Benchmark Runner

Runs the same coding benchmarks used for LM Studio evaluation, but against
llama-server from the TurboQuant fork. Tests different KV cache quantization
types (turbo2, turbo3, turbo4, q8_0, f16) to measure quality and throughput
impact.

Unlike LM Studio, llama-server loads a single model at startup with fixed
KV cache settings. This script manages the server lifecycle: start server ->
run benchmarks -> stop server -> repeat with next config.

Usage:
    # Run all benchmarks with all KV configs on all models
    python tools/turboquant_bench.py

    # Specific model and KV type
    python tools/turboquant_bench.py --models gemma-q6k --kv-types turbo3 f16

    # Throughput-only mode (skip coding benchmarks)
    python tools/turboquant_bench.py --throughput-only

    # Context scaling test (Gemma at increasing context sizes)
    python tools/turboquant_bench.py --context-scaling --models gemma-q6k
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Paths — adjust these to match your system
# ---------------------------------------------------------------------------

LLAMA_SERVER = os.environ.get(
    "LLAMA_SERVER",
    "T:/git/TheTom/llama-cpp-turboquant/build/bin/Release/llama-server.exe",
)

MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.expanduser("~/.lmstudio/models/lmstudio-community"),
)

MODEL_CONFIGS = {
    "gemma-q6k": {
        "path": f"{MODELS_DIR}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "name": "Gemma 4 26B-A4B Q6_K",
    },
    "gemma-q4km": {
        "path": f"{MODELS_DIR}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "name": "Gemma 4 26B-A4B Q4_K_M",
    },
    "qwen35b": {
        "path": f"{MODELS_DIR}/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "name": "Qwen 3.5 35B-A3B Q4_K_M",
    },
}

# KV cache configurations to test
KV_CONFIGS = {
    "f16":    {"ctk": "f16",    "ctv": "f16",    "label": "KV f16 (baseline)"},
    "q8_0":   {"ctk": "q8_0",  "ctv": "q8_0",   "label": "KV q8_0"},
    "turbo4": {"ctk": "turbo4", "ctv": "turbo4",  "label": "KV turbo4 (4-bit)"},
    "turbo3": {"ctk": "turbo3", "ctv": "turbo3",  "label": "KV turbo3 (3-bit)"},
    "turbo2": {"ctk": "turbo2", "ctv": "turbo2",  "label": "KV turbo2 (2-bit)"},
}

# Default port for llama-server (different from LM Studio's 1234)
DEFAULT_PORT = 8080

# Same prompts used in compare_outputs.py and hard_benchmarks.py
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


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(model_path, port, context_length, ctk, ctv, gpu_layers=99):
    """Start llama-server as a subprocess. Returns the Popen handle."""
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "--port", str(port),
        "-c", str(context_length),
        "-ngl", str(gpu_layers),
        "-fa",
        "-ctk", ctk,
        "-ctv", ctv,
        "--temp", "0",
    ]
    print(f"  Starting server: {' '.join(cmd[-8:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return proc


def wait_for_server(port, timeout=120):
    """Poll the health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            data = resp.json()
            if data.get("status") == "ok":
                return True
        except (requests.ConnectionError, requests.Timeout, Exception):
            pass
        time.sleep(1)
    return False


def stop_server(proc):
    """Terminate the llama-server process."""
    if proc is None:
        return
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Inference (reuses the same /v1/chat/completions endpoint as LM Studio)
# ---------------------------------------------------------------------------

def run_inference(port, prompt, max_tokens=16384, stream=True):
    """
    Send a chat completion request and measure performance.
    Works identically to the LM Studio benchmark — same endpoint, same format.
    """
    base_url = f"http://localhost:{port}"
    messages = [{"role": "user", "content": prompt}]

    if not stream:
        # Non-streaming (simpler, used for code quality benchmarks)
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=600,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return {
            "content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "usage": usage,
            "finish_reason": data["choices"][0].get("finish_reason", "?"),
            "elapsed_s": round(elapsed, 2),
            "tokens": usage.get("completion_tokens", 0),
            "tok_per_sec": round(
                usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0, 1
            ),
        }

    # Streaming (used for throughput measurement)
    start = time.perf_counter()
    first_any_token_at = None
    first_content_token_at = None
    thinking_chunks = 0
    content_chunks = 0
    output_text = []
    thinking_text = []
    server_usage = None

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=600,
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

        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            if first_any_token_at is None:
                first_any_token_at = time.perf_counter()
            thinking_chunks += 1
            thinking_text.append(reasoning)

        content = delta.get("content", "")
        if content:
            if first_any_token_at is None:
                first_any_token_at = time.perf_counter()
            if first_content_token_at is None:
                first_content_token_at = time.perf_counter()
            content_chunks += 1
            output_text.append(content)

    end = time.perf_counter()
    total_time = end - start
    total_chunks = thinking_chunks + content_chunks
    ttft = (first_any_token_at - start) if first_any_token_at else None
    gen_time = (end - first_any_token_at) if first_any_token_at else total_time

    token_count = total_chunks
    if server_usage and server_usage.get("completion_tokens"):
        token_count = server_usage["completion_tokens"]

    tps = token_count / gen_time if gen_time > 0 else 0

    return {
        "token_count": token_count,
        "thinking_tokens": thinking_chunks,
        "content_tokens": content_chunks,
        "total_time_s": round(total_time, 3),
        "time_to_first_token_s": round(ttft, 3) if ttft else None,
        "generation_time_s": round(gen_time, 3),
        "tokens_per_second": round(tps, 2),
        "server_usage": server_usage,
        "content": "".join(output_text),
        "reasoning": "".join(thinking_text),
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_code_benchmarks(port, model_name, kv_label, output_dir):
    """Run all 3 coding benchmarks and save results."""
    results = []
    for bench_key, bench in BENCHMARKS.items():
        print(f"\n    [{bench['name']}] Generating...", end="", flush=True)
        try:
            result = run_inference(port, bench["prompt"], max_tokens=16384, stream=False)
            print(
                f" {result['tok_per_sec']:.1f} tok/s | "
                f"{result['tokens']} tok | {result['elapsed_s']:.1f}s | "
                f"think: {len(result['reasoning'])}c | content: {len(result['content'])}c | "
                f"finish={result['finish_reason']}"
            )
            if result["finish_reason"] == "length":
                print("    WARNING: output truncated (hit max_tokens)")

            # Save output
            safe_name = f"{model_name}_{kv_label}_{bench_key}".replace(" ", "_")
            with open(f"{output_dir}/{safe_name}.md", "w", encoding="utf-8") as f:
                f.write(f"# {model_name} — {kv_label} — {bench['name']}\n\n")
                if result["reasoning"]:
                    f.write(f"## Thinking ({len(result['reasoning'])} chars)\n\n")
                    f.write(f"{result['reasoning']}\n\n")
                f.write(f"## Output\n\n{result['content']}\n")
            with open(f"{output_dir}/{safe_name}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_name,
                    "kv_config": kv_label,
                    "benchmark": bench_key,
                    **result,
                }, f, indent=2, ensure_ascii=False)

            results.append({
                "model": model_name,
                "kv_config": kv_label,
                "benchmark": bench["name"],
                "tok_per_sec": result["tok_per_sec"],
                "tokens": result["tokens"],
                "elapsed_s": result["elapsed_s"],
                "finish": result["finish_reason"],
                "thinking_chars": len(result["reasoning"]),
                "content_chars": len(result["content"]),
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "model": model_name,
                "kv_config": kv_label,
                "benchmark": bench["name"],
                "error": str(e),
            })
    return results


def run_throughput_benchmark(port, model_name, kv_label, max_tokens=2048):
    """Run streaming throughput benchmark (same as lmstudio_bench.py)."""
    print(f"    [Throughput] Generating...", end="", flush=True)
    try:
        result = run_inference(port, THROUGHPUT_PROMPT, max_tokens=max_tokens, stream=True)
        think_info = ""
        if result["thinking_tokens"] > 0:
            think_info = f" | Thinking: {result['thinking_tokens']} tok"
        print(
            f" {result['tokens_per_second']:.1f} tok/s | "
            f"{result['token_count']} tok "
            f"({result['content_tokens']} content){think_info} | "
            f"TTFT {result['time_to_first_token_s']}s | "
            f"Total {result['total_time_s']}s"
        )
        return {
            "model": model_name,
            "kv_config": kv_label,
            "benchmark": "throughput",
            "tok_per_sec": result["tokens_per_second"],
            "tokens": result["token_count"],
            "ttft_s": result["time_to_first_token_s"],
            "total_s": result["total_time_s"],
        }
    except Exception as e:
        print(f" ERROR: {e}")
        return {"model": model_name, "kv_config": kv_label, "benchmark": "throughput", "error": str(e)}


def run_context_scaling(port, model_name, kv_label, context_sizes, max_tokens=2048):
    """Measure throughput at increasing context sizes (same as context capacity test)."""
    results = []
    for ctx_size in context_sizes:
        print(f"    [Context {ctx_size//1024}K] ", end="", flush=True)
        # Note: for this test, the server must be restarted per context size.
        # This function assumes the server is already running at the right context size.
        # The caller handles restarting.
        try:
            result = run_inference(port, THROUGHPUT_PROMPT, max_tokens=max_tokens, stream=True)
            print(f"{result['tokens_per_second']:.1f} tok/s | {result['token_count']} tok")
            results.append({
                "model": model_name,
                "kv_config": kv_label,
                "context_size": ctx_size,
                "tok_per_sec": result["tokens_per_second"],
                "tokens": result["token_count"],
                "ttft_s": result["time_to_first_token_s"],
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "model": model_name, "kv_config": kv_label,
                "context_size": ctx_size, "error": str(e),
            })
    return results


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Print a comparison table of all results."""
    if not all_results:
        return

    print(f"\n{'=' * 110}")
    print("TURBOQUANT BENCHMARK RESULTS")
    print(f"{'=' * 110}")
    print(
        f"{'Model':30s} | {'KV Config':20s} | {'Benchmark':25s} | "
        f"{'Tok/s':>7s} | {'Tokens':>7s} | {'Time':>7s}"
    )
    print("-" * 110)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:30s} | {r['kv_config']:20s} | {r['benchmark']:25s} | ERROR: {r['error'][:30]}")
        else:
            tps = r.get("tok_per_sec", 0)
            tok = r.get("tokens", 0)
            elapsed = r.get("elapsed_s", r.get("total_s", 0))
            print(
                f"{r['model']:30s} | {r['kv_config']:20s} | {r['benchmark']:25s} | "
                f"{tps:>7.1f} | {tok:>7d} | {elapsed:>6.1f}s"
            )
    print(f"{'=' * 110}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs with TurboQuant KV cache quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help=f"Models to test (default: all). Choices: {list(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--kv-types", nargs="+", choices=list(KV_CONFIGS.keys()),
        default=["f16", "turbo3"],
        help=f"KV cache types to test (default: f16 turbo3). Choices: {list(KV_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--context-length", type=int, default=32768,
        help="Context length (default: 32768)",
    )
    parser.add_argument(
        "--throughput-only", action="store_true",
        help="Skip coding benchmarks, only run throughput test",
    )
    parser.add_argument(
        "--context-scaling", action="store_true",
        help="Run context scaling test (32K to 256K)",
    )
    parser.add_argument(
        "--output-dir", default="turboquant_results",
        help="Output directory (default: turboquant_results)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify llama-server binary exists
    if not os.path.isfile(LLAMA_SERVER):
        print(f"ERROR: llama-server not found at {LLAMA_SERVER}")
        print("Set LLAMA_SERVER env var or build the TurboQuant fork first.")
        sys.exit(1)

    all_results = []

    for model_key in args.models:
        model_cfg = MODEL_CONFIGS[model_key]
        model_path = model_cfg["path"]
        model_name = model_cfg["name"]

        if not os.path.isfile(model_path):
            print(f"\nSKIP: {model_name} — file not found at {model_path}")
            continue

        for kv_key in args.kv_types:
            kv_cfg = KV_CONFIGS[kv_key]
            kv_label = kv_cfg["label"]

            if args.context_scaling:
                # Context scaling: restart server per context size
                context_sizes = [32768, 65536, 112640, 196608, 262144]
                for ctx_size in context_sizes:
                    print(f"\n{'=' * 70}")
                    print(f"{model_name} | {kv_label} | Context {ctx_size // 1024}K")
                    print(f"{'=' * 70}")

                    proc = start_server(
                        model_path, args.port, ctx_size,
                        kv_cfg["ctk"], kv_cfg["ctv"],
                    )
                    print("  Waiting for server...", end="", flush=True)
                    if not wait_for_server(args.port, timeout=180):
                        print(" TIMEOUT — server failed to start")
                        stop_server(proc)
                        all_results.append({
                            "model": model_name, "kv_config": kv_label,
                            "context_size": ctx_size,
                            "benchmark": "context_scaling",
                            "error": "server start timeout",
                        })
                        continue
                    print(" ready")

                    result = run_throughput_benchmark(
                        args.port, model_name, kv_label, max_tokens=2048
                    )
                    result["context_size"] = ctx_size
                    result["benchmark"] = f"context_{ctx_size // 1024}K"
                    all_results.append(result)

                    stop_server(proc)
                    time.sleep(2)  # brief pause between server restarts
            else:
                # Normal benchmark: single context size
                print(f"\n{'=' * 70}")
                print(f"{model_name} | {kv_label}")
                print(f"{'=' * 70}")

                proc = start_server(
                    model_path, args.port, args.context_length,
                    kv_cfg["ctk"], kv_cfg["ctv"],
                )
                print("  Waiting for server...", end="", flush=True)
                if not wait_for_server(args.port, timeout=180):
                    print(" TIMEOUT — server failed to start")
                    stop_server(proc)
                    continue
                print(" ready")

                # Throughput benchmark
                result = run_throughput_benchmark(args.port, model_name, kv_label)
                all_results.append(result)

                # Code quality benchmarks
                if not args.throughput_only:
                    bench_results = run_code_benchmarks(
                        args.port, model_name, kv_label, args.output_dir
                    )
                    all_results.extend(bench_results)

                stop_server(proc)
                time.sleep(2)

    # Summary
    print_summary(all_results)

    # Save raw results
    outfile = f"{args.output_dir}/turboquant_results.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to {outfile}")


if __name__ == "__main__":
    main()
