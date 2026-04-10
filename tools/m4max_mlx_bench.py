#!/usr/bin/env python3
"""
M4 Max MLX benchmark — same 3 coding prompts via mlx_lm.server.

Compares MLX-format quants against the llama.cpp GGUF runs from m4max_bench.py.
The MLX framework uses Apple's Metal Performance Shaders directly and
sometimes outperforms llama.cpp on Apple Silicon by 10-30% (see footnote in
HARDWARE_SPECS.md). This script tests whether that holds for our coding
benchmark workload.

Usage:
    python tools/m4max_mlx_bench.py [--only model_key] [--max-tokens N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

PORT = 8766                          # different port from llama.cpp bench (8765)
BASE_URL = f"http://localhost:{PORT}"
OUTPUT_DIR = "m4max_bench"
MAX_TOKENS_DEFAULT = 16384

# mlx-lm 0.31+ requires Python 3.10+ and the matching mlx 0.31+ wheel.
# The system Python 3.9 only has wheels up to mlx 0.29.3 which doesn't
# support qwen3_5 / gemma4 architectures. We use a brew Python 3.14 venv.
MLX_PYTHON = "/tmp/mlx-venv/bin/python3"

# MLX models live in HuggingFace cache. We pass the repo ID directly to
# mlx_lm.server which handles the cache resolution.
MODELS = [
    {
        "key": "mlx-qwen3.5-9b-4bit-nothink",
        "name": "Qwen 3.5 9B 4bit MLX",
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "no_think": True,
    },
    {
        "key": "mlx-nemotron-3-nano-4b-6bit-nothink",
        "name": "Nemotron 3 Nano 4B 6bit MLX",
        "model": "mlx-community/NVIDIA-Nemotron-3-Nano-4B-6bit",
        # Nemotron's chat template doesn't honor enable_thinking; leave default
        "no_think": False,
    },
    {
        "key": "mlx-gemma-4-26b-a4b-6bit-nothink",
        "name": "Gemma 4 26B-A4B 6bit MLX",
        "model": "mlx-community/gemma-4-26b-a4b-it-6bit",
        "no_think": False,   # Gemma 4 isn't a thinking model
    },
    {
        "key": "mlx-gemma-4-31b-4bit-nothink",
        "name": "Gemma 4 31B 4bit MLX",
        "model": "mlx-community/gemma-4-31b-it-4bit",
        "no_think": False,
    },
    {
        "key": "mlx-qwen3.5-27b-opus-distill-4bit-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled 4bit MLX",
        "model": "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        "no_think": True,
    },
]

BENCHMARKS = {
    "expression_evaluator": (
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
    "astar": (
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
    "lru_cache": (
        "Implement an LRU cache with TTL in Python. Requirements:\n\n"
        "1. Class TTLCache with __init__(capacity, default_ttl)\n"
        "2. get(key) -> Optional[Any], put(key, value, ttl=None), delete(key) -> bool, size() -> int\n"
        "3. O(1) average time. Doubly-linked list + hash map, no OrderedDict\n"
        "4. time.monotonic() for time tracking, lazy cleanup on access\n"
        "5. Include type hints and docstrings\n"
        "6. Write 6 pytest tests using unittest.mock.patch on time.monotonic"
    ),
}


def start_server(model: str, no_think: bool):
    cmd = [
        MLX_PYTHON, "-m", "mlx_lm", "server",
        "--model", model,
        "--port", str(PORT),
        "--temp", "0",
        "--max-tokens", str(MAX_TOKENS_DEFAULT),
    ]
    if no_think:
        cmd += ["--chat-template-args", '{"enable_thinking": false}']
    print(f"  Starting: mlx_lm server --model {model} (no_think={no_think})")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(timeout=1200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/v1/models", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def generate(model: str, prompt: str, max_tokens: int):
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=1800,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    # mlx_lm.server uses "reasoning" while llama-server uses "reasoning_content"
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "usage": data.get("usage", {}),
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
        "elapsed_s": round(elapsed, 2),
    }


def run_one_model(mc: dict, max_tokens: int, summary: list):
    proc = start_server(mc["model"], mc["no_think"])
    print("  Waiting for server...", end="", flush=True)
    if not wait_for_server():
        print(" TIMEOUT")
        stop_server(proc)
        return
    print(" ready")

    for bk, prompt in BENCHMARKS.items():
        print(f"  [{bk}] generating...", end="", flush=True)
        try:
            r = generate(mc["model"], prompt, max_tokens)
            comp = r["usage"].get("completion_tokens", 0)
            tps = round(comp / r["elapsed_s"], 1) if r["elapsed_s"] > 0 else 0
            think_chars = len(r["reasoning"] or "")
            print(
                f" {tps} tok/s | {comp} tok | think={think_chars}c | "
                f"finish={r['finish_reason']} | {r['elapsed_s']}s"
            )
            if r["finish_reason"] == "length":
                print("    WARNING: hit max_tokens")

            md_path = f"{OUTPUT_DIR}/{mc['key']}_{bk}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {mc['name']} — {bk}\n\n")
                if r["reasoning"]:
                    f.write(f"## Thinking ({len(r['reasoning'])} chars)\n\n")
                    f.write(f"```\n{r['reasoning']}\n```\n\n")
                f.write(f"## Output\n\n{r['content']}\n")

            json_path = f"{OUTPUT_DIR}/{mc['key']}_{bk}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"model": mc["name"], "key": mc["key"], "benchmark": bk, **r},
                          f, indent=2, ensure_ascii=False)

            summary.append({
                "model": mc["name"],
                "key": mc["key"],
                "benchmark": bk,
                "tokens": comp,
                "tok_per_sec": tps,
                "elapsed_s": r["elapsed_s"],
                "thinking_chars": think_chars,
                "finish_reason": r["finish_reason"],
            })
        except Exception as e:
            print(f" ERROR: {e}")

    print("  Stopping server...", end="", flush=True)
    stop_server(proc)
    print(" done")
    time.sleep(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Comma-separated model keys (default: all)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    keys = set(args.only.split(",")) if args.only else None
    selected = [m for m in MODELS if (keys is None or m["key"] in keys)]
    print(f"Running {len(selected)} MLX model(s) at max_tokens={args.max_tokens}")

    for i, mc in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {mc['name']}  ({mc['key']})")
        run_one_model(mc, args.max_tokens, summary)

    out = f"{OUTPUT_DIR}/mlx_generation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {out}")


if __name__ == "__main__":
    main()
