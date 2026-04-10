#!/usr/bin/env python3
"""
M4 Max benchmark — uses llama-server directly (not LM Studio API).

Why direct llama-server: LM Studio's API silently ignores chat_template_kwargs,
reasoning_effort, and the /no_think prefix for Qwen-family presets, so we can't
get apples-to-apples comparison vs. the 5090 ranking which uses `-rea off`.
llama-server's --reasoning-budget 0 gives exact parity.

Runs the 3 standard coding prompts (Expression Evaluator, A* Pathfinding,
LRU Cache with TTL) at temp=0 against each configured model.

Usage:
    python tools/m4max_bench.py [--only model_key] [--max-tokens N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Two llama-server builds on this machine:
#   - Docker build (~/.docker/bin/inference/llama-server): older, no gemma4 support
#   - Turboquant fork (~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server):
#     freshly built from feature/turboquant-kv-cache, supports gemma4 + turbo3/4 KV
# We default to turboquant because it's newer AND supports the same f16 KV path,
# so it's a strict superset of the Docker build for our purposes.
LLAMA_SERVER_STD = "/Users/gisenberg/.docker/bin/inference/llama-server"
LLAMA_SERVER_TQ  = str(Path.home() / "git/TheTom/llama-cpp-turboquant/build/bin/llama-server")
LLAMA_SERVER     = LLAMA_SERVER_TQ
MODELS_DIR   = Path.home() / ".lmstudio" / "models"
PORT = 8765
BASE_URL = f"http://localhost:{PORT}"
OUTPUT_DIR = "m4max_bench"
MAX_TOKENS_DEFAULT = 16384

# Per-model config. context_length is picked to fit within ~27 GB Metal budget
# on a 36 GB M4 Max (default 75% iogpu.wired_limit). Dense models with big KV
# (Gemma 31B-IT, ~870 KB/token at f16) are capped at 16K so they don't spill.
MODELS = [
    {
        "key": "qwen3.5-9b-q4km-nothink",
        "name": "Qwen 3.5 9B Q4_K_M",
        "path": MODELS_DIR / "lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
        "context_length": 32768,
        "no_think": True,
    },
    {
        "key": "nemotron-3-nano-4b-q4km-nothink",
        "name": "Nemotron 3 Nano 4B Q4_K_M",
        "path": MODELS_DIR / "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
        "context_length": 32768,
        "no_think": True,
    },
    {
        "key": "gemma-4-26b-a4b-q6k-nothink",
        "name": "Gemma 4 26B-A4B Q6_K",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        # Q6_K weights = 22 GB, leaving only ~8 GB for KV+compute on the
        # M4 Max's ~30 GB Metal working set. 32K f16 KV blows the budget,
        # so cap at 16K (still > the 16384 max_tokens we generate).
        "context_length": 16384,
        "no_think": True,
    },
    {
        "key": "gemma-4-26b-a4b-q6k-turbo4-nothink",
        "name": "Gemma 4 26B-A4B Q6_K (turbo4 KV)",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        # Same 16K context as f16 baseline above — apples-to-apples turbo4
        # vs f16 KV speed comparison. Q6_K weights at 22 GB are the actual
        # bottleneck on 30 GB Metal budget; turbo4 at 32K still OOMs because
        # KV isn't the limiting factor here.
        "context_length": 16384,
        "ctk": "turbo4", "ctv": "turbo4",
        "no_think": True,
    },
    {
        "key": "gemma-4-26b-a4b-q4km-nothink",
        "name": "Gemma 4 26B-A4B Q4_K_M",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        # Q4_K_M is ~16 GB. Leaves ~14 GB for KV at 32K f16.
        "context_length": 32768,
        "no_think": True,
    },
    {
        "key": "gemma-4-31b-q4km-turbo4-nothink",
        "name": "Gemma 4 31B-IT Q4_K_M (turbo4 KV)",
        "path": MODELS_DIR / "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        # Dense 31B has ~870 KB/token KV at f16 — 16K f16 KV = 14 GB which
        # blows the 30 GB Metal budget when added to 18 GB weights. turbo4
        # KV is the only way to run this on 36 GB unified memory.
        "context_length": 16384,
        "ctk": "turbo4", "ctv": "turbo4",
        "no_think": True,
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 32768,
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


def start_server(model_path: Path, ctx: int, no_think: bool, ctk="f16", ctv="f16"):
    """Start llama-server as a subprocess. Returns the Popen object."""
    cmd = [
        LLAMA_SERVER,
        "-m", str(model_path),
        "--port", str(PORT),
        "-c", str(ctx),
        "-ngl", "999",          # full GPU offload (Metal)
        "-fa", "on",            # flash attention
        "-ctk", ctk, "-ctv", ctv,
        "-np", "1",             # single parallel slot for deterministic temp 0
        "--jinja",
    ]
    if no_think:
        cmd += ["--reasoning-budget", "0"]
    print(f"  Starting: llama-server -m {model_path.name} -c {ctx} "
          f"-ctk {ctk} -ctv {ctv} "
          f"--reasoning-budget {'0' if no_think else 'unrestricted'}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
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


def generate(prompt: str, max_tokens: int):
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "local",
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
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", ""),
        "usage": data.get("usage", {}),
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
        "elapsed_s": round(elapsed, 2),
    }


def run_one_model(mc: dict, max_tokens: int, summary: list):
    if not mc["path"].exists():
        print(f"  SKIP — {mc['path']} not found")
        return

    proc = start_server(
        mc["path"], mc["context_length"], mc["no_think"],
        ctk=mc.get("ctk", "f16"),
        ctv=mc.get("ctv", "f16"),
    )
    print("  Waiting for server...", end="", flush=True)
    if not wait_for_server():
        print(" TIMEOUT")
        stop_server(proc)
        return
    print(" ready")

    for bk, prompt in BENCHMARKS.items():
        print(f"  [{bk}] generating...", end="", flush=True)
        try:
            r = generate(prompt, max_tokens)
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
    time.sleep(2)  # let Metal release VRAM before next load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Comma-separated model keys to run (default: all)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    keys = set(args.only.split(",")) if args.only else None
    selected = [m for m in MODELS if (keys is None or m["key"] in keys)]
    print(f"Running {len(selected)} model(s) at max_tokens={args.max_tokens}")

    for i, mc in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {mc['name']}  ({mc['key']})")
        run_one_model(mc, args.max_tokens, summary)

    out = f"{OUTPUT_DIR}/generation_summary_nothink.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Generation summary: {out}")
    print(f"Score with: python tools/score_combined.py {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
