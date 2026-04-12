#!/usr/bin/env python3
"""
M4 Max RotorQuant benchmark.

Tests planar3/iso3 KV cache configs on the S/A tier M4 Max models against
their existing f16/turbo4 baselines. See ROTORQUANT_HYPOTHESIS_M4MAX.md
for the pre-experiment hypothesis.

Runs the 3 standard coding prompts via johndpope/llama-cpp-turboquant's
feature/planarquant-kv-cache branch, Metal build.

Usage:
    python tools/m4max_rotorquant_bench.py [--only KEY[,KEY2...]] [--max-tokens N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Build from ~/git/TheTom/llama-cpp-planarquant worktree, johndpope/feature/planarquant-kv-cache
LLAMA_SERVER = str(Path.home() / "git/TheTom/llama-cpp-planarquant/build/bin/llama-server")
MODELS_DIR   = Path.home() / ".lmstudio" / "models"
PORT = 8765
BASE_URL = f"http://localhost:{PORT}"
OUTPUT_DIR = "experiments/rotorquant_m4max_bench"
MAX_TOKENS_DEFAULT = 16384

# All test runs use -b 2048 -ub 256 where 32K context is involved — rotorquant
# doesn't fix the Metal compute buffer scaling (H2 in the hypothesis).
MODELS = [
    # H1: Gemma 4 26B-A4B Q6_K f16 baseline is 60.3 tok/s, turbo4 is 46 tok/s.
    # planar3 K-only is the lowest-risk config — should be closest to f16.
    {
        "key": "gemma-4-26b-a4b-q6k-planar3-fp16-nothink",
        "name": "Gemma 4 26B-A4B Q6_K (planar3/f16)",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "context_length": 16384,
        "ctk": "planar3", "ctv": "f16",
        "ub": None,
        "no_think": True,
        "hypothesis": "H1 low-risk K-only: 56-60 tok/s projected (≤7% below f16 baseline)",
    },
    # H1 primary: symmetric planar3/planar3. Moderate risk — Metal V-dequant fix
    # is listed as TODO in rotorquant CLAUDE.md, may produce broken output.
    {
        "key": "gemma-4-26b-a4b-q6k-planar3-planar3-nothink",
        "name": "Gemma 4 26B-A4B Q6_K (planar3/planar3)",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "context_length": 16384,
        "ctk": "planar3", "ctv": "planar3",
        "ub": None,
        "no_think": True,
        "hypothesis": "H1 primary symmetric: 52-58 tok/s projected vs f16=60, turbo4=46",
    },
    # H1 alternate symmetric: iso3/iso3 (quaternion 4D block rotation).
    # Fallback if planar3/planar3 produces broken output.
    {
        "key": "gemma-4-26b-a4b-q6k-iso3-iso3-nothink",
        "name": "Gemma 4 26B-A4B Q6_K (iso3/iso3)",
        "path": MODELS_DIR / "lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "context_length": 16384,
        "ctk": "iso3", "ctv": "iso3",
        "ub": None,
        "no_think": True,
        "hypothesis": "H1 fallback: 52-58 tok/s, same band as planar3/planar3",
    },
    # H3: Gemma 4 31B-IT — the model where KV is genuinely the bottleneck.
    # turbo4 baseline is 11.8 tok/s at 32K with -ub 256 (mandatory workaround).
    # Highest-value test if the Metal backend cooperates.
    {
        "key": "gemma-4-31b-q4km-planar3-planar3-nothink",
        "name": "Gemma 4 31B-IT Q4_K_M (planar3/planar3)",
        "path": MODELS_DIR / "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "planar3", "ctv": "planar3",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H3 highest-value: 13-15 tok/s projected vs turbo4=11.8 (if Metal works)",
    },
    # K-only fallback on Gemma 31B — planar3/f16 barely fits the 28.6 GB
    # Metal budget (18 weights + 1.5 planar3 K + 7 f16 V + 2 compute = 28.5 GB).
    # Added after the symmetric configs on Gemma 4 hit a runaway-generation bug.
    {
        "key": "gemma-4-31b-q4km-planar3-fp16-nothink",
        "name": "Gemma 4 31B-IT Q4_K_M (planar3/f16 K-only)",
        "path": MODELS_DIR / "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "planar3", "ctv": "f16",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H3 K-only: could match turbo4 capacity while avoiding the symmetric runaway-gen bug",
    },
    # H3 alternate: iso3/iso3 on Gemma 31B.
    {
        "key": "gemma-4-31b-q4km-iso3-iso3-nothink",
        "name": "Gemma 4 31B-IT Q4_K_M (iso3/iso3)",
        "path": MODELS_DIR / "unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "iso3", "ctv": "iso3",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H3 alternate: same band as planar3/planar3 if iso is faster at same compression",
    },
    # H5: Qwen 27B Opus-Distilled — was predicted as lowest-interest sanity check,
    # turned out to be our ONLY viable test path (planarquant fork's base llama.cpp
    # is too old for gemma4 architecture). Runs below swept through configs.
    {
        "key": "qwen3.5-27b-opus-distill-q4km-planar3-fp16-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/f16)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "planar3", "ctv": "f16",
        "ub": None,
        "no_think": True,
        "hypothesis": "H5 K-only: within ±5% of f16 baseline (13 tok/s)",
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-planar3-planar3-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "planar3", "ctv": "planar3",
        "ub": None,
        "no_think": True,
        "hypothesis": "Full symmetric: both K and V compressed. Tests the Metal V-dequant port status.",
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-iso3-iso3-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (iso3/iso3)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 32768,
        "ctk": "iso3", "ctv": "iso3",
        "ub": None,
        "no_think": True,
        "hypothesis": "Quaternion 4D symmetric: alternate rotation block size vs planar3. Same Metal V-dequant risk.",
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-f16-f16-65k-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (f16/f16 @ 65K baseline)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 65536,
        "ctk": "f16", "ctv": "f16",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H2 baseline: does f16 KV at 65K fit at all? 9.6 GB KV + 16.5 weights + 8 compute = 34 GB likely OOM",
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-planar3-planar3-65k-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3 @ 65K)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 65536,
        "ctk": "planar3", "ctv": "planar3",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H2 probe: can we reach 64K context on Qwen 27B with rotorquant where f16 OOMs?",
    },
    {
        "key": "qwen3.5-27b-opus-distill-q4km-planar3-planar3-256k-nothink",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3 @ 256K)",
        "path": MODELS_DIR / "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "context_length": 262144,
        "ctk": "planar3", "ctv": "planar3",
        "ub": 256,
        "no_think": True,
        "hypothesis": "H2 real unlock: f16 OOMs at 256K (projected 32 GB), planar3/planar3 fits (projected 27 GB). New context tier.",
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


def start_server(model_path, ctx, ctk, ctv, ub, no_think):
    cmd = [
        LLAMA_SERVER,
        "-m", str(model_path),
        "--port", str(PORT),
        "-c", str(ctx),
        "-ngl", "999",
        "-fa", "on",
        "-ctk", ctk, "-ctv", ctv,
        "-np", "1",
        "--jinja",
    ]
    if ub is not None:
        cmd += ["-b", "2048", "-ub", str(ub)]
    if no_think:
        cmd += ["--reasoning-budget", "0"]
    print(f"  Starting: llama-server -m {model_path.name} -c {ctx} "
          f"-ctk {ctk} -ctv {ctv}"
          f"{' -ub ' + str(ub) if ub else ''} "
          f"--reasoning-budget {'0' if no_think else 'unrestricted'}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def wait_for_server(proc, timeout=600):
    """Wait for server readiness. Also watch stderr for rotorquant-specific
    crashes (which is H6 in the hypothesis and the whole point of this test
    being able to fail gracefully)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check if server died
        if proc.poll() is not None:
            print(f"\n  SERVER EXITED early with code {proc.returncode}")
            return False
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
        mc["path"], mc["context_length"],
        ctk=mc["ctk"], ctv=mc["ctv"],
        ub=mc.get("ub"),
        no_think=mc["no_think"],
    )
    stderr_tail = []
    print("  Waiting for server...", end="", flush=True)
    if not wait_for_server(proc):
        # Capture stderr tail for failure diagnostics
        try:
            if proc.stderr:
                stderr_bytes = proc.stderr.read()
                if stderr_bytes:
                    stderr_tail = stderr_bytes.decode("utf-8", errors="replace").splitlines()[-20:]
        except Exception:
            pass
        print(" FAILED TO LAUNCH")
        if stderr_tail:
            print("  --- stderr tail ---")
            for line in stderr_tail:
                print(f"    {line}")
        stop_server(proc)
        summary.append({
            "model": mc["name"],
            "key": mc["key"],
            "status": "launch_failed",
            "stderr_tail": stderr_tail,
        })
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
                f.write(f"**Hypothesis:** {mc.get('hypothesis', '')}\n\n")
                if r["reasoning"]:
                    f.write(f"## Thinking ({len(r['reasoning'])} chars)\n\n")
                    f.write(f"```\n{r['reasoning']}\n```\n\n")
                f.write(f"## Output\n\n{r['content']}\n")

            json_path = f"{OUTPUT_DIR}/{mc['key']}_{bk}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"model": mc["name"], "key": mc["key"], "benchmark": bk,
                           "ctk": mc["ctk"], "ctv": mc["ctv"], "ub": mc.get("ub"),
                           "context_length": mc["context_length"], **r},
                          f, indent=2, ensure_ascii=False)

            summary.append({
                "model": mc["name"],
                "key": mc["key"],
                "benchmark": bk,
                "ctk": mc["ctk"], "ctv": mc["ctv"],
                "tokens": comp,
                "tok_per_sec": tps,
                "elapsed_s": r["elapsed_s"],
                "thinking_chars": think_chars,
                "finish_reason": r["finish_reason"],
            })
        except Exception as e:
            print(f" ERROR: {e}")
            summary.append({
                "model": mc["name"],
                "key": mc["key"],
                "benchmark": bk,
                "status": "request_failed",
                "error": str(e),
            })

    print("  Stopping server...", end="", flush=True)
    stop_server(proc)
    print(" done")
    time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Comma-separated model keys (default: all)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    args = p.parse_args()

    if not Path(LLAMA_SERVER).exists():
        print(f"ERROR: llama-server not built at {LLAMA_SERVER}")
        print("Build with:")
        print("  cd ~/git/TheTom/llama-cpp-planarquant")
        print("  cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \\")
        print("    -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release")
        print("  cmake --build build --target llama-server -j 8")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    keys = set(args.only.split(",")) if args.only else None
    selected = [m for m in MODELS if (keys is None or m["key"] in keys)]
    print(f"Running {len(selected)} rotorquant config(s) at max_tokens={args.max_tokens}")

    for i, mc in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {mc['name']}  ({mc['key']})")
        print(f"  Hypothesis: {mc.get('hypothesis', '')}")
        run_one_model(mc, args.max_tokens, summary)

    out = f"{OUTPUT_DIR}/rotorquant_generation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Generation summary: {out}")
    print(f"Score with: python tools/score_combined.py {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
