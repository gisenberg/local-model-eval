#!/usr/bin/env python3
"""
RotorQuant 5090 benchmark: S/A tier models via johndpope's planarquant fork.

Tests two rotorquant configs per model:
  1. iso3 / iso3  (4D quaternion, 10.3x compression)
  2. planar3 / f16 (K-only Givens, 5.1x compression, ~0 PPL loss claimed)

Baselines are the published turbo4 numbers in MODEL_RANKINGS_5090.md,
re-measured April 10 from WSL2 Linux client. We do NOT re-measure turbo4
here — the rotorquant numbers will be compared directly to the committed
baseline.

Designed to run from WSL2 Linux client against the Windows johndpope
llama-server binary to avoid the ~2.1s urllib3-on-Windows TTFT bug.
Each benchmark is 3 runs at temp 0.3, best-of-3 scoring.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time

import requests

# Paths are WSL2-style (this script runs from inside WSL2)
LLAMA_SERVER_WIN_PATH = "/mnt/t/git/johndpope/llama-cpp-turboquant/build/bin/Release/llama-server.exe"
MODELS_DIR = "/mnt/c/Users/gisen/.lmstudio/models"
WIN_HOST_IP = "172.20.240.1"  # WSL2 NAT gateway → Windows host
PORT = 8090  # avoid clash with lingering servers
OUTPUT_DIR = "/mnt/t/git/local-model-eval/experiments/rotorquant_5090"
TEMP = 0.3
RUNS = 3
MAX_TOKENS = 16384
CTX = 32768

# Models with their thinking mode (based on MODEL_RANKINGS_5090.md)
#
# NOTE: Three S/A tier Gemma 4 models are excluded — the johndpope fork's base
# upstream (April 1, 2026) predates Gemma 4 architecture support. Loading any
# gemma4 GGUF crashes with `unknown model architecture: 'gemma4'`. The excluded
# models are Gemma 4 26B-A4B Q6_K, Gemma 4 26B-A4B Q4_K_M, and Gemma 4 31B-IT
# Q4_K_M — the last of which was the one model where rotorquant's 10.3x
# compression would have materially unlocked extra context. See
# ROTORQUANT_HYPOTHESIS_5090.md for the full scope-cut rationale.
MODELS = [
    {
        "key": "qwen27b-opus-q4km",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M",
        "path": f"{MODELS_DIR}/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "thinking_off": True,
        "baseline_turbo4": {"decode": 60.0, "best_of_3": "31/31"},
    },
    {
        "key": "qwopus27b-q6k",
        "name": "Qwopus 3.5 27B-v3 Q6_K",
        "path": f"{MODELS_DIR}/Jackrong/Qwopus3.5-27B-v3-GGUF/Qwopus3.5-27B-v3-Q6_K.gguf",
        "thinking_off": True,
        "baseline_turbo4": {"decode": 49.6, "best_of_3": "30/31"},
    },
    {
        "key": "harmonic27b-q4km",
        "name": "Harmonic 27B Q4_K_M (thinking on)",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q4_K_M.gguf",
        "thinking_off": False,  # Harmonic wins with thinking on
        "reasoning_budget": 16384,
        "baseline_turbo4": {"decode": 61.3, "best_of_3": "31/31"},
    },
]

ROTORQUANT_CONFIGS = [
    {"key": "iso3_iso3", "ctk": "iso3", "ctv": "iso3", "label": "iso3/iso3 (symmetric 10.3x)"},
    {"key": "planar3_f16", "ctk": "planar3", "ctv": "f16", "label": "planar3/f16 (K-only 5.1x)"},
]

# Standard 4-benchmark suite (matches other *_bench.py scripts)
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


def start_windows_server(model, rq_config):
    """Spawn the Windows llama-server.exe for this model + KV config.

    The Windows binary path is mounted at /mnt/t/... in WSL, but we need
    to invoke the Windows exe directly via wsl's shared host binary path.
    Simpler: use subprocess to launch `/mnt/t/.../llama-server.exe` from
    WSL — WSL2 will run it as a native Windows process through /init.
    """
    cmd = [
        LLAMA_SERVER_WIN_PATH,
        "-m", model["path"].replace("/mnt/c/", "C:/"),  # Windows path for the .exe
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "-c", str(CTX),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", rq_config["ctk"],
        "-ctv", rq_config["ctv"],
        "-np", "1",
    ]
    if model.get("thinking_off", True):
        cmd.extend(["-rea", "off"])
    else:
        cmd.extend(["-rea", "on"])
        if model.get("reasoning_budget"):
            cmd.extend(["--reasoning-budget", str(model["reasoning_budget"])])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{WIN_HOST_IP}:{PORT}/health", timeout=2)
            if r.json().get("status") == "ok":
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


def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def run_streaming(prompt, max_tokens=MAX_TOKENS):
    t_send = time.perf_counter()
    resp = requests.post(
        f"http://{WIN_HOST_IP}:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": TEMP,
            "stream": True,
        },
        stream=True,
        timeout=1200,
    )
    resp.raise_for_status()

    first_token_at = None
    last_token_at = None
    n_tokens = 0
    content = []
    reasoning = []
    server_usage = None

    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
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
        c = delta.get("content", "")
        r = delta.get("reasoning_content", "")
        if c or r:
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            n_tokens += 1
            if c:
                content.append(c)
            if r:
                reasoning.append(r)

    t_end = time.perf_counter()
    if first_token_at is None:
        return None

    ttft = first_token_at - t_send
    decode_time = (last_token_at - first_token_at) if last_token_at > first_token_at else None
    decode_tps = (n_tokens / decode_time) if decode_time and decode_time > 0 else None

    if server_usage and server_usage.get("completion_tokens"):
        n_tokens = server_usage["completion_tokens"]
        if decode_time and decode_time > 0:
            decode_tps = n_tokens / decode_time

    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "ttft_s": round(ttft, 4),
        "decode_tps": round(decode_tps, 1) if decode_tps else None,
        "tokens": n_tokens,
        "elapsed_s": round(t_end - t_send, 2),
    }


def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "error": "no code blocks"}
    combined = "\n\n".join(b.strip() for b in blocks)
    for cls in ['ExpressionEvaluator', 'AStarGrid', 'TTLCache', 'StringProcessor', 'BST', 'Node']:
        combined = re.sub(rf'from \w+ import {cls}\b.*\n', '', combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stdout + result.stderr
        return {
            "passed": len(re.findall(r' PASSED', out)),
            "failed": len(re.findall(r' FAILED', out)),
            "errors": len(re.findall(r' ERROR', out)),
        }
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 0, "error": str(e)}


def benchmark_one(model, rq_config):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model['name']}")
    print(f"CONFIG: {rq_config['label']}")
    print(f"{'=' * 70}")

    if not os.path.isfile(model["path"]):
        print(f"  SKIP: {model['path']} not found")
        return None

    proc = start_windows_server(model, rq_config)
    print("  Starting Windows server...", end="", flush=True)
    if not wait_for_server():
        print(" TIMEOUT")
        stop_server(proc)
        return {"model": model["name"], "config": rq_config["key"], "error": "server timeout"}
    print(" ready")

    time.sleep(2)
    vram = get_vram_mb()
    print(f"  VRAM: {vram} MB")

    result = {
        "model": model["name"],
        "model_key": model["key"],
        "config": rq_config["key"],
        "kv": f"{rq_config['ctk']}/{rq_config['ctv']}",
        "vram_mb": vram,
        "baseline_turbo4": model.get("baseline_turbo4"),
        "benchmarks": {},
    }

    for bk, bench in BENCHMARKS.items():
        runs = []
        for run in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
            try:
                r = run_streaming(bench["prompt"])
                tf = f"{OUTPUT_DIR}/{model['key']}_{rq_config['key']}_{bk}_run{run}_test.py"
                tr = extract_and_test(r["content"], tf)
                print(
                    f" ttft={r['ttft_s']*1000:.0f}ms decode={r['decode_tps']:.1f} tok/s "
                    f"{tr['passed']}/{bench['expected']} ({r['tokens']} tok)"
                )
                runs.append({**r, **tr})
            except Exception as e:
                print(f" ERROR {e}")
                runs.append({"error": str(e)})

        valid = [x for x in runs if "passed" in x]
        if valid:
            best = max(x["passed"] for x in valid)
            avg = sum(x["passed"] for x in valid) / len(valid)
            decode_mean = statistics.mean(x["decode_tps"] for x in valid if x.get("decode_tps"))
            ttft_mean = statistics.mean(x["ttft_s"] for x in valid if x.get("ttft_s"))
            result["benchmarks"][bk] = {
                "expected": bench["expected"],
                "best": best,
                "avg": round(avg, 1),
                "decode_mean": round(decode_mean, 1),
                "ttft_mean": round(ttft_mean, 4),
                "runs": runs,
            }
            print(f"  -> best {best}/{bench['expected']}, avg {avg:.1f}, {decode_mean:.1f} tok/s")

    stop_server(proc)
    time.sleep(3)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("RotorQuant 5090 Benchmark")
    print("=" * 70)
    print(f"Server: {LLAMA_SERVER_WIN_PATH} (Windows johndpope fork)")
    print(f"Client: WSL2 Linux requests (bypasses urllib3-on-Windows TTFT bug)")
    print(f"Config: {len(MODELS)} models x {len(ROTORQUANT_CONFIGS)} rotorquant configs x 4 benchmarks x {RUNS} runs")
    print()

    all_results = []
    for model in MODELS:
        for rq_config in ROTORQUANT_CONFIGS:
            r = benchmark_one(model, rq_config)
            if r:
                all_results.append(r)
                # Save incrementally
                with open(f"{OUTPUT_DIR}/results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 120}")
    print("ROTORQUANT 5090 SUMMARY")
    print("=" * 120)
    print(f"{'Model':40s} | {'Config':20s} | {'Best':>8s} | {'Avg':>8s} | {'Decode':>10s} | {'Baseline':>12s} | {'Δ decode':>10s}")
    print("-" * 120)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:40s} | {r['config']:20s} | ERROR: {r['error']}")
            continue
        total_best = 0
        total_avg = 0
        total_exp = 0
        decode_samples = []
        for bk in ["expression_evaluator", "astar", "lru_cache", "string_processor"]:
            b = r["benchmarks"].get(bk, {})
            total_best += min(b.get("best", 0), b.get("expected", 5))
            total_avg += min(b.get("avg", 0), b.get("expected", 5))
            total_exp += b.get("expected", 5)
            if b.get("decode_mean"):
                decode_samples.append(b["decode_mean"])
        decode = statistics.mean(decode_samples) if decode_samples else 0
        baseline_decode = r.get("baseline_turbo4", {}).get("decode", 0)
        delta = ((decode - baseline_decode) / baseline_decode * 100) if baseline_decode else 0
        print(
            f"{r['model']:40s} | {r['config']:20s} | "
            f"{total_best}/{total_exp} | {total_avg:.1f}/{total_exp} | "
            f"{decode:>7.1f} t/s | {baseline_decode:>7.1f} t/s | "
            f"{delta:+6.1f}%"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
