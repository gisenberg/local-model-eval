#!/usr/bin/env python3
"""Coding-suite quality bench for Gemma 4 31B-IT + iso3/iso3 on the rebased
johndpope fork. Runs the standard 4-benchmark coding suite at 32K and one
sanity-check of Expression Evaluator at 128K to confirm quality doesn't
collapse at the unlocked long-context band.

This exists as a separate script from tools/rotorquant_5090_bench.py because
(a) it uses the rebased build dir (build-rebase, not build) and (b) it tests
only one model/config combination — the one Gemma 4 entry that actually
benefits from rotorquant.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time

import requests

LLAMA_SERVER_WIN_PATH = "/mnt/t/git/johndpope/llama-cpp-turboquant/build-rebase/bin/Release/llama-server.exe"
MODEL_PATH_WIN = "C:/Users/gisen/.lmstudio/models/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
WIN_HOST_IP = "172.20.240.1"
PORT = 8091
OUTPUT_DIR = "/mnt/t/git/local-model-eval/experiments/rotorquant_5090_gemma31b"
TEMP = 0.3
RUNS = 3
MAX_TOKENS = 16384

# Baseline for comparison (from MODEL_RANKINGS_5090.md, turbo4, 32K)
BASELINE_TURBO4 = {"decode": 50.3, "best_of_3": "30/31", "avg": "30.7/31"}

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


def start_server(ctx):
    cmd = [
        LLAMA_SERVER_WIN_PATH,
        "-m", MODEL_PATH_WIN,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "-c", str(ctx),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "iso3",
        "-ctv", "iso3",
        "-np", "1",
        "-rea", "off",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
            "model": "local",
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
        if c:
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            n_tokens += 1
            content.append(c)

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
    for cls in ['ExpressionEvaluator', 'AStarGrid', 'TTLCache', 'StringProcessor']:
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


def run_bench(ctx_label, ctx_size, benchmarks_to_run):
    print(f"\n{'=' * 70}")
    print(f"Gemma 4 31B-IT Q4_K_M + iso3/iso3 @ context = {ctx_label}")
    print(f"{'=' * 70}")

    proc = start_server(ctx_size)
    print("  Starting server...", end="", flush=True)
    if not wait_for_server():
        print(" TIMEOUT")
        stop_server(proc)
        return {"ctx": ctx_label, "error": "server timeout"}
    print(" ready")

    time.sleep(2)
    vram = get_vram_mb()
    print(f"  VRAM: {vram} MB")

    result = {
        "ctx_label": ctx_label,
        "ctx_size": ctx_size,
        "vram_mb": vram,
        "benchmarks": {},
    }

    for bk, bench in benchmarks_to_run.items():
        runs = []
        for run in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
            try:
                r = run_streaming(bench["prompt"])
                tf = f"{OUTPUT_DIR}/gemma31b_iso3_{ctx_label}_{bk}_run{run}_test.py"
                tr = extract_and_test(r["content"], tf)
                print(
                    f" ttft={r['ttft_s']*1000:.0f}ms decode={r['decode_tps']:.1f} "
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
            decode_samples = [x["decode_tps"] for x in valid if x.get("decode_tps")]
            ttft_samples = [x["ttft_s"] for x in valid if x.get("ttft_s")]
            result["benchmarks"][bk] = {
                "expected": bench["expected"],
                "best": best,
                "avg": round(avg, 1),
                "decode_mean": round(statistics.mean(decode_samples), 1) if decode_samples else None,
                "ttft_mean": round(statistics.mean(ttft_samples), 4) if ttft_samples else None,
                "runs": runs,
            }
            print(f"  -> best {best}/{bench['expected']}, avg {avg:.1f}, "
                  f"{result['benchmarks'][bk]['decode_mean']} tok/s")

    stop_server(proc)
    time.sleep(3)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("RotorQuant Gemma 4 31B-IT H5 quality bench")
    print("=" * 70)
    print(f"Server: {LLAMA_SERVER_WIN_PATH}")
    print(f"Baseline (turbo4 @ 32K): decode {BASELINE_TURBO4['decode']} t/s, "
          f"best_of_3 {BASELINE_TURBO4['best_of_3']}, avg {BASELINE_TURBO4['avg']}")
    print()

    results = []

    # Phase 1: full 4-benchmark suite at 32K
    results.append(run_bench("32K", 32768, BENCHMARKS))

    # Phase 2: sanity check at 128K — just Expression Evaluator
    # (confirms quality doesn't collapse when we use the unlocked context)
    results.append(run_bench("128K", 131072, {
        "expression_evaluator": BENCHMARKS["expression_evaluator"],
    }))

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 80}")
    print("GEMMA 4 31B-IT + iso3/iso3 SUMMARY")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['ctx_label']}: ERROR {r['error']}")
            continue
        total_best = 0
        total_avg = 0
        total_exp = 0
        decode_samples = []
        for bk in ["expression_evaluator", "astar", "lru_cache", "string_processor"]:
            b = r["benchmarks"].get(bk)
            if not b:
                continue
            total_best += min(b.get("best", 0), b.get("expected", 5))
            total_avg += min(b.get("avg", 0), b.get("expected", 5))
            total_exp += b.get("expected", 5)
            if b.get("decode_mean"):
                decode_samples.append(b["decode_mean"])
        decode = statistics.mean(decode_samples) if decode_samples else 0
        delta = ((decode - BASELINE_TURBO4["decode"]) / BASELINE_TURBO4["decode"] * 100) if decode else 0
        print(
            f"{r['ctx_label']:5s} | VRAM {r.get('vram_mb', '?'):>6} MB | "
            f"best {total_best}/{total_exp} | avg {total_avg:.1f}/{total_exp} | "
            f"decode {decode:>5.1f} t/s | Δ vs turbo4 {delta:+5.1f}%"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
