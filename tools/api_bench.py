#!/usr/bin/env python3
"""Benchmark cloud/free-tier models via OpenAI-compatible APIs.

Runs the standard 4-benchmark coding suite (Expression Evaluator + A* +
LRU Cache + String Processor, 22 tests total) against remote API endpoints.
Reads API keys from the opencode config at ~/.config/opencode/opencode.json.

Throughput numbers are reported but are network-dependent — the quality
scores (pytest pass/fail) are the primary metric for cross-comparison with
local model rankings.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time

import requests

CONFIG_PATH = os.path.expanduser("~/.config/opencode/opencode.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "api_bench")
TEMP = 0.3
RUNS = 3

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

# Models to benchmark — keyed by a short slug
MODELS = [
    {
        "key": "gpt-oss-120b",
        "name": "GPT-OSS 120B (OpenRouter Free)",
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b:free",
        "max_tokens": 16384,
    },
    {
        "key": "minimax-m2.5",
        "name": "MiniMax M2.5 (OpenRouter Free)",
        "provider": "openrouter",
        "model_id": "minimax/minimax-m2.5:free",
        "max_tokens": 8192,
    },
    {
        "key": "minimax-m2.7",
        "name": "MiniMax M2.7 (NVIDIA)",
        "provider": "nvidia-minimax",
        "model_id": "minimaxai/minimax-m2.7",
        "max_tokens": 16384,
    },
]


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_api(config, provider_key):
    prov = config["provider"][provider_key]
    base_url = prov["options"]["baseURL"].rstrip("/")
    api_key = prov["options"].get("apiKey", "")
    return base_url, api_key


def run_streaming(base_url, api_key, model_id, prompt, max_tokens):
    t_send = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": TEMP,
            "stream": True,
        },
        stream=True,
        timeout=300,
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
    decode_time = (last_token_at - first_token_at) if last_token_at and last_token_at > first_token_at else None
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


def benchmark_model(config, model):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model['name']}")
    print(f"{'=' * 70}")

    base_url, api_key = get_api(config, model["provider"])

    result = {
        "model": model["name"],
        "model_key": model["key"],
        "model_id": model["model_id"],
        "provider": model["provider"],
        "benchmarks": {},
    }

    for bk, bench in BENCHMARKS.items():
        runs = []
        for run_num in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run_num}/{RUNS}...", end="", flush=True)
            try:
                r = run_streaming(base_url, api_key, model["model_id"],
                                  bench["prompt"], model["max_tokens"])
                if r is None:
                    print(" EMPTY RESPONSE")
                    runs.append({"error": "empty response"})
                    continue
                tf = os.path.join(OUTPUT_DIR, f"{model['key']}_{bk}_run{run_num}_test.py")
                tr = extract_and_test(r["content"], tf)
                tps_str = f" {r['decode_tps']:.1f} tok/s" if r.get('decode_tps') else ""
                print(
                    f" ttft={r['ttft_s']:.1f}s{tps_str} "
                    f"{tr['passed']}/{bench['expected']} ({r['tokens']} tok)"
                )
                runs.append({**r, **tr})
            except requests.exceptions.Timeout:
                print(" TIMEOUT (300s)")
                runs.append({"error": "timeout"})
            except Exception as e:
                print(f" ERROR {e}")
                runs.append({"error": str(e)})
            # Small delay to avoid rate limits on free tier
            time.sleep(2)

        valid = [x for x in runs if "passed" in x]
        if valid:
            best = max(x["passed"] for x in valid)
            avg = sum(x["passed"] for x in valid) / len(valid)
            decode_samples = [x["decode_tps"] for x in valid if x.get("decode_tps")]
            result["benchmarks"][bk] = {
                "expected": bench["expected"],
                "best": best,
                "avg": round(avg, 1),
                "decode_mean": round(statistics.mean(decode_samples), 1) if decode_samples else None,
                "runs": runs,
            }
            tps_info = f", {result['benchmarks'][bk]['decode_mean']} tok/s (API)" if decode_samples else ""
            print(f"  -> best {best}/{bench['expected']}, avg {avg:.1f}{tps_info}")
        else:
            print(f"  -> ALL RUNS FAILED")
            result["benchmarks"][bk] = {"expected": bench["expected"], "best": 0, "avg": 0, "runs": runs}

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_config()

    print("=" * 70)
    print("API Model Benchmark — Free Tier Coding Suite")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")
    print(f"Suite: 4 benchmarks × {RUNS} runs × {len(MODELS)} models")
    print(f"Output: {OUTPUT_DIR}")
    print()

    all_results = []
    for model in MODELS:
        try:
            r = benchmark_model(config, model)
            all_results.append(r)
        except Exception as e:
            print(f"  SKIPPING {model['name']}: {e}")
            all_results.append({"model": model["name"], "model_key": model["key"], "error": str(e)})

        with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 100}")
    print("API BENCH SUMMARY")
    print("=" * 100)
    print(f"{'Model':45s} | {'Best':>8s} | {'Avg':>8s} | {'API tok/s':>10s}")
    print("-" * 100)
    for r in all_results:
        if "error" in r and "benchmarks" not in r:
            print(f"{r['model']:45s} | ERROR: {r['error']}")
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
        print(
            f"{r['model']:45s} | {total_best}/{total_exp} | "
            f"{total_avg:.1f}/{total_exp} | {decode:>7.1f} t/s"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
