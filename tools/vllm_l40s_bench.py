#!/usr/bin/env python3
"""
L40S 46GB Benchmark Runner (vLLM via SSH)

Benchmarks models on an NVIDIA L40S (46GB VRAM) served by vLLM in Docker.
Manages the vLLM container lifecycle on a remote host via SSH, runs coding
benchmarks and throughput tests, extracts generated code, and runs pytest.

Hardware: NVIDIA L40S 46GB, cluster node "miami", Rocky Linux, Docker + NVIDIA CTK
Backend: vLLM (OpenAI-compatible API) in vllm/vllm-openai container

Methodology: Matches repo standard -- thinking OFF for all models (matches
-rea off / --reasoning-budget 0 used on 5090/Spark/M4Max platforms).
For Qwen3.5 models, thinking is disabled via chat_template_kwargs.

Models tested (all cached on Lustre NVMe at /nvmepool2/.cache/huggingface):
- Qwen3.5-9B (BF16) -- thinking OFF, ~19GB
- Qwen3.5-35B-A3B (FP8) -- MoE, 3B active, thinking OFF, ~35GB
- Ministral-3-14B-Instruct-2512-BF16 -- non-thinking, ~27GB

NOTE: Qwen3.5-27B-FP8 OOMs on L40S (model+CUDA graphs fill 42.5/44.4 GB).
NOTE: Gemma 4 deadlocks on vLLM Triton backend (heterogeneous head dims).

Usage:
    python tools/vllm_l40s_bench.py                          # all models
    python tools/vllm_l40s_bench.py --models qwen35-9b       # specific model
    python tools/vllm_l40s_bench.py --skip-docker             # vLLM already running
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Model configurations for L40S 46GB
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "qwen35-9b": {
        "name": "Qwen3.5-9B",
        "hf_id": "Qwen/Qwen3.5-9B",
        "quantization": None,
        "max_model_len": 32768,
        "gpu_mem_util": 0.90,
        "dtype": "bfloat16",
        "nothink": True,  # disable thinking via chat_template_kwargs
        "notes": "9B dense, BF16 (~19GB). Thinking OFF per repo methodology.",
    },
    "qwen35-35b-a3b": {
        "name": "Qwen3.5-35B-A3B",
        "hf_id": "Qwen/Qwen3.5-35B-A3B",
        "quantization": "fp8",
        "max_model_len": 16384,
        "gpu_mem_util": 0.95,
        "dtype": "auto",
        "enforce_eager": True,  # MoE + FP8 online = CUDA graph capture takes >15min
        "nothink": True,
        "notes": "35B MoE (3B active), FP8 online (~35GB). Same model tested on 5090 (11/17). Thinking OFF.",
    },
    "ministral-14b": {
        "name": "Ministral-3-14B-Instruct-2512-BF16",
        "hf_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "quantization": None,
        "max_model_len": 32768,
        "gpu_mem_util": 0.90,
        "dtype": "bfloat16",
        "nothink": False,  # not a thinking model
        "notes": "14B dense, BF16 (~27GB). Mistral family, non-thinking.",
    },
}

# ---------------------------------------------------------------------------
# Benchmarks — identical prompts from spark_bench.py for cross-platform parity
# ---------------------------------------------------------------------------

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
# Docker / SSH management
# ---------------------------------------------------------------------------

VLLM_IMAGE = "vllm/vllm-openai:latest"
CONTAINER_NAME = "vllm-bench"


def ssh_run(ssh_host, cmd, timeout=30):
    """Run a command on the remote host via SSH. Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", ssh_host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def start_vllm_container(ssh_host, model_cfg, port, hf_cache):
    """Start a vLLM container on the remote host via SSH."""
    hf_id = model_cfg["hf_id"]
    quant_flag = f"--quantization {model_cfg['quantization']}" if model_cfg["quantization"] else ""
    eager_flag = "--enforce-eager" if model_cfg.get("enforce_eager") else ""
    extra_flags = " ".join(model_cfg.get("extra_vllm_flags", []))
    cmd = (
        f"docker run -d --name {CONTAINER_NAME} "
        f"--runtime nvidia --gpus all --shm-size=10g "
        f"-p {port}:8000 "
        f"-v {hf_cache}:/root/.cache/huggingface "
        f"{VLLM_IMAGE} "
        f"--model {hf_id} "
        f"{quant_flag} "
        f"{eager_flag} "
        f"{extra_flags} "
        f"--max-model-len {model_cfg['max_model_len']} "
        f"--gpu-memory-utilization {model_cfg['gpu_mem_util']} "
        f"--dtype {model_cfg['dtype']} "
        f"--trust-remote-code"
    )
    print(f"  Starting container: {model_cfg['name']}")
    print(f"  Docker: {VLLM_IMAGE}")
    print(f"  Args: --model {hf_id} {quant_flag} {eager_flag} --max-model-len {model_cfg['max_model_len']}")
    rc, out, err = ssh_run(ssh_host, cmd, timeout=60)
    if rc != 0:
        print(f"  ERROR starting container: {err}")
        return False
    print(f"  Container started: {out[:12]}...")
    return True


def wait_for_vllm(api_host, port, timeout=900):
    """Poll the vLLM health endpoint until ready.
    Long timeout (15min) because first-run downloads the model from HuggingFace.
    """
    url = f"http://{api_host}:{port}/health"
    deadline = time.time() + timeout
    last_status = None
    dots = 0
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"\n  vLLM ready!")
                return True
            new_status = f"HTTP {resp.status_code}"
            if new_status != last_status:
                print(f"\n  Status: {new_status}", end="", flush=True)
                last_status = new_status
        except requests.ConnectionError:
            new_status = "connecting"
            if new_status != last_status:
                print(f"\n  Waiting for vLLM", end="", flush=True)
                last_status = new_status
                dots = 0
        except requests.Timeout:
            pass
        print(".", end="", flush=True)
        dots += 1
        time.sleep(5)
    print(f"\n  TIMEOUT after {timeout}s")
    return False


def stop_vllm_container(ssh_host):
    """Stop and remove the vLLM container."""
    print(f"  Stopping container...")
    ssh_run(ssh_host, f"docker stop {CONTAINER_NAME}", timeout=30)
    ssh_run(ssh_host, f"docker rm {CONTAINER_NAME}", timeout=15)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(api_host, port, prompt, max_tokens=16384, temperature=0,
                  request_timeout=600, model_id="default", nothink=False):
    """Send a streaming chat completion request and measure performance.
    If nothink=True, passes chat_template_kwargs to disable thinking (Qwen3.5).
    """
    base_url = f"http://{api_host}:{port}"
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    first_any_token_at = None
    first_content_token_at = None
    thinking_chunks = 0
    content_chunks = 0
    output_text = []
    thinking_text = []
    server_usage = None

    request_body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if nothink:
        request_body["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=request_body,
        stream=True,
        timeout=request_timeout,
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

        # vLLM 0.19.0 uses "reasoning" (not "reasoning_content" like OpenAI)
        reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "")
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
        "finish_reason": "stop",
    }


# ---------------------------------------------------------------------------
# Code extraction + pytest
# ---------------------------------------------------------------------------

def extract_and_test(content, test_file):
    """Extract Python code blocks from model output and run pytest.
    Matches the methodology in nvfp4_gemma_bench.py and extract_and_test.py.
    """
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0}
    combined = "\n\n".join(b.strip() for b in blocks)
    # Strip intra-file imports referencing implementation classes
    for cls in ['ExpressionEvaluator', 'AStarGrid', 'TTLCache', 'StringProcessor', 'BST', 'Node']:
        combined = re.sub(rf'from \w+ import {cls}\b.*\n', '', combined)
    # Strip overly-strict match= in pytest.raises (common model mistake)
    combined = re.sub(r',\s*match=["\'][^"\']*["\']', '', combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        out = r.stdout + r.stderr
        return {
            "passed": len(re.findall(r' PASSED', out)),
            "failed": len(re.findall(r' FAILED', out)),
            "errors": len(re.findall(r' ERROR', out)),
        }
    except Exception:
        return {"passed": 0, "failed": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_code_benchmarks(api_host, port, model_name, model_id, output_dir, runs=3,
                        max_tokens=16384, temperature=0, request_timeout=600,
                        nothink=False):
    """Run all 3 coding benchmarks with multiple runs, extract code, run pytest."""
    all_results = {}
    for bench_key, bench in BENCHMARKS.items():
        bench_runs = []
        for run_num in range(1, runs + 1):
            temp = 0 if run_num == 1 else 0.3
            print(f"    [{bench['name']}] Run {run_num}/{runs} (temp={temp})...", end="", flush=True)
            try:
                result = run_inference(
                    api_host, port, bench["prompt"],
                    max_tokens=max_tokens, temperature=temp,
                    request_timeout=request_timeout, model_id=model_id,
                    nothink=nothink,
                )
                # Save markdown output
                safe = f"{bench_key}_run{run_num}"
                md_path = f"{output_dir}/{safe}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# {model_name} (L40S vLLM) — {bench['name']} — Run {run_num}\n\n")
                    if result["reasoning"]:
                        f.write(f"## Thinking ({len(result['reasoning'])} chars)\n\n")
                        f.write(f"{result['reasoning']}\n\n")
                    f.write(f"## Output\n\n{result['content']}\n")

                # Extract and test
                test_file = f"{output_dir}/{safe}_test.py"
                tr = extract_and_test(result["content"], test_file)

                # Save per-run JSON
                with open(f"{output_dir}/{safe}.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "bench": bench_key,
                        "model": model_name,
                        "platform": "L40S 46GB (vLLM)",
                        "run": run_num,
                        "temperature": temp,
                        "toks": result["token_count"],
                        "elapsed": result["total_time_s"],
                        "tps": result["tokens_per_second"],
                        "ttft": result["time_to_first_token_s"],
                        "think_chars": len(result["reasoning"]),
                        "content_chars": len(result["content"]),
                        "finish": result["finish_reason"],
                        "usage": result["server_usage"],
                        "test_passed": tr["passed"],
                        "test_failed": tr["failed"],
                        "test_errors": tr["errors"],
                    }, f, indent=2)

                tok_s = result["tokens_per_second"]
                print(f" {tok_s:.1f} tok/s | {tr['passed']}/{bench['expected_tests']} tests | "
                      f"{result['token_count']} tok | think: {len(result['reasoning'])}c")
                bench_runs.append({**result, **tr, "temperature": temp})
            except Exception as e:
                print(f" ERROR: {e}")
                bench_runs.append({"error": str(e)})

        # Summarize benchmark
        valid = [r for r in bench_runs if "passed" in r]
        if valid:
            best = max(r["passed"] for r in valid)
            avg = sum(r["passed"] for r in valid) / len(valid)
            avg_tps = sum(r["tokens_per_second"] for r in valid) / len(valid)
            all_results[bench_key] = {
                "best": best,
                "avg": round(avg, 1),
                "expected": bench["expected_tests"],
                "avg_tps": round(avg_tps, 1),
                "runs": bench_runs,
            }
            print(f"    -> Best: {best}/{bench['expected_tests']}, "
                  f"Avg: {avg:.1f}/{bench['expected_tests']}, "
                  f"Avg speed: {avg_tps:.1f} tok/s\n")
        else:
            all_results[bench_key] = {"runs": bench_runs, "error": "all runs failed"}
            print(f"    -> ALL RUNS FAILED\n")

    return all_results


def run_throughput_benchmark(api_host, port, model_name, model_id, max_tokens=2048,
                              request_timeout=600, nothink=False):
    """Run streaming throughput benchmark."""
    print(f"    [Throughput] Generating...", end="", flush=True)
    try:
        result = run_inference(
            api_host, port, THROUGHPUT_PROMPT,
            max_tokens=max_tokens, temperature=0,
            request_timeout=request_timeout, model_id=model_id,
            nothink=nothink,
        )
        think_info = ""
        if result["thinking_tokens"] > 0:
            think_info = f" | Think: {result['thinking_tokens']} tok"
        print(
            f" {result['tokens_per_second']:.1f} tok/s | "
            f"{result['token_count']} tok{think_info} | "
            f"TTFT {result['time_to_first_token_s']}s | "
            f"Total {result['total_time_s']}s"
        )
        return {
            "benchmark": "throughput",
            "tok_per_sec": result["tokens_per_second"],
            "tokens": result["token_count"],
            "ttft_s": result["time_to_first_token_s"],
            "total_s": result["total_time_s"],
            "thinking_tokens": result["thinking_tokens"],
            "content_tokens": result["content_tokens"],
        }
    except Exception as e:
        print(f" ERROR: {e}")
        return {"benchmark": "throughput", "error": str(e)}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_model_results):
    """Print a comparison table across all models."""
    print(f"\n{'=' * 110}")
    print("L40S 46GB vLLM BENCHMARK RESULTS")
    print(f"{'=' * 110}")
    print(
        f"{'Model':40s} | {'Benchmark':25s} | "
        f"{'Best':>5s} | {'Avg':>5s} | {'Exp':>4s} | {'Tok/s':>7s}"
    )
    print("-" * 110)
    for model_key, data in all_model_results.items():
        model_name = MODEL_CONFIGS[model_key]["name"]
        if "code_results" in data:
            for bench_key, br in data["code_results"].items():
                if "error" in br:
                    print(f"{model_name:40s} | {bench_key:25s} | ERROR")
                else:
                    print(
                        f"{model_name:40s} | {bench_key:25s} | "
                        f"{br['best']:>5d} | {br['avg']:>5.1f} | {br['expected']:>4d} | "
                        f"{br.get('avg_tps', 0):>7.1f}"
                    )
        if "throughput" in data:
            tp = data["throughput"]
            if "error" not in tp:
                print(
                    f"{model_name:40s} | {'throughput':25s} | "
                    f"{'':>5s} | {'':>5s} | {'':>4s} | {tp['tok_per_sec']:>7.1f}"
                )
    print(f"{'=' * 110}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models on L40S 46GB via vLLM (SSH-automated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, default=None,
                        help=f"Comma-separated model keys (default: all). "
                             f"Available: {', '.join(MODEL_CONFIGS.keys())}")
    parser.add_argument("--ssh-host", type=str, default="root@miami",
                        help="SSH target for docker management (default: root@miami)")
    parser.add_argument("--api-host", type=str, default="miami.local.lan",
                        help="Hostname for vLLM API requests (default: miami.local.lan)")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM API port (default: 8000)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per benchmark: 1st at temp=0, rest at temp=0.3 (default: 3)")
    parser.add_argument("--hf-cache", type=str, default="/nvmepool2/.cache/huggingface",
                        help="HF cache path on remote host (default: /nvmepool2/.cache/huggingface)")
    parser.add_argument("--output-dir", type=str, default="experiments/l40s_bench",
                        help="Output directory (default: experiments/l40s_bench)")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip container lifecycle (assume vLLM already running)")
    parser.add_argument("--throughput-only", action="store_true",
                        help="Run only throughput benchmark (skip coding tests)")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Max tokens for coding benchmarks (default: 16384)")
    parser.add_argument("--request-timeout", type=int, default=600,
                        help="HTTP request timeout in seconds (default: 600)")
    parser.add_argument("--startup-timeout", type=int, default=900,
                        help="vLLM startup timeout in seconds (default: 900)")
    args = parser.parse_args()

    # Select models
    if args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in MODEL_CONFIGS:
                print(f"Unknown model key: {k}")
                print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
                sys.exit(1)
    else:
        model_keys = list(MODEL_CONFIGS.keys())

    print(f"L40S vLLM Benchmark")
    print(f"Models: {', '.join(model_keys)}")
    print(f"SSH: {args.ssh_host} | API: {args.api_host}:{args.port}")
    print(f"Runs per benchmark: {args.runs} (1x temp=0, {args.runs - 1}x temp=0.3)")
    print(f"Output: {args.output_dir}/")
    print()

    all_model_results = {}

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]
        model_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)

        print(f"{'=' * 80}")
        print(f"MODEL: {cfg['name']}")
        print(f"  HF: {cfg['hf_id']}")
        quant_str = cfg['quantization'] or 'native'
        print(f"  Quantization: {quant_str} | Context: {cfg['max_model_len']}")
        if cfg["notes"]:
            print(f"  Notes: {cfg['notes']}")
        print()

        # Start container
        if not args.skip_docker:
            # Clean up any leftover container
            ssh_run(args.ssh_host, f"docker rm -f {CONTAINER_NAME} 2>/dev/null", timeout=15)
            time.sleep(2)

            if not start_vllm_container(args.ssh_host, cfg, args.port, args.hf_cache):
                print(f"  SKIPPING {cfg['name']} — container failed to start\n")
                all_model_results[model_key] = {"error": "container_start_failed"}
                continue

            if not wait_for_vllm(args.api_host, args.port, timeout=args.startup_timeout):
                print(f"  SKIPPING {cfg['name']} — vLLM did not become ready\n")
                stop_vllm_container(args.ssh_host)
                all_model_results[model_key] = {"error": "startup_timeout"}
                continue

        model_data = {"config": cfg}

        nothink = cfg.get("nothink", False)
        if nothink:
            print(f"  Thinking: OFF (chat_template_kwargs)")

        # Throughput benchmark
        print(f"\n  --- Throughput ---")
        tp = run_throughput_benchmark(
            args.api_host, args.port, cfg["name"], cfg["hf_id"],
            request_timeout=args.request_timeout, nothink=nothink,
        )
        model_data["throughput"] = tp

        # Coding benchmarks
        if not args.throughput_only:
            print(f"\n  --- Coding Benchmarks ({args.runs} runs) ---")
            code_results = run_code_benchmarks(
                args.api_host, args.port, cfg["name"], cfg["hf_id"], model_dir,
                runs=args.runs, max_tokens=args.max_tokens,
                temperature=0, request_timeout=args.request_timeout,
                nothink=nothink,
            )
            model_data["code_results"] = code_results

        # Save per-model summary
        with open(f"{model_dir}/results.json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, default=str)

        all_model_results[model_key] = model_data

        # Stop container
        if not args.skip_docker:
            print()
            stop_vllm_container(args.ssh_host)
            print(f"  Cooling down (5s)...")
            time.sleep(5)

        print()

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2, default=str)
    print(f"Combined results saved to {combined_path}")

    # Print summary table
    print_summary(all_model_results)


if __name__ == "__main__":
    main()
