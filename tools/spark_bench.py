#!/usr/bin/env python3
"""
Spark 128GB Benchmark Runner

Benchmarks large models on the DGX Spark (GB10, 128GB unified memory).
Manages llama-server lifecycle, runs coding benchmarks and throughput tests,
and performs context scaling to find the practical max context per model.

Models tested:
- Gemma 4 31B-IT Q8_0 (dense — needs TurboQuant for large context)
- Qwen3.5-122B-A10B Q4_K_M (MoE, DeltaNet hybrid, tiny KV)
- Qwen3-Coder-Next Q4_K_M (MoE, DeltaNet hybrid, tiny KV)
- MiniMax-M2.5 UD-Q3_K_XL (MoE, Lightning Attention)

Usage:
    # Run all models with default settings
    python tools/spark_bench.py

    # Specific model
    python tools/spark_bench.py --models gemma31b-q8

    # Context scaling (find max context per model)
    python tools/spark_bench.py --context-scaling --models qwen122b

    # Throughput only
    python tools/spark_bench.py --throughput-only
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
# Paths — Spark (Linux/aarch64) environment
# ---------------------------------------------------------------------------

TURBOQUANT_SERVER = os.environ.get(
    "TURBOQUANT_SERVER",
    os.path.expanduser("~/git/TheTom/llama-cpp-turboquant/build/bin/llama-server"),
)

STANDARD_SERVER = os.environ.get(
    "STANDARD_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
)

MODELS_DIR = os.path.expanduser("~/.lmstudio/models")

# Model configurations for 128GB Spark
# KV budget analysis (120GB usable):
#   Gemma 31B Q8_0: 34GB weights + 870 KB/tok KV → turbo4 essential for >32K ctx
#   Qwen3.5-122B: 70GB weights + ~24 KB/tok KV → f16 KV fine, 256K fits
#   Qwen3-Coder-Next: 48GB weights + ~24 KB/tok KV → f16 KV fine, 262K fits
#   MiniMax-M2.5: 101GB weights + small KV (Lightning Attn) → tight, test empirically

MODEL_CONFIGS = {
    "gemma31b-q8": {
        "name": "Gemma 4 31B-IT Q8_0",
        "path": f"{MODELS_DIR}/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q8_0.gguf",
        "server": "standard",  # 128GB Spark has plenty of room for f16 KV
        "kv_configs": ["f16", "asym-q8"],
        "context_sizes": [32768, 65536, 90112],  # ~107GB at 90K with f16 KV
        "default_context": 32768,
        "default_kv": "f16",
        "reasoning": "off",
        "notes": "Dense 31B, 870KB/tok KV. f16 KV fits up to ~90K on 128GB Spark; turbo only needed for >100K.",
    },
    "qwen122b": {
        "name": "Qwen3.5-122B-A10B Q4_K_M (unsloth)",
        "path": f"{MODELS_DIR}/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        "server": "standard",  # MoE with tiny KV, no turbo needed
        "kv_configs": ["f16"],
        "context_sizes": [32768, 65536, 131072, 196608, 262144],
        "default_context": 32768,
        "default_kv": "f16",
        "reasoning": "off",
        "notes": "122B/10B MoE, DeltaNet hybrid (12 attn layers of 48), ~24KB/tok KV. Unsloth Q4_K_M.",
    },
    "qwen122b-bartowski": {
        "name": "Qwen3.5-122B-A10B Q4_K_M (bartowski)",
        "path": f"{MODELS_DIR}/bartowski/Qwen3.5-122B-A10B-GGUF/Qwen_Qwen3.5-122B-A10B-Q4_K_M/Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf",
        "server": "standard",
        "kv_configs": ["f16"],
        "context_sizes": [32768, 65536, 131072, 196608, 262144],
        "default_context": 32768,
        "default_kv": "f16",
        "reasoning": "off",
        "notes": "122B/10B MoE, DeltaNet hybrid. Bartowski Q4_K_M (more accurate than unsloth per Nauful).",
    },
    "qwen-coder": {
        "name": "Qwen3-Coder-Next UD-Q4_K_M",
        "path": f"{MODELS_DIR}/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_M.gguf",
        "server": "standard",
        "kv_configs": ["f16"],
        "context_sizes": [32768, 65536, 131072, 196608, 262144],
        "default_context": 32768,
        "default_kv": "f16",
        "reasoning": "off",
        "notes": "80B/3B MoE, DeltaNet hybrid (12 attn of 48), ~24KB/tok KV. Coding specialist.",
    },
    "minimax-m25": {
        "name": "MiniMax-M2.5 UD-Q3_K_XL",
        "path": f"{MODELS_DIR}/unsloth/MiniMax-M2.5-GGUF/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf",
        "server": "standard",  # Lightning Attention = tiny KV, but weights are 101GB
        "kv_configs": ["f16"],
        "context_sizes": [32768, 65536, 131072],  # conservative — only 19GB left for KV
        "default_context": 32768,
        "default_kv": "f16",
        "reasoning": "thinking",  # MiniMax is a thinking model; -rea off doesn't work
        # Custom chat template fixes llama.cpp issue #21465 (the official template
        # injects <think> in add_generation_prompt which breaks reasoning detection).
        "chat_template": os.path.expanduser(
            "~/git/gisenberg/local-model-eval/templates/minimax-m25-no-think.jinja"
        ),
        "max_tokens": 32768,        # Thinking model needs much bigger budget
        "request_timeout": 1500,    # 25 min — thinking takes time on this hardware
        "notes": "230B/10B MoE, Lightning Attention, 62 layers. 101GB weights → tight fit. "
                 "Thinking model: needs 32K max_tokens and longer request timeout.",
    },
}

# KV cache configs (turbo requires TurboQuant fork)
# Per Nauful: don't quantize K (breaks tool calls), only quantize V.
# Asymmetric f16-K/q8_0-V is recommended for production/agentic use.
KV_CONFIGS = {
    "f16":       {"ctk": "f16",    "ctv": "f16",    "label": "KV f16 (baseline)"},
    "asym-q8":   {"ctk": "f16",    "ctv": "q8_0",   "label": "KV f16K/q8V (asymmetric)"},
    "q8_0":      {"ctk": "q8_0",   "ctv": "q8_0",   "label": "KV q8_0 (symmetric)"},
    "turbo4":    {"ctk": "turbo4",  "ctv": "turbo4",  "label": "KV turbo4 (4-bit sym)"},
    "turbo3":    {"ctk": "turbo3",  "ctv": "turbo3",  "label": "KV turbo3 (3-bit sym)"},
    "asym-t4":   {"ctk": "f16",    "ctv": "turbo4",  "label": "KV f16K/turbo4V (asymmetric)"},
}

DEFAULT_PORT = 8080

# Same prompts as turboquant_bench.py — consistent with existing results
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

def get_server_binary(model_cfg):
    """Return the correct llama-server binary based on model config."""
    if model_cfg["server"] == "turboquant":
        if not os.path.isfile(TURBOQUANT_SERVER):
            print(f"WARNING: TurboQuant server not found at {TURBOQUANT_SERVER}")
            print("Falling back to standard server (turbo KV types won't work)")
            return STANDARD_SERVER
        return TURBOQUANT_SERVER
    return STANDARD_SERVER


def start_server(server_bin, model_path, port, context_length, ctk, ctv,
                 gpu_layers=99, reasoning="off", chat_template=None):
    """Start llama-server as a subprocess. Returns the Popen handle."""
    cmd = [
        server_bin,
        "-m", model_path,
        "--port", str(port),
        "-c", str(context_length),
        "-ngl", str(gpu_layers),
        "-fa", "on",
        "-ctk", ctk,
        "-ctv", ctv,
        "-np", "1",  # single slot for max context
        "--temp", "0",
        "--no-mmap",  # load fully into memory (recommended for Spark unified mem)
        "--jinja",    # enable Jinja chat templates
    ]
    if reasoning == "off":
        cmd.extend(["-rea", "off"])
    elif reasoning == "thinking":
        # Model is a thinking model that doesn't honor -rea off
        # (or has a known llama.cpp bug). Don't pass -rea flag at all,
        # let the model and template control it.
        pass
    else:
        cmd.extend(["-rea", "on", "--reasoning-budget", "16384"])

    if chat_template:
        cmd.extend(["--chat-template-file", chat_template])

    print(f"  Server: {os.path.basename(server_bin)}")
    template_note = f" --chat-template-file {os.path.basename(chat_template)}" if chat_template else ""
    print(f"  Args: -c {context_length} -ctk {ctk} -ctv {ctv} -np 1 -rea {reasoning}{template_note}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_for_server(port, timeout=300):
    """Poll the health endpoint until the server is ready.
    Longer timeout for large models on Spark (loading 100GB+ takes time).
    """
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
        time.sleep(2)
    return False


def stop_server(proc):
    """Terminate the llama-server process."""
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(port, prompt, max_tokens=16384, stream=True, temperature=0,
                  request_timeout=600):
    """Send a chat completion request and measure performance."""
    base_url = f"http://localhost:{port}"
    messages = [{"role": "user", "content": prompt}]

    if not stream:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=request_timeout,
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

    # Streaming (throughput measurement)
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
            "temperature": temperature,
            "stream": True,
        },
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

def run_code_benchmarks(port, model_name, kv_label, output_dir,
                        max_tokens=16384, request_timeout=600):
    """Run all 3 coding benchmarks and save results."""
    results = []
    for bench_key, bench in BENCHMARKS.items():
        print(f"\n    [{bench['name']}] Generating...", end="", flush=True)
        try:
            result = run_inference(port, bench["prompt"], max_tokens=max_tokens,
                                   stream=False, request_timeout=request_timeout)
            print(
                f" {result['tok_per_sec']:.1f} tok/s | "
                f"{result['tokens']} tok | {result['elapsed_s']:.1f}s | "
                f"think: {len(result['reasoning'])}c | content: {len(result['content'])}c | "
                f"finish={result['finish_reason']}"
            )
            if result["finish_reason"] == "length":
                print("    WARNING: output truncated (hit max_tokens)")

            safe_name = (
                f"{model_name}_{kv_label}_{bench_key}"
                .replace(" ", "_")
                .replace("/", "-")  # KV labels like "f16K/q8V" must not become path separators
            )
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
                    "platform": "DGX Spark GB10 128GB",
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


def run_throughput_benchmark(port, model_name, kv_label, max_tokens=2048,
                              request_timeout=600):
    """Run streaming throughput benchmark."""
    print(f"    [Throughput] Generating...", end="", flush=True)
    try:
        result = run_inference(port, THROUGHPUT_PROMPT, max_tokens=max_tokens,
                               stream=True, request_timeout=request_timeout)
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Print a comparison table."""
    if not all_results:
        return

    print(f"\n{'=' * 120}")
    print("SPARK 128GB BENCHMARK RESULTS")
    print(f"{'=' * 120}")
    print(
        f"{'Model':35s} | {'KV Config':20s} | {'Benchmark':25s} | "
        f"{'Tok/s':>7s} | {'Tokens':>7s} | {'Time':>7s}"
    )
    print("-" * 120)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:35s} | {r['kv_config']:20s} | {r['benchmark']:25s} | ERROR: {r['error'][:30]}")
        else:
            tps = r.get("tok_per_sec", 0)
            tok = r.get("tokens", 0)
            elapsed = r.get("elapsed_s", r.get("total_s", 0))
            print(
                f"{r['model']:35s} | {r['kv_config']:20s} | {r['benchmark']:25s} | "
                f"{tps:>7.1f} | {tok:>7d} | {elapsed:>6.1f}s"
            )
    print(f"{'=' * 120}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark large models on DGX Spark (128GB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help=f"Models to test. Choices: {list(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
    )
    parser.add_argument(
        "--throughput-only", action="store_true",
        help="Skip coding benchmarks, only run throughput test",
    )
    parser.add_argument(
        "--context-scaling", action="store_true",
        help="Run context scaling test to find max usable context",
    )
    parser.add_argument(
        "--output-dir", default="experiments/spark_bench",
    )
    parser.add_argument(
        "--kv", choices=list(KV_CONFIGS.keys()), default=None,
        help="Override the model's default KV cache config (e.g. 'asym-q8' for f16K/q8V)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for model_key in args.models:
        model_cfg = MODEL_CONFIGS[model_key]
        model_path = model_cfg["path"]
        model_name = model_cfg["name"]
        server_bin = get_server_binary(model_cfg)

        if not os.path.isfile(model_path):
            print(f"\nSKIP: {model_name} — not found at {model_path}")
            continue

        # Per-model overrides
        chat_template = model_cfg.get("chat_template")
        max_tokens = model_cfg.get("max_tokens", 16384)
        request_timeout = model_cfg.get("request_timeout", 600)

        if args.kv:
            kv_types = [args.kv]
        elif args.context_scaling:
            kv_types = model_cfg["kv_configs"]
        else:
            kv_types = [model_cfg["default_kv"]]

        for kv_key in kv_types:
            kv_cfg = KV_CONFIGS[kv_key]
            kv_label = kv_cfg["label"]

            if args.context_scaling:
                for ctx_size in model_cfg["context_sizes"]:
                    print(f"\n{'=' * 80}")
                    print(f"{model_name} | {kv_label} | Context {ctx_size // 1024}K")
                    print(f"  {model_cfg['notes']}")
                    print(f"{'=' * 80}")

                    proc = start_server(
                        server_bin, model_path, args.port, ctx_size,
                        kv_cfg["ctk"], kv_cfg["ctv"],
                        reasoning=model_cfg["reasoning"],
                        chat_template=chat_template,
                    )
                    print("  Waiting for server...", end="", flush=True)
                    if not wait_for_server(args.port, timeout=600):
                        print(" TIMEOUT — server failed to start (OOM?)")
                        stop_server(proc)
                        all_results.append({
                            "model": model_name, "kv_config": kv_label,
                            "context_size": ctx_size,
                            "benchmark": f"context_{ctx_size // 1024}K",
                            "error": "server start timeout",
                        })
                        break  # if one size fails, larger ones will too
                    print(" ready")

                    result = run_throughput_benchmark(args.port, model_name, kv_label)
                    result["context_size"] = ctx_size
                    result["benchmark"] = f"context_{ctx_size // 1024}K"
                    all_results.append(result)

                    stop_server(proc)
                    time.sleep(3)
            else:
                print(f"\n{'=' * 80}")
                print(f"{model_name} | {kv_label}")
                print(f"  {model_cfg['notes']}")
                print(f"{'=' * 80}")

                proc = start_server(
                    server_bin, model_path, args.port,
                    model_cfg["default_context"],
                    kv_cfg["ctk"], kv_cfg["ctv"],
                    reasoning=model_cfg["reasoning"],
                    chat_template=chat_template,
                )
                print("  Waiting for server...", end="", flush=True)
                if not wait_for_server(args.port, timeout=600):
                    print(" TIMEOUT — server failed to start")
                    stop_server(proc)
                    continue
                print(" ready")

                result = run_throughput_benchmark(
                    args.port, model_name, kv_label,
                    request_timeout=request_timeout,
                )
                all_results.append(result)

                if not args.throughput_only:
                    bench_results = run_code_benchmarks(
                        args.port, model_name, kv_label, args.output_dir,
                        max_tokens=max_tokens, request_timeout=request_timeout,
                    )
                    all_results.extend(bench_results)

                stop_server(proc)
                time.sleep(3)

    print_summary(all_results)

    outfile = f"{args.output_dir}/spark_results.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to {outfile}")


if __name__ == "__main__":
    main()
