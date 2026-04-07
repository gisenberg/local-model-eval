#!/usr/bin/env python3
"""
LM Studio Model Benchmark Tool

Automatically loads, benchmarks, and unloads local models via the LM Studio
REST API, measuring tokens/sec, time-to-first-token, and generation time.

Usage:
    # Benchmark all downloaded models with default coding prompt
    python lmstudio_bench.py

    # Benchmark specific models
    python lmstudio_bench.py --models "lmstudio-community/qwen2.5-coder-7b" "bartowski/gemma-2-9b"

    # Custom prompt, multiple runs, custom max tokens
    python lmstudio_bench.py --prompt "Write a Python quicksort" --max-tokens 1024 --runs 3

    # Different port / context length
    python lmstudio_bench.py --port 1234 --context-length 8192
"""

import argparse
import json
import sys
import time

import requests

DEFAULT_PROMPT = """\
Write a Python function that takes a list of integers and returns the longest \
increasing subsequence. Include type hints, handle edge cases, and add a brief \
docstring. Then write 3 unit tests using pytest.\
"""

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(base_url, path):
    resp = requests.get(f"{base_url}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url, path, body, timeout=300):
    resp = requests.post(
        f"{base_url}{path}",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def discover_models(base_url):
    """Return list of LLM model dicts from LM Studio (excludes embeddings)."""
    data = api_get(base_url, "/api/v1/models")
    return [m for m in data.get("models", data.get("data", [])) if m.get("type", "llm") == "llm"]


def get_model_id(model):
    """Extract the usable model identifier from a model dict."""
    return model.get("key", model.get("id", ""))


def is_model_loaded(model):
    """Check if a model has any loaded instances."""
    return bool(model.get("loaded_instances"))


def load_model(base_url, model_id, context_length=4096):
    """Load a model into memory. Returns the API response."""
    body = {
        "model": model_id,
        "context_length": context_length,
        "flash_attention": True,
    }
    return api_post(base_url, "/api/v1/models/load", body, timeout=600)


def unload_model(base_url, instance_id):
    """Unload a model from memory."""
    return api_post(base_url, "/api/v1/models/unload", {"instance_id": instance_id})


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def benchmark_streaming(base_url, model_id, prompt, max_tokens):
    """
    Send a streaming chat completion and measure performance.

    Tracks both reasoning_content (thinking) and content (visible output)
    token streams separately — important for models like Qwen 3.5 that use
    chain-of-thought reasoning tokens.
    """
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    first_any_token_at = None   # first token of any kind (thinking or content)
    first_content_token_at = None  # first visible content token
    thinking_chunks = 0
    content_chunks = 0
    output_text = []
    thinking_text = []
    server_usage = None

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_id,
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

        # Capture server-side usage if present (often in the last chunk)
        if "usage" in chunk and chunk["usage"]:
            server_usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        # Track reasoning/thinking tokens
        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            if first_any_token_at is None:
                first_any_token_at = time.perf_counter()
            thinking_chunks += 1
            thinking_text.append(reasoning)

        # Track visible content tokens
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

    # Time spent in thinking vs content phases
    thinking_time = None
    if thinking_chunks > 0 and first_content_token_at and first_any_token_at:
        thinking_time = first_content_token_at - first_any_token_at

    # Prefer server-reported completion tokens if available
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
        "thinking_time_s": round(thinking_time, 3) if thinking_time else None,
        "generation_time_s": round(gen_time, 3),
        "tokens_per_second": round(tps, 2),
        "server_usage": server_usage,
        "response_preview": "".join(output_text)[:300],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results):
    """Pretty-print a comparison table of all benchmark results."""
    if not results:
        return

    # Header
    cols = [
        ("Model", 45),
        ("Tok/s", 8),
        ("TTFT(s)", 8),
        ("Tokens", 8),
        ("GenTime(s)", 11),
        ("Total(s)", 9),
    ]
    header = " | ".join(name.ljust(w) for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)

    print(f"\n{'=' * len(header)}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    # Sort by tokens/sec descending
    for r in sorted(results, key=lambda x: x["avg_tokens_per_second"], reverse=True):
        model_short = r["model"]
        if len(model_short) > 45:
            model_short = "..." + model_short[-42:]
        row = [
            model_short.ljust(45),
            f"{r['avg_tokens_per_second']:>7.1f}".ljust(8),
            f"{r['avg_ttft_s']:>7.3f}".ljust(8) if r["avg_ttft_s"] is not None else "   N/A ".ljust(8),
            f"{r['avg_tokens']:>7.0f}".ljust(8),
            f"{r['avg_gen_time_s']:>10.2f}".ljust(11),
            f"{r['avg_total_time_s']:>8.2f}".ljust(9),
        ]
        print(" | ".join(row))

    print(f"{'=' * len(header)}\n")


def run_benchmarks(args):
    base_url = f"http://localhost:{args.port}"

    # Check connectivity
    try:
        api_get(base_url, "/api/v1/models")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to LM Studio at {base_url}")
        print("Make sure LM Studio is running and the local server is started.")
        sys.exit(1)

    # Resolve model list
    all_models = discover_models(base_url)
    model_map = {get_model_id(m): m for m in all_models}

    if args.models:
        model_ids = args.models
    else:
        model_ids = list(model_map.keys())
        if not model_ids:
            print("No models found. Download some models in LM Studio first.")
            sys.exit(1)
        print(f"Discovered {len(model_ids)} LLM model(s):")
        for mid in model_ids:
            m = model_map.get(mid, {})
            q = m.get("quantization", {}).get("name", "?")
            p = m.get("params_string", "?")
            loaded = " [loaded]" if is_model_loaded(m) else ""
            print(f"  - {mid} ({p}, {q}){loaded}")

    prompt = args.prompt
    print(f"\nPrompt ({len(prompt)} chars): {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"Max tokens: {args.max_tokens} | Runs per model: {args.runs} | Context length: {args.context_length}")

    all_results = []

    for i, model_id in enumerate(model_ids, 1):
        print(f"\n{'-' * 60}")
        print(f"[{i}/{len(model_ids)}] {model_id}")
        print(f"{'-' * 60}")

        # Load
        was_already_loaded = is_model_loaded(model_map.get(model_id, {}))
        if was_already_loaded:
            print(f"  Already loaded — skipping load step")
        else:
            print(f"  Loading (context_length={args.context_length})...")
            t0 = time.perf_counter()
            try:
                load_resp = load_model(base_url, model_id, args.context_length)
                load_time = time.perf_counter() - t0
                print(f"  Loaded in {load_time:.1f}s")
            except requests.HTTPError as e:
                print(f"  WARN: Load returned {e.response.status_code} — may already be loaded, continuing...")
            except Exception as e:
                print(f"  ERROR loading model: {e} — skipping")
                continue

        # Benchmark runs
        run_results = []
        for run_num in range(1, args.runs + 1):
            label = f"  Run {run_num}/{args.runs}" if args.runs > 1 else "  Generating"
            print(f"{label}...", end="", flush=True)
            try:
                result = benchmark_streaming(base_url, model_id, prompt, args.max_tokens)
                run_results.append(result)
                think_info = ""
                if result["thinking_tokens"] > 0:
                    think_info = (
                        f" | Thinking: {result['thinking_tokens']} tok"
                        f" ({result['thinking_time_s']}s)"
                    )
                print(
                    f" {result['tokens_per_second']:.1f} tok/s | "
                    f"{result['token_count']} total tokens "
                    f"({result['content_tokens']} content){think_info} | "
                    f"TTFT {result['time_to_first_token_s']}s | "
                    f"Total {result['total_time_s']}s"
                )
            except Exception as e:
                print(f" ERROR: {e}")

        if run_results:
            n = len(run_results)
            avg_tps = sum(r["tokens_per_second"] for r in run_results) / n
            ttft_vals = [r["time_to_first_token_s"] for r in run_results if r["time_to_first_token_s"] is not None]
            avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else None
            avg_tokens = sum(r["token_count"] for r in run_results) / n
            avg_gen = sum(r["generation_time_s"] for r in run_results) / n
            avg_total = sum(r["total_time_s"] for r in run_results) / n

            all_results.append({
                "model": model_id,
                "avg_tokens_per_second": round(avg_tps, 2),
                "avg_ttft_s": round(avg_ttft, 3) if avg_ttft is not None else None,
                "avg_tokens": round(avg_tokens, 1),
                "avg_gen_time_s": round(avg_gen, 3),
                "avg_total_time_s": round(avg_total, 3),
                "runs": run_results,
            })

        # Unload (skip if it was already loaded before we started)
        if was_already_loaded:
            print(f"  Was pre-loaded — leaving in memory")
        else:
            print(f"  Unloading...")
            try:
                unload_model(base_url, model_id)
                print(f"  Unloaded.")
            except Exception as e:
                print(f"  WARN: Unload failed ({e}), continuing...")

    # Summary table
    print_table(all_results)

    # Dump raw JSON results
    outfile = "benchmark_results.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs via the LM Studio REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="The prompt to send to each model (default: coding task)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Specific model ID(s) to benchmark. If omitted, benchmarks all discovered models.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens to generate per run (default: 512)",
    )
    parser.add_argument(
        "--context-length", type=int, default=4096,
        help="Context length to use when loading models (default: 4096)",
    )
    parser.add_argument(
        "--port", type=int, default=1234,
        help="LM Studio server port (default: 1234)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of benchmark runs per model for averaging (default: 1)",
    )
    run_benchmarks(parser.parse_args())


if __name__ == "__main__":
    main()
