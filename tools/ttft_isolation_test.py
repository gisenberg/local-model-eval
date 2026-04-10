#!/usr/bin/env python3
"""
TTFT isolation test: measure where the ~1.8s Windows TTFT overhead comes from.

Tests the same model + same llama.cpp binary path on different OS/client combinations
to isolate which layer of the stack is responsible.

Run this script from BOTH Windows and WSL2 (after pointing it at the appropriate
server). Compare the output across the four-cell test matrix:

  | Server                 | Windows client | WSL2 client |
  |------------------------|----------------|-------------|
  | Windows llama-server   | Cell A         | Cell C      |
  | WSL2 llama-server      | Cell D         | Cell B      |

Cell A is our existing 5090 baseline (~2.3s TTFT).
Cell B is the clean Linux baseline we want to measure.
Cells C and D isolate which side of the boundary contributes the overhead.
"""

import argparse
import json
import statistics
import time

import requests

DEFAULT_PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test."
)


def measure_ttft(base_url, model_id, prompt, max_tokens=64):
    """Send a streaming request and measure time-to-first-token."""
    t_send = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    first_token_at = None
    last_token_at = None
    n_tokens = 0
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
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content") or delta.get("reasoning_content"):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            n_tokens += 1

    t_end = time.perf_counter()
    ttft = (first_token_at - t_send) if first_token_at else None
    decode_time = (last_token_at - first_token_at) if (first_token_at and last_token_at) else None
    decode_tps = (n_tokens / decode_time) if (decode_time and decode_time > 0) else None
    return {
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "n_tokens": n_tokens,
        "total_s": t_end - t_send,
    }


def main():
    p = argparse.ArgumentParser(description="TTFT isolation test")
    p.add_argument("--url", default="http://localhost:8080", help="llama-server base URL")
    p.add_argument("--model", default="local-model", help="model id (matches what llama-server reports)")
    p.add_argument("--label", default="unlabeled", help="label this run for the comparison table")
    p.add_argument("--runs", type=int, default=5, help="number of timed requests")
    p.add_argument("--warmup", type=int, default=1, help="number of warmup requests before timing")
    args = p.parse_args()

    print(f"=== TTFT Isolation Test: {args.label} ===")
    print(f"Server: {args.url}")
    print(f"Warmup: {args.warmup} req | Timed: {args.runs} req")
    print()

    # Warmup runs (not counted)
    for i in range(args.warmup):
        print(f"  warmup {i+1}/{args.warmup}...", end="", flush=True)
        try:
            r = measure_ttft(args.url, args.model, DEFAULT_PROMPT)
            print(f" ttft={r['ttft_s']:.3f}s")
        except Exception as e:
            print(f" ERROR: {e}")
            return

    # Timed runs
    ttfts = []
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs}...", end="", flush=True)
        try:
            r = measure_ttft(args.url, args.model, DEFAULT_PROMPT)
            print(f" ttft={r['ttft_s']:.3f}s | decode={r['decode_tps']:.1f} tok/s | total={r['total_s']:.2f}s")
            ttfts.append(r["ttft_s"])
        except Exception as e:
            print(f" ERROR: {e}")

    if ttfts:
        print()
        print(f"=== Results: {args.label} ===")
        print(f"  TTFT mean:   {statistics.mean(ttfts):.3f}s")
        print(f"  TTFT median: {statistics.median(ttfts):.3f}s")
        print(f"  TTFT min:    {min(ttfts):.3f}s")
        print(f"  TTFT max:    {max(ttfts):.3f}s")
        if len(ttfts) > 1:
            print(f"  TTFT stdev:  {statistics.stdev(ttfts):.3f}s")


if __name__ == "__main__":
    main()
