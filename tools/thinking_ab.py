#!/usr/bin/env python3
"""A/B test: thinking ON vs OFF for each model."""

import json
import time
import requests

BASE_URL = "http://localhost:1234"

PROMPT = "Explain the difference between a mutex and a semaphore. Give a short Python example of each."

MODELS = [
    "nvidia/nemotron-3-nano-4b",
    "qwen/qwen3.5-9b",
    "nvidia/nemotron-3-nano",
    "qwen/qwen3.5-35b-a3b",
]

# Thinking-off mechanisms:
# - Qwen 3.5: /no_think prefix in user message
# - Nemotron: "detailed thinking off" in system prompt
# We'll use both to cover all models

CONFIGS = [
    {
        "label": "thinking ON",
        "messages": [
            {"role": "user", "content": PROMPT},
        ],
    },
    {
        "label": "thinking OFF",
        "messages": [
            {"role": "system", "content": "detailed thinking off"},
            {"role": "user", "content": "/no_think " + PROMPT},
        ],
    },
]

results = []

for model_id in MODELS:
    print(f"\n{'=' * 60}")
    print(f"{model_id}")
    print(f"{'=' * 60}")

    # Load
    print("  Loading...", end="", flush=True)
    try:
        requests.post(f"{BASE_URL}/api/v1/models/load", json={
            "model": model_id, "context_length": 32768, "flash_attention": True,
        }, timeout=600)
        print(" done")
    except Exception as e:
        print(f" warn: {e}")

    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"  {label}...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
                "model": model_id,
                "messages": cfg["messages"],
                "max_tokens": 4096,
                "temperature": 0,
            }, timeout=300)
            elapsed = time.perf_counter() - t0
            data = resp.json()
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            finish = data["choices"][0].get("finish_reason", "?")

            comp_tokens = usage.get("completion_tokens", 0)
            thinking_chars = len(msg.get("reasoning_content", "") or "")
            content_chars = len(msg.get("content", "") or "")
            tps = comp_tokens / elapsed if elapsed > 0 else 0

            print(
                f" {tps:.1f} tok/s | {comp_tokens} tok | {elapsed:.1f}s | "
                f"thinking: {thinking_chars} chars | content: {content_chars} chars | "
                f"finish={finish}"
            )

            results.append({
                "model": model_id,
                "mode": label,
                "tokens_per_second": round(tps, 1),
                "total_tokens": comp_tokens,
                "elapsed_s": round(elapsed, 1),
                "thinking_chars": thinking_chars,
                "content_chars": content_chars,
                "finish_reason": finish,
            })

        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "model": model_id,
                "mode": label,
                "error": str(e),
            })

    # Unload
    print("  Unloading...", end="", flush=True)
    try:
        requests.post(f"{BASE_URL}/api/v1/models/unload",
                       json={"instance_id": model_id}, timeout=60)
        print(" done")
    except:
        print(" warn")

# Summary table
print(f"\n{'=' * 100}")
print("THINKING ON vs OFF COMPARISON")
print(f"{'=' * 100}")
print(f"{'Model':40s} | {'Mode':14s} | {'Tok/s':>7s} | {'Tokens':>7s} | {'Time':>7s} | {'Think':>8s} | {'Content':>8s}")
print("-" * 100)

for r in results:
    if "error" in r:
        print(f"{r['model']:40s} | {r['mode']:14s} | ERROR: {r['error']}")
    else:
        print(
            f"{r['model']:40s} | {r['mode']:14s} | "
            f"{r['tokens_per_second']:>7.1f} | {r['total_tokens']:>7d} | "
            f"{r['elapsed_s']:>6.1f}s | {r['thinking_chars']:>7d}c | {r['content_chars']:>7d}c"
        )

with open("thinking_ab_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw results saved to thinking_ab_results.json")
