#!/usr/bin/env python3
"""Run the same coding prompt through all models and save full outputs for comparison."""

import json
import os
import time
import requests

BASE_URL = "http://localhost:1234"

PROMPT = (
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
)

# Per-model config
# context_length = 32K for all (enough for benchmark, avoids OOM on large models)
# The "practical_max_ctx" column is the theoretical max at 32GB VRAM, for the table only
MODELS = [
    {"id": "nvidia/nemotron-3-nano-4b",  "context_length": 32768, "practical_max_ctx": "262K", "note": "4.2GB weights, hybrid 4 attn layers"},
    {"id": "qwen/qwen3.5-9b",           "context_length": 32768, "practical_max_ctx": "~130K", "note": "10.4GB weights, dense 32 layers"},
    {"id": "nvidia/nemotron-3-nano",     "context_length": 32768, "practical_max_ctx": "~128K", "note": "24.5GB weights, hybrid ~23 attn layers"},
    {"id": "qwen/qwen3.5-35b-a3b",      "context_length": 32768, "practical_max_ctx": "~256K", "note": "22.1GB weights, hybrid 10 attn layers"},
]

OUTPUT_DIR = "compare4"
MAX_TOKENS = 16384

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch model metadata to show context sizes
print("Fetching model info...")
all_model_info = requests.get(f"{BASE_URL}/api/v1/models", timeout=30).json()
model_meta = {m["key"]: m for m in all_model_info.get("models", []) if m.get("type") == "llm"}

for i, model_cfg in enumerate(MODELS, 1):
    model_id = model_cfg["id"]
    ctx_len = model_cfg["context_length"]
    safe_name = model_id.replace("/", "_")
    meta = model_meta.get(model_id, {})
    max_ctx = meta.get("max_context_length", "?")
    print(f"\n[{i}/{len(MODELS)}] {model_id}")
    print(f"  Trained max: {max_ctx} | Loading with: {ctx_len} | {model_cfg['note']}")

    # Load
    print("  Loading...", end="", flush=True)
    try:
        requests.post(f"{BASE_URL}/api/v1/models/load", json={
            "model": model_id, "context_length": ctx_len, "flash_attention": True,
        }, timeout=600)
        print(" done")
    except Exception as e:
        print(f" warn: {e}, continuing")

    # Generate (non-streaming to get complete response)
    print(f"  Generating (max_tokens={MAX_TOKENS}, ctx={ctx_len})...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }, timeout=600)
        elapsed = time.perf_counter() - t0
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        usage = data.get("usage", {})
        finish = data["choices"][0].get("finish_reason", "?")

        thinking_tokens = usage.get("completion_tokens", 0) - len(content.split())  # rough estimate
        print(
            f" done in {elapsed:.1f}s | "
            f"{usage.get('completion_tokens', '?')} tokens | "
            f"finish_reason={finish}"
        )
        if finish == "length":
            print("  WARNING: output was truncated (hit max_tokens limit)")

        # Save output
        with open(f"{OUTPUT_DIR}/{safe_name}.md", "w", encoding="utf-8") as f:
            f.write(f"# {model_id}\n\n")
            if reasoning:
                f.write(f"## Thinking ({len(reasoning)} chars)\n\n")
                f.write(f"{reasoning}\n\n")
            f.write(f"## Output\n\n{content}\n")

        # Save raw JSON
        with open(f"{OUTPUT_DIR}/{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": model_id,
                "context_length": ctx_len,
                "elapsed_s": round(elapsed, 2),
                "finish_reason": finish,
                "usage": usage,
                "reasoning": reasoning,
                "content": content,
            }, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f" ERROR: {e}")

    # Unload
    print("  Unloading...", end="", flush=True)
    try:
        requests.post(f"{BASE_URL}/api/v1/models/unload",
                       json={"instance_id": model_id}, timeout=60)
        print(" done")
    except:
        print(" warn: failed")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
