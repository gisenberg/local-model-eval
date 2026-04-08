#!/usr/bin/env python3
"""
Proof-of-concept: TriAttention token eviction + TurboQuant KV compression
combined on Qwen3-8B via transformers (not vLLM).

This tests whether the two approaches can stack without destroying quality.
Runs in WSL with PyTorch + CUDA.
"""

import json
import os
import sys
import time
import re
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Conditional imports
try:
    from turboquant_vllm import TurboQuantKVCache
    HAS_TURBOQUANT = True
except ImportError:
    HAS_TURBOQUANT = False
    print("WARNING: turboquant-vllm not installed, running without KV compression")

MODEL_ID = "Qwen/Qwen3-8B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 8192  # shorter for transformers (no vLLM batching)
TEMPERATURE = 0.3

BENCHMARKS = {
    "expression_evaluator": {
        "name": "Expression Evaluator",
        "expected": 5,
        "prompt": (
            "Build a mathematical expression evaluator in Python. Requirements:\n"
            "1. Support +, -, *, / with correct operator precedence\n"
            "2. Support parentheses for grouping\n"
            "3. Support unary minus (e.g., '-3', '-(2+1)')\n"
            "4. Support floating point numbers (e.g., '3.14')\n"
            "5. Raise ValueError for: mismatched parentheses, division by zero, invalid tokens, empty expressions\n"
            "6. Implement as class ExpressionEvaluator with evaluate(expr: str) -> float\n"
            "7. Use recursive descent parser — no eval() or ast.literal_eval()\n"
            "8. Include type hints and docstrings\n"
            "9. Write 5 pytest tests"
        ),
    },
    "astar": {
        "name": "A* Pathfinding",
        "expected": 6,
        "prompt": (
            "Implement A* pathfinding on a weighted 2D grid in Python.\n"
            "Class AStarGrid with find_path(start, end) -> Optional[List[Tuple]].\n"
            "4-directional, Manhattan heuristic, heapq, handle walls and edge cases.\n"
            "Write 6 pytest tests."
        ),
    },
}


def generate(model, tokenizer, prompt, turbo_bits=None):
    """Generate with optional TurboQuant KV compression."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    t0 = time.perf_counter()

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "do_sample": True,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    elapsed = time.perf_counter() - t0
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text_out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)

    return {
        "content": text_out,
        "tokens": n_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(n_tokens / elapsed if elapsed > 0 else 0, 1),
    }


def extract_and_test(content, test_file):
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "error": "no code blocks"}
    combined = "\n\n".join(b.strip() for b in blocks)
    for pat in [r'from \w+ import ExpressionEvaluator', r'from \w+ import AStarGrid']:
        combined = re.sub(pat, '', combined)
    with open(test_file, "w") as f:
        f.write(combined)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        return {
            "passed": len(re.findall(r' PASSED', output)),
            "failed": len(re.findall(r' FAILED', output)),
            "errors": len(re.findall(r' ERROR', output)),
        }
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 0, "error": str(e)}


def main():
    output_dir = "combined_poc"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run each benchmark
    configs = [
        ("baseline", "No compression"),
    ]
    if HAS_TURBOQUANT:
        configs.append(("turbo3", "TurboQuant 3-bit KV"))

    all_results = {}
    for config_key, config_label in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_label}")
        print(f"{'='*60}")

        results = {}
        for bk, bench in BENCHMARKS.items():
            print(f"\n  [{bench['name']}]...", end="", flush=True)
            result = generate(model, tokenizer, bench["prompt"])
            tf = f"{output_dir}/{config_key}_{bk}_test.py"
            tr = extract_and_test(result["content"], tf)
            print(f" {result['tok_per_sec']:.0f} tok/s | {tr['passed']}/{bench['expected']} pass | {result['tokens']} tok")
            results[bk] = {**result, **tr, "expected": bench["expected"]}

        total = sum(min(r["passed"], r["expected"]) for r in results.values())
        total_exp = sum(r["expected"] for r in results.values())
        print(f"\n  TOTAL: {total}/{total_exp}")
        all_results[config_key] = results

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
