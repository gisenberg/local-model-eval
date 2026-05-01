#!/usr/bin/env python3
"""
Coding benchmark for nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning, served via
vLLM in BF16 and NVFP4. Same 4 coding tasks / 3 runs / pytest scoring as
tools/nvfp4_qwen36_27b_bench.py — sampling defaults differ.

The model card ships explicit sampling presets; we honor them rather than
reusing the Qwen3.6 defaults that Ling-2.6 was tripped up by:

  --preset instruct  (default, non-thinking)
      temperature=0.2, top_k=1
      enable_thinking=False, max_tokens=4096

  --preset thinking
      temperature=0.6, top_p=0.95
      enable_thinking=True, max_tokens=20480

Assumes `vllm serve` is already running with --reasoning-parser nemotron_v3.
Example serve commands (BF16 then NVFP4):

  vllm serve ~/models-vllm/nemotron3-30b-bf16 \\
      --served-model-name nemotron3-omni-bf16 --port 8090 \\
      --max-model-len 32768 --gpu-memory-utilization 0.90 \\
      --trust-remote-code --reasoning-parser nemotron_v3

  vllm serve ~/models-vllm/nemotron3-30b-nvfp4 \\
      --served-model-name nemotron3-omni-nvfp4 --port 8090 \\
      --quantization modelopt_fp4 \\
      --max-model-len 32768 --gpu-memory-utilization 0.90 \\
      --trust-remote-code --reasoning-parser nemotron_v3
"""

import argparse, json, os, re, subprocess, sys, time
import requests

RUNS = 3

PRESETS = {
    "instruct": {
        "temperature": 0.2,
        "top_k": 1,
        "max_tokens": 4096,
        "enable_thinking": False,
    },
    "thinking": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 20480,
        "enable_thinking": True,
        "reasoning_budget": 16384,
        "grace_period": 1024,
    },
}

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


def run_inference(prompt, port, model, preset_cfg):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": preset_cfg["max_tokens"],
        "temperature": preset_cfg["temperature"],
        # Nemotron 3's reasoning gate; vLLM forwards through chat_template_kwargs.
        "chat_template_kwargs": {"enable_thinking": preset_cfg["enable_thinking"]},
    }
    if "top_p" in preset_cfg:
        body["top_p"] = preset_cfg["top_p"]
    if "top_k" in preset_cfg:
        body["top_k"] = preset_cfg["top_k"]
    # Reasoning-mode controls used by --reasoning-parser nemotron_v3:
    # reasoning_budget caps thinking tokens; grace_period extends past the budget
    # to let the model close the </think> tag cleanly.
    extra_body = {}
    for k in ("reasoning_budget", "grace_period"):
        if k in preset_cfg:
            extra_body[k] = preset_cfg[k]
    if extra_body:
        body.update(extra_body)

    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=body,
        timeout=900,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    d = resp.json()
    msg = d["choices"][0]["message"]
    u = d.get("usage", {})
    comp = u.get("completion_tokens", 0)
    return {
        "content": msg.get("content", ""),
        # vLLM 0.20+ surfaces reasoning under "reasoning"; older versions used
        # "reasoning_content". Read both to be safe.
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content", ""),
        "tokens": comp,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(comp / elapsed if elapsed > 0 else 0, 1),
        "finish_reason": d["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content, test_file):
    # Strip any reasoning markers that leaked into content. vLLM with
    # --reasoning-parser nemotron_v3 routes them to reasoning_content, but
    # belt-and-suspenders in case the parser misses a trailing fragment.
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0}
    combined = "\n\n".join(b.strip() for b in blocks)
    for cls in ["ExpressionEvaluator", "AStarGrid", "TTLCache", "StringProcessor", "BST", "Node"]:
        combined = re.sub(rf"from \w+ import {cls}\b.*\n", "", combined)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(combined)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
        )
        out = r.stdout + r.stderr
        return {
            "passed": len(re.findall(r" PASSED", out)),
            "failed": len(re.findall(r" FAILED", out)),
            "errors": len(re.findall(r" ERROR", out)),
        }
    except Exception:
        return {"passed": 0, "failed": 0, "errors": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="instruct")
    args = ap.parse_args()

    preset_cfg = PRESETS[args.preset]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Nemotron-3-Omni 30B-A3B coding bench ({args.served_name}) ===")
    print(f"Preset={args.preset} {preset_cfg}, {RUNS} runs, port={args.port}\n")

    all_results = {}
    for bk, bench in BENCHMARKS.items():
        runs = []
        for run in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run}/{RUNS}...", end="", flush=True)
            try:
                result = run_inference(bench["prompt"], args.port, args.served_name, preset_cfg)
                tf = f"{args.output_dir}/{bk}_run{run}_test.py"
                tr = extract_and_test(result["content"], tf)
                print(f" {result['tok_per_sec']:.0f} tok/s | {tr['passed']}/{bench['expected']} | {result['tokens']} tok")
                runs.append({**result, **tr})
            except Exception as e:
                print(f" ERROR: {e}")
                runs.append({"error": str(e)})

        valid = [r for r in runs if "passed" in r]
        if valid:
            best = max(r["passed"] for r in valid)
            avg = sum(r["passed"] for r in valid) / len(valid)
            all_results[bk] = {"best": best, "avg": round(avg, 1), "expected": bench["expected"], "runs": runs}
            print(f"  -> Best: {best}/{bench['expected']}, Avg: {avg:.1f}/{bench['expected']}\n")
        else:
            all_results[bk] = {"runs": runs, "error": "all runs failed"}

    total_best = sum(min(v.get("best", 0), v.get("expected", 0)) for v in all_results.values())
    total_avg = sum(min(v.get("avg", 0), v.get("expected", 0)) for v in all_results.values())
    total_exp = sum(v.get("expected", 0) for v in all_results.values())
    print(
        f"\nTOTAL: Best-of-{RUNS} = {total_best}/{total_exp} "
        f"({total_best/total_exp*100:.0f}%), Avg = {total_avg:.1f}/{total_exp}"
    )

    all_results["_meta"] = {
        "served_name": args.served_name,
        "port": args.port,
        "preset": args.preset,
        "preset_cfg": preset_cfg,
        "runs": RUNS,
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
