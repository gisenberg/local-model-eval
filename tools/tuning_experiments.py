#!/usr/bin/env python3
"""Tuning experiments: parallel sequences, eval batch size, KV cache quant."""

import json
import time
import requests

BASE_URL = "http://localhost:1234"

# Simple prompt — short enough that TTFT differences are visible,
# long enough output to measure sustained tok/s
PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted list. "
    "Include type hints and a docstring."
)
MAX_TOKENS = 1024

MODELS = [
    "google/gemma-4-26b-a4b",
    "qwen/qwen3.5-35b-a3b",
    "nvidia/nemotron-3-nano-4b",
]

# Experiment configurations: each is a dict of load params + a label
EXPERIMENTS = [
    # Baseline
    {"label": "baseline (parallel=4, batch=512, kv=f16)",
     "load": {"context_length": 32768, "flash_attention": True}},

    # Parallel 1 vs 4
    {"label": "parallel=1, batch=512, kv=f16",
     "load": {"context_length": 32768, "flash_attention": True, "parallel": 1}},

    # Eval batch size sweep
    {"label": "parallel=1, batch=1024, kv=f16",
     "load": {"context_length": 32768, "flash_attention": True, "parallel": 1,
              "eval_batch_size": 1024}},
    {"label": "parallel=1, batch=2048, kv=f16",
     "load": {"context_length": 32768, "flash_attention": True, "parallel": 1,
              "eval_batch_size": 2048}},

    # KV cache quantization (parallel=1 to isolate the effect)
    {"label": "parallel=1, batch=512, kv=q8_0",
     "load": {"context_length": 32768, "flash_attention": True, "parallel": 1,
              "cache_type_k": "q8_0", "cache_type_v": "q8_0"}},
    {"label": "parallel=1, batch=512, kv=q4_0",
     "load": {"context_length": 32768, "flash_attention": True, "parallel": 1,
              "cache_type_k": "q4_0", "cache_type_v": "q4_0"}},
]


def stream_benchmark(model_id):
    """Run streaming benchmark, return metrics dict."""
    start = time.perf_counter()
    first_tok = None
    content_chunks = 0
    thinking_chunks = 0

    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

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
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("reasoning_content"):
                if first_tok is None:
                    first_tok = time.perf_counter()
                thinking_chunks += 1
            if delta.get("content"):
                if first_tok is None:
                    first_tok = time.perf_counter()
                content_chunks += 1
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    end = time.perf_counter()
    total = end - start
    ttft = (first_tok - start) if first_tok else None
    gen_time = (end - first_tok) if first_tok else total
    total_tokens = content_chunks + thinking_chunks
    tps = total_tokens / gen_time if gen_time > 0 else 0

    return {
        "tokens": total_tokens,
        "content_tokens": content_chunks,
        "thinking_tokens": thinking_chunks,
        "tok_per_sec": round(tps, 1),
        "ttft_s": round(ttft, 3) if ttft else None,
        "gen_time_s": round(gen_time, 2),
        "total_time_s": round(total, 2),
    }


def load_model(model_id, load_params):
    """Load model, return (success, load_time, response)."""
    body = {"model": model_id, **load_params}
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/models/load", json=body, timeout=600
        )
        load_time = time.perf_counter() - t0
        if resp.status_code == 200:
            return True, load_time, resp.json()
        else:
            return False, load_time, resp.text
    except Exception as e:
        return False, time.perf_counter() - t0, str(e)


def unload_model(model_id):
    try:
        requests.post(
            f"{BASE_URL}/api/v1/models/unload",
            json={"instance_id": model_id},
            timeout=60,
        )
    except Exception:
        pass


# ── Run all experiments ──────────────────────────────────────────────
results = []
total_runs = len(MODELS) * len(EXPERIMENTS)
run_num = 0

for model_id in MODELS:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_id}")
    print(f"{'=' * 70}")

    for exp in EXPERIMENTS:
        run_num += 1
        label = exp["label"]
        load_params = exp["load"]
        print(f"\n  [{run_num}/{total_runs}] {label}")

        # Load
        print(f"    Loading...", end="", flush=True)
        ok, load_time, load_resp = load_model(model_id, load_params)
        if not ok:
            print(f" FAILED: {str(load_resp)[:100]}")
            results.append({
                "model": model_id, "experiment": label,
                "status": "load_failed", "error": str(load_resp)[:200],
            })
            continue
        print(f" {load_time:.1f}s")

        # Warmup run (discard)
        print(f"    Warmup...", end="", flush=True)
        try:
            stream_benchmark(model_id)
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}")
            unload_model(model_id)
            results.append({
                "model": model_id, "experiment": label,
                "status": "warmup_failed", "error": str(e)[:200],
            })
            continue

        # Measured run
        print(f"    Benchmark...", end="", flush=True)
        try:
            metrics = stream_benchmark(model_id)
            print(
                f" {metrics['tok_per_sec']} tok/s | "
                f"TTFT {metrics['ttft_s']}s | "
                f"{metrics['tokens']} tok ({metrics['content_tokens']}c + {metrics['thinking_tokens']}t) | "
                f"{metrics['total_time_s']}s"
            )
            results.append({
                "model": model_id, "experiment": label,
                "status": "ok", **metrics,
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "model": model_id, "experiment": label,
                "status": "bench_failed", "error": str(e)[:200],
            })

        # Unload
        unload_model(model_id)

# ── Summary table ────────────────────────────────────────────────────
print(f"\n{'=' * 120}")
print("TUNING EXPERIMENT RESULTS")
print(f"{'=' * 120}")
header = (
    f"{'Model':35s} | {'Experiment':42s} | {'Tok/s':>7s} | {'TTFT':>7s} | "
    f"{'Tokens':>6s} | {'Time':>6s} | {'Status'}"
)
print(header)
print("-" * 120)

for r in results:
    model_short = r["model"].split("/")[-1]
    if r["status"] == "ok":
        print(
            f"{model_short:35s} | {r['experiment']:42s} | "
            f"{r['tok_per_sec']:>7.1f} | {r['ttft_s']:>6.3f}s | "
            f"{r['tokens']:>6d} | {r['total_time_s']:>5.1f}s | OK"
        )
    else:
        print(
            f"{model_short:35s} | {r['experiment']:42s} | "
            f"{'---':>7s} | {'---':>7s} | {'---':>6s} | {'---':>6s} | "
            f"{r['status']}: {r.get('error','')[:40]}"
        )

print(f"{'=' * 120}")

with open("tuning_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Raw results saved to tuning_results.json")
