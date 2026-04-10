#!/usr/bin/env python3
"""
Re-measure 5090 throughput numbers using a Linux client to bypass the
~2.1s urllib3-on-Windows TTFT bug + ~24% decode throughput understatement.

This script is designed to run INSIDE WSL2. It launches the WSL2-built
TurboQuant llama-server for each model in MODELS, runs a 5-run streaming
throughput test, and collects the results into a JSON.

Models live on the Windows filesystem at /mnt/c/Users/gisen/.lmstudio/models/...
The WSL2 llama-server reads from there directly (model load is slower due to
NTFS bridge but per-token decode is unaffected once weights are in VRAM).
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time

import requests

LLAMA_SERVER = "/mnt/t/git/TheTom/llama-cpp-turboquant/build-wsl/bin/llama-server"
MODELS_DIR = "/mnt/c/Users/gisen/.lmstudio/models"
PORT = 8080
OUTPUT_DIR = "/mnt/t/git/local-model-eval/rebench_5090"

PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test."
)
MAX_TOKENS = 256
WARMUP_RUNS = 1
TIMED_RUNS = 5

MODELS = [
    {
        "key": "gemma26b-q6k",
        "name": "Gemma 4 26B-A4B Q6_K",
        "path": f"{MODELS_DIR}/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q6_K.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "gemma26b-q4km",
        "name": "Gemma 4 26B-A4B Q4_K_M",
        "path": f"{MODELS_DIR}/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "gemma31b-q4km",
        "name": "Gemma 4 31B-IT Q4_K_M",
        "path": f"{MODELS_DIR}/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "gemma31b-opus-q4km",
        "name": "Gemma 31B Opus-Distill Q4_K_M",
        "path": f"{MODELS_DIR}/TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF/gemma-4-31B-it-Claude-Opus-Distill.q4_k_m.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "qwopus27b-q6k",
        "name": "Qwopus 3.5 27B-v3 Q6_K",
        "path": f"{MODELS_DIR}/Jackrong/Qwopus3.5-27B-v3-GGUF/Qwopus3.5-27B-v3-Q6_K.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "harmonic27b-q4km",
        "name": "Harmonic 27B Q4_K_M",
        "path": f"{MODELS_DIR}/DJLougen/Harmonic-27B-GGUF/Harmonic-27B-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "qwen27b-opus-q4km",
        "name": "Qwen 3.5 27B Opus-Distilled Q4_K_M",
        "path": f"{MODELS_DIR}/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "qwen27b-q6k",
        "name": "Qwen 3.5 27B Q6_K (base)",
        "path": f"{MODELS_DIR}/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q6_K.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
    {
        "key": "qwen35b-a3b",
        "name": "Qwen 3.5 35B-A3B Q4_K_M",
        "path": f"{MODELS_DIR}/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "ctk": "turbo4", "ctv": "turbo4", "rea": "off",
    },
]


def start_server(model_cfg):
    cmd = [
        LLAMA_SERVER,
        "-m", model_cfg["path"],
        "--port", str(PORT),
        "-c", "32768",
        "-ngl", "99",
        "-fa", "on",
        "-ctk", model_cfg["ctk"],
        "-ctv", model_cfg["ctv"],
        "-np", "1",
        "-rea", model_cfg["rea"],
    ]
    env = {"PATH": "/usr/local/cuda/bin:/usr/bin:/bin", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


def measure_streaming(prompt=PROMPT, max_tokens=MAX_TOKENS):
    """Send a streaming request and measure TTFT + decode rate."""
    t_send = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    first_token_at = None
    last_token_at = None
    n_tokens = 0
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
        if delta.get("content") or delta.get("reasoning_content"):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            n_tokens += 1

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
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "n_tokens": n_tokens,
        "total_s": t_end - t_send,
    }


def benchmark_model(model_cfg):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_cfg['name']}")
    print(f"{'=' * 70}")

    if not os.path.isfile(model_cfg["path"]):
        print(f"  SKIP: model file not found at {model_cfg['path']}")
        return None

    proc = start_server(model_cfg)
    print("  Starting server...", end="", flush=True)
    if not wait_for_server(timeout=300):
        print(" TIMEOUT")
        stop_server(proc)
        return {"model": model_cfg["name"], "error": "server start timeout"}
    print(" ready")

    # Capture VRAM
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        vram_mb = int(out.strip().split("\n")[0])
    except Exception:
        vram_mb = None

    # Warmup
    for i in range(WARMUP_RUNS):
        try:
            r = measure_streaming()
            print(f"  warmup {i+1}/{WARMUP_RUNS}: ttft={r['ttft_s']:.3f}s decode={r['decode_tps']:.1f} tok/s")
        except Exception as e:
            print(f"  warmup {i+1}/{WARMUP_RUNS}: ERROR {e}")

    # Timed runs
    runs = []
    for i in range(TIMED_RUNS):
        try:
            r = measure_streaming()
            print(f"  run {i+1}/{TIMED_RUNS}:    ttft={r['ttft_s']:.3f}s decode={r['decode_tps']:.1f} tok/s ({r['n_tokens']} tok)")
            runs.append(r)
        except Exception as e:
            print(f"  run {i+1}/{TIMED_RUNS}:    ERROR {e}")

    stop_server(proc)
    time.sleep(2)

    if not runs:
        return {"model": model_cfg["name"], "error": "no successful runs"}

    ttfts = [r["ttft_s"] for r in runs]
    decodes = [r["decode_tps"] for r in runs if r["decode_tps"] is not None]

    summary = {
        "model": model_cfg["name"],
        "key": model_cfg["key"],
        "vram_mb": vram_mb,
        "kv_config": f"{model_cfg['ctk']}/{model_cfg['ctv']}",
        "thinking": model_cfg["rea"],
        "runs": runs,
        "ttft_mean": round(statistics.mean(ttfts), 4),
        "ttft_median": round(statistics.median(ttfts), 4),
        "ttft_min": round(min(ttfts), 4),
        "ttft_max": round(max(ttfts), 4),
        "decode_mean": round(statistics.mean(decodes), 1) if decodes else None,
        "decode_median": round(statistics.median(decodes), 1) if decodes else None,
    }
    print(f"  -> TTFT mean={summary['ttft_mean']:.3f}s | decode mean={summary['decode_mean']:.1f} tok/s | VRAM={vram_mb}MB")
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== 5090 Re-Benchmark via WSL2 Linux Client ===")
    print(f"Server: WSL2-built llama-server (TurboQuant fork, same source as Windows build)")
    print(f"Client: requests on Linux (no urllib3-on-Windows overhead)")
    print(f"Models: {len(MODELS)}")

    results = []
    for m in MODELS:
        r = benchmark_model(m)
        if r:
            results.append(r)
            with open(f"{OUTPUT_DIR}/results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Final summary table
    print(f"\n{'=' * 90}")
    print("REBENCHMARK SUMMARY (WSL2 client, Linux llama-server, same TurboQuant source)")
    print("=" * 90)
    print(f"{'Model':40s} | {'TTFT':>8s} | {'Decode':>10s} | {'VRAM':>8s}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['model']:40s} | ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:40s} | "
            f"{r['ttft_mean']*1000:>6.0f} ms | "
            f"{r['decode_mean']:>7.1f} tok/s | "
            f"{r.get('vram_mb', '?'):>5} MB"
        )
    print("=" * 90)
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
