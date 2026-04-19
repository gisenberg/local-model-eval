#!/usr/bin/env python3
"""
Minimal throughput benchmark for RTX Pro 6000 BF16 runs via llama.cpp Vulkan.

Uses only Python stdlib (no pip on target box). Measures TTFT + decode tok/s
across N streaming runs. Captures VRAM via nvidia-smi. Writes JSON.

Invocation:
  python3 rtxpro6000_bench.py <model_key>

Model configs below. -c (context) is per model; -ngl 99 for full GPU offload.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

BACKEND = os.environ.get("LLAMA_BACKEND", "vulkan")
if BACKEND == "cuda":
    LLAMA_DIR = "/home/gisenberg/llama-build/src/build/bin"
    _LD_EXTRA = ":/home/gisenberg/.micromamba/envs/cuda/lib"
else:
    LLAMA_DIR = "/home/gisenberg/llama/llama-b8826"
    _LD_EXTRA = ""
MODELS_ROOT = "/home/gisenberg/models"
OUTPUT_ROOT = f"/home/gisenberg/git/gisenberg/local-model-eval/experiments/rtxpro6000_bench_{BACKEND}"
PORT = 8080

PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test."
)
MAX_TOKENS = 256
WARMUP_RUNS = 1
TIMED_RUNS = 5

MODELS = {
    "gemma4-31b-bf16": {
        "name": "Gemma-4-31B-it BF16",
        "path": f"{MODELS_ROOT}/gemma-4-31b-it-bf16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "qwen36-35b-a3b-bf16": {
        "name": "Qwen3.6-35B-A3B BF16",
        "path": f"{MODELS_ROOT}/qwen36-35b-a3b-bf16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "gemopus-31b-bf16": {
        "name": "Gemopus-4-31B-it BF16",
        "path": f"{MODELS_ROOT}/gemopus-4-31b-bf16/Gemopus-4-31B-it-BF16.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "gemma4-31b-q8": {
        "name": "Gemma-4-31B-it Q8_0",
        "path": f"{MODELS_ROOT}/gemma-4-31b-it-q8/gemma-4-31B-it-Q8_0.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "qwen36-35b-a3b-q8": {
        "name": "Qwen3.6-35B-A3B Q8_0",
        "path": f"{MODELS_ROOT}/qwen36-35b-a3b-q8/Qwen3.6-35B-A3B-Q8_0.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "gemopus-31b-q8": {
        "name": "Gemopus-4-31B-it Q8_0",
        "path": f"{MODELS_ROOT}/gemopus-4-31b-q8/Gemopus-4-31B-it-Q8_0.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "gpt-oss-120b-q8": {
        "name": "gpt-oss-120b Q8_0",
        "path": f"{MODELS_ROOT}/gpt-oss-120b-q8/gpt-oss-120b-Q8_0-00001-of-00002.gguf",
        "ctx": 131072,
        "extra": [],
    },
    "qwen3-coder-next-q6": {
        "name": "Qwen3-Coder-Next Q6_K",
        "path": f"{MODELS_ROOT}/qwen3-coder-next-q6/Qwen3-Coder-Next-Q6_K-00001-of-00003.gguf",
        "ctx": 262144,
        "extra": [],
    },
    "qwen36-opus-distill-q8": {
        "name": "Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled Q8_0",
        "path": f"{MODELS_ROOT}/qwen36-opus-distill-q8/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q8_0.gguf",
        "ctx": 262144,
        "extra": [],
    },
}


def start_server(model_cfg, ctx_override=None):
    ctx = ctx_override or model_cfg["ctx"]
    cmd = [
        f"{LLAMA_DIR}/llama-server",
        "-m", model_cfg["path"],
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-c", str(ctx),
        "-ngl", "99",
        "-fa", "on",
        "-np", "1",
        "--no-mmap",  # Force full load to VRAM (not OS page-cache backed)
    ] + model_cfg.get("extra", [])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_DIR + _LD_EXTRA
    log_path = f"/tmp/llama-server-{model_cfg['key']}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_path


def wait_for_server(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def measure_streaming(prompt=PROMPT, max_tokens=MAX_TOKENS):
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_send = time.perf_counter()
    first_token_at = None
    last_token_at = None
    n_tokens = 0
    server_usage = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    server_usage = obj["usage"]
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
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
        "ttft_s": round(ttft, 4),
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "n_tokens": n_tokens,
        "total_s": round(t_end - t_send, 3),
    }


def benchmark(model_key, ctx_override=None):
    cfg = dict(MODELS[model_key])
    cfg["key"] = model_key
    if not os.path.isfile(cfg["path"]):
        print(f"MISSING: {cfg['path']}", file=sys.stderr)
        return None

    print(f"{'='*70}\n{cfg['name']}\n{'='*70}")
    print(f"ctx={ctx_override or cfg['ctx']}, path={cfg['path']}")

    t_load_start = time.perf_counter()
    proc, log = start_server(cfg, ctx_override=ctx_override)
    print(f"Waiting for server (log: {log})...", flush=True)
    ok = wait_for_server(timeout=900)
    t_loaded = time.perf_counter() - t_load_start
    if not ok:
        print("SERVER TIMEOUT")
        stop_server(proc)
        return {"model": cfg["name"], "error": "server start timeout"}
    print(f"Ready after {t_loaded:.1f}s")

    vram_mb = vram_used_mb()
    print(f"VRAM used after load: {vram_mb} MB")

    for i in range(WARMUP_RUNS):
        try:
            r = measure_streaming()
            print(f"  warmup {i+1}/{WARMUP_RUNS}: ttft={r['ttft_s']}s decode={r['decode_tps']} tok/s  ({r['n_tokens']} tok)")
        except Exception as e:
            print(f"  warmup {i+1}: ERROR {e}")

    runs = []
    for i in range(TIMED_RUNS):
        try:
            r = measure_streaming()
            runs.append(r)
            print(f"  run {i+1}/{TIMED_RUNS}:   ttft={r['ttft_s']}s decode={r['decode_tps']} tok/s  ({r['n_tokens']} tok)")
        except Exception as e:
            print(f"  run {i+1}: ERROR {e}")

    stop_server(proc)

    if not runs:
        return {"model": cfg["name"], "error": "no successful runs"}

    ttfts = [r["ttft_s"] for r in runs]
    decodes = [r["decode_tps"] for r in runs if r["decode_tps"] is not None]
    summary = {
        "model": cfg["name"],
        "key": model_key,
        "path": cfg["path"],
        "ctx": ctx_override or cfg["ctx"],
        "vram_mb": vram_mb,
        "load_seconds": round(t_loaded, 1),
        "backend": BACKEND,
        "runs": runs,
        "ttft_mean": round(statistics.mean(ttfts), 4),
        "ttft_median": round(statistics.median(ttfts), 4),
        "decode_mean": round(statistics.mean(decodes), 2) if decodes else None,
        "decode_median": round(statistics.median(decodes), 2) if decodes else None,
    }
    print(f"SUMMARY: VRAM={vram_mb}MB | TTFT={summary['ttft_mean']*1000:.0f}ms | decode={summary['decode_mean']} tok/s")
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: rtxpro6000_bench.py <model_key> [ctx]")
        print("Keys:", " ".join(MODELS.keys()))
        sys.exit(1)
    key = sys.argv[1]
    ctx_override = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if key not in MODELS:
        print(f"Unknown key {key!r}. Available: {list(MODELS)}")
        sys.exit(1)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    result = benchmark(key, ctx_override=ctx_override)
    if result is None:
        sys.exit(2)
    out_path = f"{OUTPUT_ROOT}/{key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
