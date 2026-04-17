#!/usr/bin/env python3
"""
vLLM-backed throughput + coding bench. Mirrors rtxpro6000_bench.py's measurement
methodology so results are directly comparable to the llama.cpp CUDA numbers.

Usage:
  rtxpro6000_vllm_bench.py <model_key> [--coding]

Keys:
  gpt-oss-120b-vllm       openai/gpt-oss-120b (MXFP4 native)
  gemma-4-31b-nvfp4-vllm  LilaRest/gemma-4-31B-it-NVFP4-turbo
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/gisenberg/git/gisenberg/local-model-eval")
VLLM_BIN = os.environ.get("VLLM_BIN", "/home/gisenberg/.micromamba/envs/vllm/bin/vllm")
MODELS_ROOT = "/home/gisenberg/models-vllm"
PORT = 8081  # different from llama-server default to avoid collisions
OUTPUT_ROOT = REPO / "experiments/rtxpro6000_vllm"

sys.path.insert(0, str(REPO / "tools"))
from rtxpro6000_coding_bench import (
    BENCHMARKS, load_prompt, score_response,
)

PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test."
)
MAX_TOKENS = 256
WARMUP_RUNS = 1
TIMED_RUNS = 5

MODELS = {
    "gpt-oss-120b-vllm": {
        "name": "gpt-oss-120b MXFP4 (vLLM)",
        "path": f"{MODELS_ROOT}/gpt-oss-120b-mxfp4",
        "ctx": 131072,
        "extra": [],
    },
    "gemma-4-31b-nvfp4-vllm": {
        "name": "Gemma-4-31B-it NVFP4-turbo (vLLM)",
        "path": f"{MODELS_ROOT}/gemma-4-31b-nvfp4",
        "ctx": 32768,  # NVFP4 at 90% gpu util caps context
        "extra": [],
    },
}


def start_server(cfg, ctx_override=None):
    ctx = ctx_override or cfg["ctx"]
    cmd = [
        VLLM_BIN, "serve", cfg["path"],
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-model-len", str(ctx),
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
    ] + cfg.get("extra", [])
    log_path = f"/tmp/vllm-{cfg['key']}.log"
    log_f = open(log_path, "w")
    env = os.environ.copy()
    # Triton JIT needs a C compiler available
    env_bin = os.path.dirname(VLLM_BIN)
    gcc = f"{env_bin}/x86_64-conda-linux-gnu-gcc"
    if os.path.isfile(gcc):
        env["CC"] = gcc
    env.setdefault("PATH", "")
    env["PATH"] = env_bin + ":" + env["PATH"]
    # vLLM runtime kernels need nvcc; use the CUDA env we set up earlier
    cuda_home = "/home/gisenberg/.micromamba/envs/cuda"
    if os.path.isfile(f"{cuda_home}/bin/nvcc"):
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:" + env["PATH"]
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_path


def wait_for_server(timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
        proc.wait(timeout=20)


def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def get_served_model_name():
    with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5) as r:
        data = json.loads(r.read())
    return data["data"][0]["id"]


def measure_streaming(served_model, prompt=PROMPT, max_tokens=MAX_TOKENS):
    body = json.dumps({
        "model": served_model,
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
    first = None
    last = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "): continue
                payload = line[6:].strip()
                if payload == "[DONE]": break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"): usage = obj["usage"]
                for c in obj.get("choices", []):
                    delta = c.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        now = time.perf_counter()
                        if first is None: first = now
                        last = now
                        n += 1
    t_end = time.perf_counter()
    if first is None: return None
    ttft = first - t_send
    decode_time = (last - first) if last and last > first else None
    decode_tps = (n / decode_time) if decode_time and decode_time > 0 else None
    if usage and usage.get("completion_tokens"):
        n = usage["completion_tokens"]
        if decode_time and decode_time > 0:
            decode_tps = n / decode_time
    return {
        "ttft_s": round(ttft, 4),
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "n_tokens": n,
        "total_s": round(t_end - t_send, 3),
    }


def chat_nonstream(served_model, prompt, max_tokens=16384):
    body = json.dumps({
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"]["content"]


def run_throughput(served_model, ctx_override=None):
    runs = []
    print(f"Warmup...")
    try:
        r = measure_streaming(served_model)
        print(f"  warmup: ttft={r['ttft_s']}s decode={r['decode_tps']} tok/s ({r['n_tokens']} tok)")
    except Exception as e:
        print(f"  warmup ERROR: {e}")
    print(f"Timed runs:")
    for i in range(TIMED_RUNS):
        try:
            r = measure_streaming(served_model)
            runs.append(r)
            print(f"  run {i+1}/{TIMED_RUNS}: ttft={r['ttft_s']}s decode={r['decode_tps']} tok/s ({r['n_tokens']} tok)")
        except Exception as e:
            print(f"  run {i+1}: ERROR {e}")
    return runs


def run_coding(served_model, cfg, key):
    artifacts_dir = OUTPUT_ROOT / f"coding_{key}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for bench_name, expected, module_name in BENCHMARKS:
        print(f"\n  --- {bench_name} ---")
        prompt = load_prompt(bench_name)
        t0 = time.perf_counter()
        try:
            resp = chat_nonstream(served_model, prompt)
        except Exception as e:
            print(f"    ERROR: {e}")
            results[bench_name] = {"error": str(e), "expected": expected}
            continue
        elapsed = time.perf_counter() - t0
        (artifacts_dir / f"{bench_name}.md").write_text(resp)
        score = score_response(bench_name, module_name, resp)
        score["expected"] = expected
        score["elapsed_s"] = round(elapsed, 2)
        results[bench_name] = score
        print(f"    {score.get('passed', 0)}/{expected} passed in {elapsed:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_key")
    p.add_argument("--coding", action="store_true")
    p.add_argument("--ctx", type=int, default=None)
    args = p.parse_args()

    if args.model_key not in MODELS:
        print(f"Unknown: {args.model_key}"); sys.exit(1)
    cfg = dict(MODELS[args.model_key]); cfg["key"] = args.model_key
    if not os.path.isdir(cfg["path"]):
        print(f"Missing: {cfg['path']}"); sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{'='*70}\n{cfg['name']}\n{'='*70}")
    t_load = time.perf_counter()
    proc, log = start_server(cfg, ctx_override=args.ctx)
    print(f"Waiting for server (log: {log})...", flush=True)
    if not wait_for_server(timeout=1800):
        print("TIMEOUT")
        stop_server(proc)
        sys.exit(2)
    load_s = time.perf_counter() - t_load
    print(f"Ready after {load_s:.1f}s")
    served_model = get_served_model_name()
    vram_mb = vram_used_mb()
    print(f"Served model: {served_model}")
    print(f"VRAM: {vram_mb} MB")

    summary = {
        "model": cfg["name"], "key": args.model_key, "path": cfg["path"],
        "ctx": args.ctx or cfg["ctx"], "backend": "vllm",
        "vram_mb": vram_mb, "load_seconds": round(load_s, 1),
        "served_model_id": served_model,
    }

    print("\n--- Throughput ---")
    runs = run_throughput(served_model, ctx_override=args.ctx)
    if runs:
        ttfts = [r["ttft_s"] for r in runs]
        decodes = [r["decode_tps"] for r in runs if r["decode_tps"] is not None]
        summary["throughput"] = {
            "runs": runs,
            "ttft_mean": round(statistics.mean(ttfts), 4),
            "ttft_median": round(statistics.median(ttfts), 4),
            "decode_mean": round(statistics.mean(decodes), 2) if decodes else None,
            "decode_median": round(statistics.median(decodes), 2) if decodes else None,
        }
        print(f"\n  MEAN: TTFT={summary['throughput']['ttft_mean']*1000:.0f}ms | decode={summary['throughput']['decode_mean']} tok/s")

    if args.coding:
        print("\n--- Coding ---")
        coding = run_coding(served_model, cfg, args.model_key)
        summary["coding"] = coding
        total_p = sum(r.get("passed", 0) for r in coding.values() if isinstance(r, dict))
        total_e = sum(r.get("expected", 0) for r in coding.values() if isinstance(r, dict))
        summary["coding_total"] = f"{total_p}/{total_e}"
        print(f"\n  CODING: {total_p}/{total_e}")

    stop_server(proc)

    out = OUTPUT_ROOT / f"{args.model_key}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
