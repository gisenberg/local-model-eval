#!/usr/bin/env python3
"""Run Qwen3-Coder-Next FP8 through the local coding benchmark.

This is the same 4-task / 3-run pytest-scored coding suite used by the
Nemotron/Qwen vLLM experiments, but with a Qwen3-Coder-Next FP8 serve command:

  - native 256K context (`--max-model-len 262144`)
  - single sequence (`--max-num-seqs 1`)
  - non-thinking mode (the model card says this model does not emit think blocks)
  - FP8 KV cache for full-context fit
  - Qwen model-card sampling defaults

Output:
  experiments/qwen3_coder_next_fp8_vllm_256k/
    serve_cmd.txt
    vllm.log
    metrics.txt
    results.json
    summary.json
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
OUT_BASE = REPO / "experiments" / "qwen3_coder_next_fp8_vllm_256k"
VENV = "/home/gisenberg/venvs/dflash-pr40898"
CUDA_HOME = f"{VENV}/lib/python3.12/site-packages/nvidia/cu13"

MODEL = "/mnt/extended/gisenberg/models/qwen3-coder-next-fp8"
SERVED_NAME = "qwen3-coder-next-fp8"
PORT = 8092
RUNS = 3

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
            "7. Use recursive descent parser - no eval() or ast.literal_eval()\n"
            "8. Include type hints and docstrings\n"
            "9. Write 5 pytest tests"
        ),
    },
    "astar": {
        "name": "A* Pathfinding",
        "expected": 6,
        "prompt": (
            "Implement A* pathfinding on a weighted 2D grid in Python.\n"
            "1. Class AStarGrid with find_path(start, end) -> Optional[List[Tuple[int,int]]]\n"
            "2. 4-directional, Manhattan heuristic, heapq, walls (0), weighted cells\n"
            "3. Handle: start==end, walls, out-of-bounds (ValueError)\n"
            "4. Path must be optimal. Include type hints and docstrings\n"
            "5. Write 6 pytest tests"
        ),
    },
    "lru_cache": {
        "name": "LRU Cache with TTL",
        "expected": 6,
        "prompt": (
            "Implement LRU cache with TTL in Python.\n"
            "1. Class TTLCache(capacity, default_ttl)\n"
            "2. get(key), put(key, value, ttl=None), delete(key), size()\n"
            "3. O(1) avg time. Doubly-linked list + hash map, no OrderedDict\n"
            "4. time.monotonic(), lazy cleanup. Type hints and docstrings\n"
            "5. Write 6 pytest tests using unittest.mock.patch on time.monotonic"
        ),
    },
    "string_processor": {
        "name": "String Processor",
        "expected": 5,
        "prompt": (
            "Write class StringProcessor with:\n"
            "1. reverse_words(s) -> str\n"
            "2. count_vowels(s) -> int (case-insensitive)\n"
            "3. is_palindrome(s) -> bool (ignore case, spaces, punctuation)\n"
            "4. caesar_cipher(s, shift) -> str (a-z/A-Z only, support negative)\n"
            "5. most_common_word(s) -> Optional[str] (case-insensitive, first if tied)\n"
            "Include type hints, docstrings, and 5 pytest tests."
        ),
    },
}


def build_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{VENV}/bin:{CUDA_HOME}/bin:/usr/bin:/bin"
    env["CUDA_HOME"] = CUDA_HOME
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    return env


def wait_for_ready(
    port: int,
    served_name: str,
    timeout: int,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
) -> float:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            log_tail = ""
            if log_path is not None and log_path.exists():
                log_tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(
                f"vLLM exited during startup with code {proc.returncode}\n{log_tail}"
            )
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            if r.status_code == 200 and served_name in r.text:
                return time.time() - t0
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s (last err: {last_err})")


def vram_used_mb() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def run_inference(prompt: str, port: int, served_name: str) -> dict:
    body = {
        "model": served_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=body,
        timeout=1200,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content", ""),
        "tokens": completion_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(completion_tokens / elapsed, 1) if completion_tokens and elapsed > 0 else 0,
        "finish_reason": data["choices"][0].get("finish_reason", "?"),
    }


def extract_and_test(content: str, test_file: Path) -> dict:
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    if not blocks:
        return {"passed": 0, "failed": 0, "errors": 0, "error": "no code blocks"}

    combined = "\n\n".join(block.strip() for block in blocks)
    for cls in ["ExpressionEvaluator", "AStarGrid", "TTLCache", "StringProcessor", "BST", "Node"]:
        combined = re.sub(rf"from \w+ import {cls}\b.*\n", "", combined)
    test_file.write_text(combined, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": 0, "error": str(e)}

    output = result.stdout + result.stderr
    return {
        "passed": len(re.findall(r" PASSED", output)),
        "failed": len(re.findall(r" FAILED", output)),
        "errors": len(re.findall(r" ERROR", output)),
        "pytest_tail": output[-2000:],
    }


def run_benchmark(out_dir: Path, port: int, served_name: str) -> dict:
    results = {}
    print(f"=== Qwen3-Coder-Next FP8 coding bench ({served_name}) ===")
    print("Preset=non-thinking temp=1.0 top_p=0.95 top_k=40 max_tokens=16000, 3 runs\n")

    for bench_key, bench in BENCHMARKS.items():
        runs = []
        for run_num in range(1, RUNS + 1):
            print(f"  [{bench['name']}] Run {run_num}/{RUNS}...", end="", flush=True)
            try:
                result = run_inference(bench["prompt"], port, served_name)
                md = out_dir / f"{bench_key}_run{run_num}.md"
                md.write_text(
                    (result.get("reasoning") or "") + "\n\n" + result.get("content", ""),
                    encoding="utf-8",
                )
                test_file = out_dir / f"{bench_key}_run{run_num}_test.py"
                test_result = extract_and_test(result["content"], test_file)
                print(
                    f" {result['tok_per_sec']:.1f} tok/s | "
                    f"{test_result['passed']}/{bench['expected']} | "
                    f"{result['tokens']} tok | {result['finish_reason']}"
                )
                runs.append({**result, **test_result})
            except Exception as e:
                print(f" ERROR: {e}")
                runs.append({"error": str(e)})

        valid = [r for r in runs if "passed" in r]
        if valid:
            best = max(r["passed"] for r in valid)
            avg = sum(r["passed"] for r in valid) / len(valid)
            results[bench_key] = {
                "best": best,
                "avg": round(avg, 1),
                "expected": bench["expected"],
                "runs": runs,
            }
            print(f"  -> Best: {best}/{bench['expected']}, Avg: {avg:.1f}/{bench['expected']}\n")
        else:
            results[bench_key] = {"expected": bench["expected"], "runs": runs, "error": "all runs failed"}

    total_best = sum(min(v.get("best", 0), v.get("expected", 0)) for v in results.values())
    total_avg = sum(min(v.get("avg", 0), v.get("expected", 0)) for v in results.values())
    total_expected = sum(v.get("expected", 0) for v in results.values())
    print(
        f"\nTOTAL: Best-of-{RUNS} = {total_best}/{total_expected} "
        f"({total_best / total_expected * 100:.0f}%), Avg = {total_avg:.1f}/{total_expected}"
    )

    results["_meta"] = {
        "served_name": served_name,
        "port": port,
        "runs": RUNS,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 16000,
        "enable_thinking": False,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_vllm_memory(log_path: Path) -> dict:
    out = {}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="replace")
    if m := re.search(r"Model loading took ([\d.]+) GiB memory", text):
        out["model_load_gib"] = float(m.group(1))
    if m := re.search(r"Available KV cache memory: ([\d.]+) GiB", text):
        out["kv_pool_gib"] = float(m.group(1))
    if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", text):
        out["kv_pool_tokens"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text):
        out["max_concurrency_tokens"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    if m := re.search(r"kv_cache_dtype=([^,\s]+)", text):
        out["kv_cache_dtype"] = m.group(1)
    return out


def scrape_metrics(port: int, out_path: Path) -> dict:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
        out_path.write_text(r.text, encoding="utf-8")
        spec = {}
        for line in r.text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if "spec_decode" in line or "speculative" in line:
                spec[line.split()[0]] = line.split()[-1]
        return spec
    except Exception as e:
        return {"error": str(e)}


def stop_vllm(proc: subprocess.Popen, served_name: str) -> int | None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        pass
    subprocess.run(["pkill", "-KILL", "-f", "VLLM::EngineCore"], check=False)
    subprocess.run(["pkill", "-KILL", "-f", served_name], check=False)
    subprocess.run(["pkill", "-KILL", "-f", "multiprocessing.resource_tracker"], check=False)
    final_mb = None
    for _ in range(30):
        time.sleep(2)
        final_mb = vram_used_mb()
        if final_mb is not None and final_mb < 1000:
            break
    return final_mb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("QWEN3_CODER_NEXT_FP8_MODEL", MODEL))
    parser.add_argument("--served-name", default=SERVED_NAME)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--ready-timeout", type=int, default=3600)
    args = parser.parse_args()

    out_dir = OUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"

    cmd = [
        f"{VENV}/bin/vllm",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_name,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--moe-backend",
        args.moe_backend,
        "--enable-expert-parallel",
        "--async-scheduling",
        "--trust-remote-code",
        "--mamba-backend",
        "flashinfer",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    (out_dir / "serve_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    summary = {
        "model": args.model,
        "served_name": args.served_name,
        "port": args.port,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "moe_backend": args.moe_backend,
        "kv_cache_dtype": args.kv_cache_dtype,
    }

    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=build_env(),
        cwd=str(REPO),
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    summary["pid"] = proc.pid
    try:
        print(f"vLLM pid {proc.pid}, waiting for ready (log: {log_path})...")
        ready_s = wait_for_ready(
            args.port, args.served_name, args.ready_timeout, proc, log_path
        )
        summary["startup_s"] = round(ready_s, 1)
        print(f"ready in {ready_s:.1f}s; running benchmark...")

        results = run_benchmark(out_dir, args.port, args.served_name)
        summary["vram_used_mb_post_bench"] = vram_used_mb()
        summary["spec_metrics"] = scrape_metrics(args.port, out_dir / "metrics.txt")
        summary["memory"] = parse_vllm_memory(log_path)

        best = sum(min(b.get("best", 0), b.get("expected", 0)) for k, b in results.items() if k != "_meta")
        avg = sum(min(b.get("avg", 0), b.get("expected", 0)) for k, b in results.items() if k != "_meta")
        tps_runs = [
            r["tok_per_sec"]
            for k, b in results.items()
            if k != "_meta"
            for r in b.get("runs", [])
            if "tok_per_sec" in r
        ]
        summary["best_of_3"] = best
        summary["avg"] = round(avg, 1)
        summary["mean_tok_per_sec"] = round(sum(tps_runs) / len(tps_runs), 1) if tps_runs else None
        summary["min_tok_per_sec"] = min(tps_runs) if tps_runs else None
        summary["max_tok_per_sec"] = max(tps_runs) if tps_runs else None
    except Exception as e:
        summary["error"] = repr(e)
        if log_path.exists():
            summary["log_tail"] = log_path.read_text(errors="replace")[-5000:]
        raise
    finally:
        print("stopping vLLM...")
        summary["vram_freed_mb"] = stop_vllm(proc, args.served_name)
        log.close()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        keys = [
            "startup_s",
            "best_of_3",
            "avg",
            "mean_tok_per_sec",
            "min_tok_per_sec",
            "max_tok_per_sec",
            "vram_used_mb_post_bench",
            "vram_freed_mb",
            "error",
        ]
        print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=2))


if __name__ == "__main__":
    main()
