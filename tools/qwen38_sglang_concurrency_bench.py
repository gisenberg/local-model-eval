#!/usr/bin/env python3
"""Measure Qwen3.8 aggregate throughput on an OpenAI-compatible SGLang server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from muse_glimmer_fp8_bench import stream_once


def gpu_sample() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        utilization, memory_utilization, power, memory_used = (
            float(value.strip()) for value in output.splitlines()[0].split(",")
        )
        return {
            "gpu_utilization_pct": utilization,
            "memory_utilization_pct": memory_utilization,
            "power_w": power,
            "memory_used_mb": memory_used,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def summarize_gpu(samples: list[dict[str, float]]) -> dict[str, float] | None:
    if not samples:
        return None
    return {
        f"{key}_{suffix}": round(function(sample[key] for sample in samples), 2)
        for key in samples[0]
        for suffix, function in (("mean", statistics.mean), ("max", max))
    }


def run_wave(
    endpoint: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    reasoning_effort: str,
    reasoning_budget_tokens: int,
    reasoning_budget_message: str,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)
    stop_sampling = threading.Event()
    gpu_samples: list[dict[str, float]] = []

    def request() -> dict[str, Any]:
        barrier.wait()
        return stream_once(
            endpoint,
            model,
            reasoning_effort,
            max_tokens,
            20,
            reasoning_effort,
            repetition_penalty=1.0,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_budget_message=reasoning_budget_message,
        )

    def sample_gpu() -> None:
        while not stop_sampling.is_set():
            sample = gpu_sample()
            if sample is not None:
                gpu_samples.append(sample)
            stop_sampling.wait(0.25)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request) for _ in range(concurrency)]
        started_at = time.perf_counter()
        barrier.wait()
        responses = [future.result() for future in futures]
        elapsed_s = time.perf_counter() - started_at
    stop_sampling.set()
    sampler.join(timeout=5)

    completion_tokens = sum(response["completion_tokens"] for response in responses)
    ttfts = [
        response["ttft_s"]
        for response in responses
        if response["ttft_s"] is not None
    ]
    return {
        "concurrency": concurrency,
        "elapsed_s": round(elapsed_s, 4),
        "completion_tokens": completion_tokens,
        "aggregate_tps": round(completion_tokens / elapsed_s, 2),
        "per_agent_tps": round(completion_tokens / elapsed_s / concurrency, 2),
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "request_elapsed_mean_s": round(
            statistics.mean(response["elapsed_s"] for response in responses), 4
        ),
        "request_completion_tokens": [
            response["completion_tokens"] for response in responses
        ],
        "gpu": summarize_gpu(gpu_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--model", default="qwen38-27b-fp8-sglang-mtp4")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--reasoning-budget-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-budget-message",
        default="\n\nWait, I'm overthinking this. Thinking budget complete. Let's answer now.",
    )
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/qwen38_27b_fp8_sglang_mtp4/concurrency.json"),
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    waves = []

    print("warming up", flush=True)
    warmup = run_wave(
        endpoint,
        args.model,
        1,
        args.max_tokens,
        args.reasoning_effort,
        args.reasoning_budget_tokens,
        args.reasoning_budget_message,
    )
    print(json.dumps(warmup, indent=2), flush=True)

    for concurrency in concurrencies:
        for trial in range(1, args.trials + 1):
            print(f"concurrency={concurrency} trial={trial}/{args.trials}", flush=True)
            result = run_wave(
                endpoint,
                args.model,
                concurrency,
                args.max_tokens,
                args.reasoning_effort,
                args.reasoning_budget_tokens,
                args.reasoning_budget_message,
            )
            result["trial"] = trial
            waves.append(result)
            print(json.dumps(result, indent=2), flush=True)

    summary = {}
    for concurrency in concurrencies:
        matching = [wave for wave in waves if wave["concurrency"] == concurrency]
        summary[str(concurrency)] = {
            "aggregate_tps_mean": round(
                statistics.mean(wave["aggregate_tps"] for wave in matching), 2
            ),
            "per_agent_tps_mean": round(
                statistics.mean(wave["per_agent_tps"] for wave in matching), 2
            ),
            "ttft_mean_s": round(
                statistics.mean(wave["ttft_mean_s"] for wave in matching), 4
            ),
        }

    artifact = {
        "base_url": args.base_url,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_budget_tokens": args.reasoning_budget_tokens,
        "max_tokens": args.max_tokens,
        "warmup": warmup,
        "waves": waves,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
