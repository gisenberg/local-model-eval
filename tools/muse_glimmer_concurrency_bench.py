#!/usr/bin/env python3
"""Measure concurrent aggregate throughput against the Muse Glimmer server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from muse_glimmer_fp8_bench import stream_once


METRIC_NAMES = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)


def speculative_counters(metrics_url: str) -> dict[str, float]:
    with urllib.request.urlopen(metrics_url, timeout=30) as response:
        metrics = response.read().decode("utf-8")
    result = {}
    for name in METRIC_NAMES:
        prefix = f"vllm:{name}{{"
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith(prefix)
        ]
        result[name] = sum(values)
    return result


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
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[f"{key}_mean"] = round(statistics.mean(values), 2)
        result[f"{key}_max"] = round(max(values), 2)
    return result


def run_wave(
    endpoint: str,
    metrics_url: str,
    model: str,
    reasoning_strength: str,
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)
    stop_sampling = threading.Event()
    gpu_samples: list[dict[str, float]] = []

    def request() -> dict[str, Any]:
        barrier.wait()
        return stream_once(endpoint, model, reasoning_strength, max_tokens)

    def sample_gpu() -> None:
        while not stop_sampling.is_set():
            sample = gpu_sample()
            if sample is not None:
                gpu_samples.append(sample)
            stop_sampling.wait(0.25)

    before = speculative_counters(metrics_url)
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
    after = speculative_counters(metrics_url)

    completion_tokens = sum(response["completion_tokens"] for response in responses)
    ttfts = [response["ttft_s"] for response in responses]
    deltas = {name: after[name] - before[name] for name in METRIC_NAMES}
    drafts = deltas["spec_decode_num_drafts_total"]
    accepted = deltas["spec_decode_num_accepted_tokens_total"]
    draft_tokens = deltas["spec_decode_num_draft_tokens_total"]
    return {
        "concurrency": concurrency,
        "elapsed_s": round(elapsed_s, 4),
        "completion_tokens": completion_tokens,
        "aggregate_tps": round(completion_tokens / elapsed_s, 2),
        "per_agent_tps": round(completion_tokens / elapsed_s / concurrency, 2),
        "ttft_mean_s": round(statistics.mean(ttfts), 4),
        "ttft_median_s": round(statistics.median(ttfts), 4),
        "request_elapsed_mean_s": round(
            statistics.mean(response["elapsed_s"] for response in responses), 4
        ),
        "request_completion_tokens": [
            response["completion_tokens"] for response in responses
        ],
        "speculative": {
            "drafts": int(drafts),
            "draft_tokens": int(draft_tokens),
            "accepted_tokens": int(accepted),
            "accepted_per_draft": round(accepted / drafts, 3) if drafts else None,
            "draft_acceptance_pct": (
                round(accepted / draft_tokens * 100, 2) if draft_tokens else None
            ),
        },
        "gpu": summarize_gpu(gpu_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--model", default="muse-glimmer-30b-fp8")
    parser.add_argument("--reasoning-strength", default="high")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/muse_glimmer_30b_fp8_dflash15/concurrency.json"),
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    metrics_url = f"{args.base_url.removesuffix('/v1').rstrip('/')}/metrics"
    concurrencies = [int(value) for value in args.concurrency.split(",")]

    print("warming up", flush=True)
    warmup = run_wave(
        endpoint,
        metrics_url,
        args.model,
        args.reasoning_strength,
        1,
        args.max_tokens,
    )
    print(json.dumps(warmup, indent=2), flush=True)

    waves = []
    for concurrency in concurrencies:
        for trial in range(1, args.trials + 1):
            print(
                f"concurrency={concurrency} trial={trial}/{args.trials}", flush=True
            )
            result = run_wave(
                endpoint,
                metrics_url,
                args.model,
                args.reasoning_strength,
                concurrency,
                args.max_tokens,
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
            "aggregate_tps_median": round(
                statistics.median(wave["aggregate_tps"] for wave in matching), 2
            ),
            "per_agent_tps_mean": round(
                statistics.mean(wave["per_agent_tps"] for wave in matching), 2
            ),
            "ttft_mean_s": round(
                statistics.mean(wave["ttft_mean_s"] for wave in matching), 4
            ),
        }

    output = {
        "model": args.model,
        "reasoning_strength": args.reasoning_strength,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "warmup": warmup,
        "waves": waves,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"summary": summary, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
