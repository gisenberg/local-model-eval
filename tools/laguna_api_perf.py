#!/usr/bin/env python3
"""Measure a Laguna vLLM endpoint through its OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import requests
from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

PROMPTS = [
    (
        "Implement a production-quality Python function that applies a unified diff "
        "to an in-memory text file. Include type hints, validation, and pytest tests. "
        "Return the implementation and tests in one Python code block."
    ),
    (
        "Implement a thread-safe bounded LRU cache in Python without OrderedDict. "
        "Include type hints, invariant checks, and pytest tests. Return the complete "
        "implementation and tests in one Python code block."
    ),
    (
        "Implement A* search for a weighted grid in Python, including unreachable "
        "targets and input validation. Include pytest tests and return one complete "
        "Python code block."
    ),
    (
        "Implement a recursive-descent parser for arithmetic expressions in Python. "
        "Support unary operators, parentheses, and useful errors. Include pytest "
        "tests and return one complete Python code block."
    ),
]

_NVML_HANDLE: Any | None = None


def gpu_memory_mib() -> int:
    global _NVML_HANDLE
    if _NVML_HANDLE is None:
        nvmlInit()
        _NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
    return int(nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used // (1024 * 1024))


class MemorySampler:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(gpu_memory_mib())
            except (NVMLError, OSError, RuntimeError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def stream_completion(
    port: int,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
        stream=True,
        timeout=900,
    )
    response.raise_for_status()

    first_token_at: float | None = None
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if content or reasoning:
            first_token_at = first_token_at or time.perf_counter()
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)

    ended = time.perf_counter()
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft_s = first_token_at - started if first_token_at is not None else None
    decode_s = ended - first_token_at if first_token_at is not None else None
    decode_tps = (
        (completion_tokens - 1) / decode_s
        if completion_tokens > 1 and decode_s and decode_s > 0
        else None
    )
    return {
        "seed": seed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "elapsed_s": round(ended - started, 4),
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "decode_tps": round(decode_tps, 3) if decode_tps is not None else None,
        "finish_reason": finish_reason,
        "content_chars": len("".join(content_parts)),
        "reasoning_chars": len("".join(reasoning_parts)),
    }


def metrics_snapshot(port: int) -> list[str]:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
    response.raise_for_status()
    prefixes = (
        "vllm:spec_decode",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:gpu_cache_usage_perc",
        "vllm:request_success_total",
    )
    return [
        line
        for line in response.text.splitlines()
        if not line.startswith("#") and line.startswith(prefixes)
    ]


def speculative_metrics(lines: list[str]) -> dict[str, Any] | None:
    totals = {
        "num_drafts": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
    }
    found = False
    for line in lines:
        if not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{", 1)[0]
        if not metric_name.endswith("_total") or len(parts) != 2:
            continue
        try:
            value = int(float(parts[1]))
        except ValueError:
            continue
        found = True
        if "num_drafts" in metric_name:
            totals["num_drafts"] += value
        elif "num_draft_tokens" in metric_name:
            totals["draft_tokens"] += value
        elif (
            "num_accepted_tokens_per_pos" not in metric_name
            and "num_accepted_tokens" in metric_name
        ):
            totals["accepted_tokens"] += value
    return totals if found else None


def speculative_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    delta = {key: int(after[key]) - int(before[key]) for key in before}
    drafts = delta["num_drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    delta["acceptance_rate_percent"] = (
        round(100 * accepted / draft_tokens, 3) if draft_tokens else None
    )
    delta["acceptance_length"] = (
        round(1 + accepted / drafts, 3) if drafts else None
    )
    return delta


def summarize(runs: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    completion_tokens = sum(int(run["completion_tokens"]) for run in runs)
    ttfts = [float(run["ttft_s"]) for run in runs if run.get("ttft_s") is not None]
    per_request_tps = [
        float(run["decode_tps"]) for run in runs if run.get("decode_tps") is not None
    ]
    return {
        "requests": len(runs),
        "completion_tokens": completion_tokens,
        "wall_s": round(wall_s, 4),
        "aggregate_output_tps": round(completion_tokens / wall_s, 3),
        "mean_ttft_s": round(statistics.mean(ttfts), 4),
        "p50_ttft_s": round(statistics.median(ttfts), 4),
        "mean_request_decode_tps": round(statistics.mean(per_request_tps), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--mode", choices=("baseline", "dflash"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    idle_memory_mib = gpu_memory_mib()

    stream_completion(
        args.port,
        args.served_name,
        PROMPTS[0],
        seed=9000,
        max_tokens=128,
    )
    metrics_before_lines = metrics_snapshot(args.port)
    metrics_before = speculative_metrics(metrics_before_lines)

    with MemorySampler() as memory:
        serial_started = time.perf_counter()
        serial_runs = [
            stream_completion(
                args.port,
                args.served_name,
                prompt,
                seed=1000 + index,
                max_tokens=args.max_tokens,
            )
            for index, prompt in enumerate(PROMPTS)
        ]
        serial_wall_s = time.perf_counter() - serial_started

        concurrent_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    stream_completion,
                    args.port,
                    args.served_name,
                    prompt,
                    2000 + index,
                    args.max_tokens,
                )
                for index, prompt in enumerate(PROMPTS)
            ]
            concurrent_runs = [future.result() for future in futures]
        concurrent_wall_s = time.perf_counter() - concurrent_started

    metrics_after_lines = metrics_snapshot(args.port)
    metrics_after = speculative_metrics(metrics_after_lines)
    result = {
        "mode": args.mode,
        "served_name": args.served_name,
        "request_config": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": args.max_tokens,
            "preserve_thinking": True,
        },
        "memory": {
            "idle_used_mib": idle_memory_mib,
            "peak_used_mib": max(memory.samples, default=idle_memory_mib),
        },
        "serial": {
            "summary": summarize(serial_runs, serial_wall_s),
            "runs": serial_runs,
        },
        "concurrent_4": {
            "summary": summarize(concurrent_runs, concurrent_wall_s),
            "runs": concurrent_runs,
        },
        "speculative_decoding": speculative_delta(metrics_before, metrics_after),
        "metrics_before": metrics_before_lines,
        "metrics_after": metrics_after_lines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
