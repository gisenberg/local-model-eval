#!/usr/bin/env python3
"""Benchmark an already-running Muse Glimmer OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from rtxpro6000_coding_bench import (  # noqa: E402
    BENCHMARKS,
    load_prompt,
    normalize_score,
    score_response,
)

THROUGHPUT_PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test. "
    "Return the implementation in one python code block."
)


def sampling(reasoning_strength: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant.\n\n"
                    f"Reasoning strength: {reasoning_strength}."
                ),
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }


def request_json(url: str, body: dict[str, Any], timeout: int = 1800) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def vram_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def stream_once(
    endpoint: str,
    model: str,
    reasoning_strength: str,
    max_tokens: int,
) -> dict[str, Any]:
    settings = sampling(reasoning_strength)
    body = {
        "model": model,
        "messages": settings.pop("messages")
        + [{"role": "user", "content": THROUGHPUT_PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **settings,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    sent_at = time.perf_counter()
    first_at: float | None = None
    last_at: float | None = None
    content_events = 0
    usage: dict[str, Any] | None = None

    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if any(delta.get(key) for key in ("content", "reasoning", "reasoning_content")):
                now = time.perf_counter()
                first_at = first_at or now
                last_at = now
                content_events += 1

    ended_at = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens") or content_events)
    decode_s = None if first_at is None or last_at is None else last_at - first_at
    decode_tps = None
    if decode_s and decode_s > 0:
        decode_tps = max(completion_tokens - 1, 1) / decode_s
    return {
        "ttft_s": None if first_at is None else round(first_at - sent_at, 4),
        "decode_s": None if decode_s is None else round(decode_s, 4),
        "decode_tps": None if decode_tps is None else round(decode_tps, 2),
        "completion_tokens": completion_tokens,
        "content_events": content_events,
        "elapsed_s": round(ended_at - sent_at, 4),
        "usage": usage,
    }


def throughput(
    endpoint: str,
    model: str,
    reasoning_strength: str,
    warmups: int,
    runs: int,
    max_tokens: int,
) -> dict[str, Any]:
    warmup_results = []
    for index in range(warmups):
        result = stream_once(endpoint, model, reasoning_strength, max_tokens)
        warmup_results.append(result)
        print(f"warmup {index + 1}/{warmups}: {result}", flush=True)

    timed_results = []
    for index in range(runs):
        result = stream_once(endpoint, model, reasoning_strength, max_tokens)
        timed_results.append(result)
        print(f"run {index + 1}/{runs}: {result}", flush=True)

    ttfts = [r["ttft_s"] for r in timed_results if r["ttft_s"] is not None]
    rates = [r["decode_tps"] for r in timed_results if r["decode_tps"] is not None]
    return {
        "prompt": THROUGHPUT_PROMPT,
        "max_tokens": max_tokens,
        "warmups": warmup_results,
        "runs": timed_results,
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "decode_mean_tps": round(statistics.mean(rates), 2) if rates else None,
        "decode_median_tps": round(statistics.median(rates), 2) if rates else None,
    }


def coding(
    endpoint: str,
    model: str,
    reasoning_strength: str,
    max_tokens: int,
    artifacts_dir: Path,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for bench_name, expected, module_name in BENCHMARKS:
        print(f"coding: {bench_name}", flush=True)
        settings = sampling(reasoning_strength)
        body = {
            "model": model,
            "messages": settings.pop("messages")
            + [{"role": "user", "content": load_prompt(bench_name)}],
            "max_tokens": max_tokens,
            "stream": False,
            **settings,
        }
        started_at = time.perf_counter()
        response = request_json(endpoint, body)
        elapsed_s = time.perf_counter() - started_at
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        (artifacts_dir / f"{bench_name}.md").write_text(content)
        (artifacts_dir / f"{bench_name}.json").write_text(
            json.dumps(response, indent=2) + "\n"
        )
        score = normalize_score(score_response(bench_name, module_name, content), expected)
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        score.update(
            {
                "elapsed_s": round(elapsed_s, 2),
                "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "reasoning_chars": len(reasoning),
            }
        )
        results[bench_name] = score
        print(
            f"  {score['scored_passed']}/{expected}, "
            f"finish={score['finish_reason']}, elapsed={score['elapsed_s']}s",
            flush=True,
        )

    passed = sum(result.get("scored_passed", 0) for result in results.values())
    expected = sum(result.get("expected", 0) for result in results.values())
    return {
        "total_passed": passed,
        "total_expected": expected,
        "total_score": f"{passed}/{expected}",
        "benchmarks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--model", default="muse-glimmer-30b-fp8")
    parser.add_argument(
        "--reasoning-strength",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--throughput-tokens", type=int, default=512)
    parser.add_argument("--coding-tokens", type=int, default=16384)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "experiments" / "muse_glimmer_30b_fp8",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "endpoint": endpoint,
        "reasoning_strength": args.reasoning_strength,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "vram_used_mb": vram_used_mb(),
        "throughput": throughput(
            endpoint,
            args.model,
            args.reasoning_strength,
            args.warmups,
            args.runs,
            args.throughput_tokens,
        ),
        "coding": coding(
            endpoint,
            args.model,
            args.reasoning_strength,
            args.coding_tokens,
            args.output / "coding",
        ),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
