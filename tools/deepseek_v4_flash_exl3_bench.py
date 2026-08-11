#!/usr/bin/env python3
"""Benchmark an already-running TabbyAPI DeepSeek V4 Flash EXL3 server."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
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
    "Include type hints, a docstring, and one pytest test."
)


def request_json(url: str, body: dict[str, Any], timeout: int = 1800) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


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


def template_kwargs(mode: str) -> dict[str, Any]:
    return {
        "enable_thinking": mode == "native",
        "drop_thinking": False,
        "reasoning_effort": "low",
    }


def stream_once(
    endpoint: str,
    model: str,
    mode: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": THROUGHPUT_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template_kwargs(mode),
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
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
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
    mode: str,
    warmups: int,
    runs: int,
    max_tokens: int,
) -> dict[str, Any]:
    warmup_results = []
    for index in range(warmups):
        result = stream_once(endpoint, model, mode, max_tokens)
        warmup_results.append(result)
        print(f"warmup {index + 1}/{warmups}: {result}", flush=True)

    timed_results = []
    for index in range(runs):
        result = stream_once(endpoint, model, mode, max_tokens)
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
    mode: str,
    max_tokens: int,
    artifacts_dir: Path,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for bench_name, expected, module_name in BENCHMARKS:
        print(f"coding: {bench_name}", flush=True)
        started_at = time.perf_counter()
        response = request_json(
            endpoint,
            {
                "model": model,
                "messages": [{"role": "user", "content": load_prompt(bench_name)}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": template_kwargs(mode),
            },
        )
        elapsed_s = time.perf_counter() - started_at
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        (artifacts_dir / f"{bench_name}.md").write_text(content)
        (artifacts_dir / f"{bench_name}.json").write_text(
            json.dumps(response, indent=2) + "\n"
        )
        score = normalize_score(score_response(bench_name, module_name, content), expected)
        score.update(
            {
                "elapsed_s": round(elapsed_s, 2),
                "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "reasoning_chars": len(message.get("reasoning_content") or ""),
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash-0731-exl3-2.04bpw-b5526bab",
    )
    parser.add_argument("--mode", choices=("no-think", "native"), default="no-think")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--throughput-tokens", type=int, default=256)
    parser.add_argument("--coding-tokens", type=int, default=16384)
    parser.add_argument(
        "--skip-coding",
        action="store_true",
        help="Run only the throughput portion of the benchmark.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    result = {
        "model": args.model,
        "mode": args.mode,
        "base_url": args.base_url,
        "vram_before_mb": vram_used_mb(),
        "throughput": throughput(
            endpoint,
            args.model,
            args.mode,
            args.warmups,
            args.runs,
            args.throughput_tokens,
        ),
    }
    result["vram_after_throughput_mb"] = vram_used_mb()
    if not args.skip_coding:
        result["coding"] = coding(
            endpoint,
            args.model,
            args.mode,
            args.coding_tokens,
            args.output_dir / "coding",
        )
        result["vram_after_coding_mb"] = vram_used_mb()
    output_path = args.output_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    if "coding" in result:
        print(f"TOTAL: {result['coding']['total_score']}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
