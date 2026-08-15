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


def sampling(
    reasoning_strength: str,
    top_k: int | None = 64,
    thinking_token_budget: int | None = None,
    reasoning_budget_tokens: int | None = None,
    reasoning_budget_message: str | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
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
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None and top_k >= 0:
        settings["top_k"] = top_k
    if thinking_token_budget is not None and thinking_token_budget >= 0:
        settings["thinking_token_budget"] = thinking_token_budget
    if reasoning_budget_tokens is not None and reasoning_budget_tokens >= 0:
        settings["reasoning_budget_tokens"] = reasoning_budget_tokens
    if reasoning_budget_message:
        settings["reasoning_budget_message"] = reasoning_budget_message
    chat_template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        settings["chat_template_kwargs"] = chat_template_kwargs
    return settings


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
    top_k: int | None = 64,
    reasoning_effort: str | None = None,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    reasoning_budget_tokens: int | None = None,
    reasoning_budget_message: str | None = None,
) -> dict[str, Any]:
    settings = sampling(
        reasoning_strength,
        top_k,
        reasoning_budget_tokens=reasoning_budget_tokens,
        reasoning_budget_message=reasoning_budget_message,
        reasoning_effort=reasoning_effort,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
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
    top_k: int | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    warmup_results = []
    for index in range(warmups):
        result = stream_once(
            endpoint,
            model,
            reasoning_strength,
            max_tokens,
            top_k,
            reasoning_effort,
        )
        warmup_results.append(result)
        print(f"warmup {index + 1}/{warmups}: {result}", flush=True)

    timed_results = []
    for index in range(runs):
        result = stream_once(
            endpoint,
            model,
            reasoning_strength,
            max_tokens,
            top_k,
            reasoning_effort,
        )
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
    top_k: int | None,
    thinking_token_budget: int | None,
    reasoning_budget_tokens: int | None,
    reasoning_budget_message: str | None,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
    presence_penalty: float,
    repetition_penalty: float,
    selected_benchmarks: set[str] | None,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for bench_name, expected, module_name in BENCHMARKS:
        if selected_benchmarks is not None and bench_name not in selected_benchmarks:
            continue
        print(f"coding: {bench_name}", flush=True)
        settings = sampling(
            reasoning_strength,
            top_k,
            thinking_token_budget,
            reasoning_budget_tokens,
            reasoning_budget_message,
            enable_thinking,
            reasoning_effort,
            presence_penalty,
            repetition_penalty,
        )
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
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default=None,
        help="Send a Qwen-style reasoning_effort chat-template argument.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--throughput-tokens", type=int, default=512)
    parser.add_argument("--coding-tokens", type=int, default=16384)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-coding", action="store_true")
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Optional comma-separated subset of coding benchmarks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Set a negative value to omit top_k from requests.",
    )
    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        default=-1,
        help="Set a non-negative value to send vLLM's thinking_token_budget.",
    )
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=int,
        default=-1,
        help="Set a non-negative llama.cpp reasoning token budget per request.",
    )
    parser.add_argument(
        "--reasoning-budget-message",
        default="",
        help="Message llama.cpp injects when the reasoning budget is exhausted.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control the model chat template's enable_thinking setting.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="OpenAI-compatible presence penalty used for coding requests.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="vLLM repetition penalty used for coding requests.",
    )
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
        "reasoning_effort": args.reasoning_effort,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": args.top_k if args.top_k >= 0 else None,
            "thinking_token_budget": (
                args.thinking_token_budget
                if args.thinking_token_budget >= 0
                else None
            ),
            "reasoning_budget_tokens": (
                args.reasoning_budget_tokens
                if args.reasoning_budget_tokens >= 0
                else None
            ),
            "reasoning_budget_message": args.reasoning_budget_message or None,
            "enable_thinking": {
                "auto": None,
                "on": True,
                "off": False,
            }[args.thinking_mode],
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
        },
        "vram_used_mb": vram_used_mb(),
        "throughput": None
        if args.skip_throughput
        else throughput(
            endpoint,
            args.model,
            args.reasoning_strength,
            args.warmups,
            args.runs,
            args.throughput_tokens,
            args.top_k,
            args.reasoning_effort,
        ),
        "coding": None
        if args.skip_coding
        else coding(
            endpoint,
            args.model,
            args.reasoning_strength,
            args.coding_tokens,
            args.output / "coding",
            args.top_k,
            args.thinking_token_budget
            if args.thinking_token_budget >= 0
            else None,
            args.reasoning_budget_tokens
            if args.reasoning_budget_tokens >= 0
            else None,
            args.reasoning_budget_message or None,
            {
                "auto": None,
                "on": True,
                "off": False,
            }[args.thinking_mode],
            args.reasoning_effort,
            args.presence_penalty,
            args.repetition_penalty,
            {name.strip() for name in args.benchmarks.split(",") if name.strip()}
            or None,
        ),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
