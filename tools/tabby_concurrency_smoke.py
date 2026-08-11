#!/usr/bin/env python3
"""Exercise simultaneous OpenAI-compatible TabbyAPI chat requests."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def run_request(
    request_id: int,
    args: argparse.Namespace,
    barrier: threading.Barrier,
) -> dict:
    prompt = (
        f"Request {request_id}: Write a Python function named triangular_{request_id} "
        "that returns the nth triangular number. Include type hints, a short docstring, "
        "input validation, and three assert examples. Return only one Python code block."
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    barrier.wait()
    started = time.perf_counter()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    usage = body.get("usage") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    completion_tokens = usage.get("completion_tokens")
    if not completion_tokens:
        token_response = requests.post(
            f"{args.base_url.rstrip('/')}/token/encode",
            json={"text": content, "add_bos_token": False},
            timeout=args.timeout,
        )
        token_response.raise_for_status()
        token_body = token_response.json()
        completion_tokens = token_body.get("length")
        if not isinstance(completion_tokens, int):
            completion_tokens = len(token_body["tokens"])
    return {
        "request_id": request_id,
        "elapsed_s": round(elapsed, 4),
        "completion_tokens": completion_tokens,
        "client_completion_tps": (
            round(completion_tokens / elapsed, 2) if completion_tokens else None
        ),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "passed": bool(content) and not reasoning,
        "usage": usage,
    }


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least one")

    barrier = threading.Barrier(args.concurrency)
    wall_started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_request, request_id, args, barrier)
            for request_id in range(1, args.concurrency + 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    total_elapsed = time.perf_counter() - wall_started
    results.sort(key=lambda item: item["request_id"])
    generation_wall_elapsed = max(item["elapsed_s"] for item in results)

    total_completion_tokens = sum(
        item["completion_tokens"] or 0 for item in results
    )
    output = {
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_elapsed_s": generation_wall_elapsed,
        "total_elapsed_s": round(total_elapsed, 4),
        "total_completion_tokens": total_completion_tokens,
        "aggregate_completion_tps": (
            round(total_completion_tokens / generation_wall_elapsed, 2)
            if total_completion_tokens
            else None
        ),
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
