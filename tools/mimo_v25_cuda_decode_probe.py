#!/usr/bin/env python3
"""Drive repeated long-context MiMo generations and record streaming evidence."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FILLER = (
    "Repository audit record: module alpha validates an integer identifier, "
    "normalizes the associated path, and returns the unchanged payload. "
    "The implementation is deterministic and has no external side effects.\n"
)
INSTRUCTION = """

Act as a compiler-verification expert.
Write a detailed, self-contained formal analysis of a deterministic state machine.
Continue expanding definitions, invariants, transition cases, and proof obligations until the response limit.
Do not summarize early, do not call tools, and do not stop merely because the proof is repetitive.
This is isolation request {request_index}.
"""


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def token_count(base_url: str, content: str, timeout: int) -> int:
    response = post_json(
        f"{base_url.rstrip('/')}/tokenize",
        {"content": content, "add_special": True},
        timeout,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected tokenize response: {response}")
    return len(tokens)


def fit_prefix(
    base_url: str,
    target_tokens: int,
    timeout: int,
) -> tuple[str, int, int]:
    low = 0
    high = max(1, target_tokens // 16)
    while token_count(base_url, FILLER * high, timeout) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        count = token_count(base_url, FILLER * middle, timeout)
        if count <= target_tokens:
            low = middle
        else:
            high = middle

    prefix = FILLER * low
    return prefix, token_count(base_url, prefix, timeout), low


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    request_index: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_chunk_seconds: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    chunks = 0
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    saw_done = False

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                saw_done = True
                break
            event = json.loads(payload)
            chunks += 1
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    elapsed = time.perf_counter() - started
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    complete = saw_done and finish_reason is not None and bool(usage)
    return {
        "request_index": request_index,
        "ok": complete,
        "error": None if complete else "stream ended without a complete response",
        "elapsed_seconds": round(elapsed, 3),
        "first_chunk_seconds": (
            round(first_chunk_seconds, 3) if first_chunk_seconds is not None else None
        ),
        "chunks": chunks,
        "finish_reason": finish_reason,
        "content_characters": len(content),
        "reasoning_characters": len(reasoning),
        "content_tail": content[-500:],
        "reasoning_tail": reasoning[-500:],
        "usage": usage,
    }


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="mimo-v2.5-iq2-xxs")
    parser.add_argument("--target-prefix-tokens", type=int, default=100_000)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact: dict[str, Any] = {
        "variant": args.variant,
        "base_url": args.base_url,
        "model": args.model,
        "target_prefix_tokens": args.target_prefix_tokens,
        "requests_requested": args.requests,
        "max_tokens": args.max_tokens,
        "started_unix": time.time(),
        "results": [],
    }
    write_artifact(args.output, artifact)

    try:
        prefix, prefix_tokens, repetitions = fit_prefix(
            args.base_url,
            args.target_prefix_tokens,
            args.timeout,
        )
    except Exception as error:
        artifact["setup_error"] = f"{type(error).__name__}: {error}"
        artifact["completed_unix"] = time.time()
        write_artifact(args.output, artifact)
        print(artifact["setup_error"], flush=True)
        return 2

    artifact["prefix_tokens"] = prefix_tokens
    artifact["filler_repetitions"] = repetitions
    write_artifact(args.output, artifact)
    print(
        f"variant={args.variant} prefix_tokens={prefix_tokens} "
        f"requests={args.requests} max_tokens={args.max_tokens}",
        flush=True,
    )

    for request_index in range(1, args.requests + 1):
        prompt = prefix + INSTRUCTION.format(request_index=request_index)
        print(f"request={request_index} starting", flush=True)
        request_started = time.perf_counter()
        try:
            result = stream_completion(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
                request_index,
            )
        except Exception as error:
            result = {
                "request_index": request_index,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(
                    time.perf_counter() - request_started,
                    3,
                ),
            }
            artifact["results"].append(result)
            artifact["completed_unix"] = time.time()
            artifact["passed"] = False
            write_artifact(args.output, artifact)
            print(
                f"request={request_index} failed error={result['error']}",
                flush=True,
            )
            return 1

        artifact["results"].append(result)
        write_artifact(args.output, artifact)
        if not result["ok"]:
            artifact["completed_unix"] = time.time()
            artifact["passed"] = False
            write_artifact(args.output, artifact)
            print(
                f"request={request_index} failed error={result['error']}",
                flush=True,
            )
            return 1
        print(
            f"request={request_index} ok elapsed={result['elapsed_seconds']} "
            f"completion_tokens={result['usage'].get('completion_tokens')} "
            f"finish_reason={result['finish_reason']}",
            flush=True,
        )

    artifact["completed_unix"] = time.time()
    artifact["passed"] = True
    write_artifact(args.output, artifact)
    print("probe_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
