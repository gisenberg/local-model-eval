#!/usr/bin/env python3
"""Run a deterministic near-250K-token retrieval probe through an OAI server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FILLER = (
    "The archive entry describes an ordinary shipment, its inspection date, "
    "and the shelf where the paper record was filed.\n"
)
NEEDLE_VALUE = "739184"
NEEDLE = (
    "\nCritical retrieval note: the verification number for project LANTERN "
    f"is {NEEDLE_VALUE}. Retain this exact number.\n"
)
QUESTION = (
    "\nWhat is the verification number for project LANTERN? "
    "Reply with only the six digits."
)


def post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
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


def build_prompt(repetitions: int) -> str:
    left = repetitions * 3 // 4
    right = repetitions - left
    return FILLER * left + NEEDLE + FILLER * right + QUESTION


def token_count(
    base_url: str,
    content: str,
    timeout: int,
    tokenizer_api: str,
) -> int:
    if tokenizer_api == "tabby":
        endpoint = "/v1/token/encode"
        body = {"text": content, "add_bos_token": True}
    elif tokenizer_api == "vllm":
        endpoint = "/tokenize"
        body = {"prompt": content, "add_special_tokens": True}
    else:
        endpoint = "/tokenize"
        body = {"content": content, "add_special": True}

    response = post_json(f"{base_url.rstrip('/')}{endpoint}", body, timeout)
    length = response.get("length")
    if isinstance(length, int):
        return length
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected tokenize response: {response}")
    return len(tokens)


def fit_prompt(
    base_url: str,
    target_tokens: int,
    timeout: int,
    tokenizer_api: str,
) -> tuple[str, int, int]:
    low = 0
    high = max(1, target_tokens // 8)
    while token_count(base_url, build_prompt(high), timeout, tokenizer_api) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        count = token_count(base_url, build_prompt(middle), timeout, tokenizer_api)
        if count <= target_tokens:
            low = middle
        else:
            high = middle

    prompt = build_prompt(low)
    return prompt, token_count(base_url, prompt, timeout, tokenizer_api), low


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="local")
    parser.add_argument("--target-tokens", type=int, default=250_000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--tokenizer-api",
        choices=("llama", "tabby", "vllm"),
        default="llama",
        help="Tokenizer endpoint dialect exposed by the server.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt, raw_tokens, repetitions = fit_prompt(
        args.base_url,
        args.target_tokens,
        args.timeout,
        args.tokenizer_api,
    )
    started = time.perf_counter()
    response = post_json(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.timeout,
    )
    elapsed = time.perf_counter() - started
    message = response["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    usage = response.get("usage") or {}
    passed = content == NEEDLE_VALUE

    artifact = {
        "base_url": args.base_url,
        "model": args.model,
        "target_raw_tokens": args.target_tokens,
        "raw_tokens": raw_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "filler_repetitions": repetitions,
        "needle_depth": 0.75,
        "expected": NEEDLE_VALUE,
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "elapsed_seconds": round(elapsed, 3),
        "passed": passed,
        "response": response,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"Raw prompt tokens: {raw_tokens}")
    print(f"API prompt tokens: {usage.get('prompt_tokens')}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Answer: {content!r}")
    print(f"Retrieval: {'PASS' if passed else 'FAIL'}")
    print(f"Saved: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
