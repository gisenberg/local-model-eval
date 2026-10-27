#!/usr/bin/env python3
"""Run concurrent long-context needle probes, then measure decode at depth.

Each session gets its own filler and needle, so no prompt prefix is shared
between sessions and every session must hold its full context in KV at once.
The filler is dense with distractor numbers so the needle is not the only
six-digit value in the prompt.

Phase 1 sends every needle prompt concurrently and checks exact retrieval.
Phase 2 re-sends each prompt with a long-form follow-up so the server reuses
the cached prefix, and measures streaming decode throughput at full depth
while all sessions decode together.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SHELVES = ("north", "south", "east", "west", "upper", "lower", "annex", "vault")
GOODS = ("valves", "cable", "lenses", "bearings", "fabric", "resin", "tiles", "gaskets")
QUESTION = "\nWhat is the verification number for project {name}? Reply with only the six digits."
FOLLOW_UP = (
    "\nIgnore the retrieval task. Write a detailed, roughly 600-word essay about "
    "how warehouse archives are organized and audited. Do not quote any numbers."
)
PROJECTS = ("LANTERN", "HARBOR", "QUARTZ", "MERIDIAN", "FALCON", "ORCHID", "SUMMIT", "CINDER")


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


def filler_lines(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        lines.append(
            f"Entry {rng.randrange(100000, 999999)}: {rng.randrange(2, 400)} crates of "
            f"{rng.choice(GOODS)} inspected on day {rng.randrange(1, 365)}, filed on the "
            f"{rng.choice(SHELVES)} shelf, bay {rng.randrange(1, 90)}.\n"
        )
    return lines


def build_prompt(lines: list[str], count: int, needle: str, depth: float, suffix: str) -> str:
    left = int(count * depth)
    return "".join(lines[:left]) + needle + "".join(lines[left:count]) + suffix


def token_count(base_url: str, model: str, content: str, timeout: int) -> int:
    # llama.cpp reads "content"; vLLM reads "model" and "prompt". Both return
    # the token list under "tokens" and ignore the other server's fields.
    response = post_json(
        f"{base_url}/tokenize",
        {"content": content, "add_special": True, "model": model, "prompt": content},
        timeout,
    )
    return len(response["tokens"])


def fit_count(
    base_url: str, model: str, lines: list[str], needle: str, depth: float, suffix: str,
    target: int, timeout: int,
) -> int:
    low, high = 0, len(lines)
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(base_url, model, build_prompt(lines, middle, needle, depth, suffix), timeout) <= target:
            low = middle
        else:
            high = middle
    return low


def stream_chat(base_url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = dict(body, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    if first is None:
                        first = time.perf_counter()
                    content.append(delta.get("content") or "")
    ended = time.perf_counter()
    return {
        "ttft_s": round((first or ended) - started, 3),
        "elapsed_s": round(ended - started, 3),
        "content": "".join(content).strip(),
        "usage": usage,
        "timings": timings,
    }


def run_parallel(jobs: list[Any]) -> list[Any]:
    results: list[Any] = [None] * len(jobs)

    def worker(index: int) -> None:
        try:
            results[index] = jobs[index]()
        except Exception as error:  # noqa: BLE001 - record and report every failure
            results[index] = {"error": repr(error)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(jobs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8097")
    parser.add_argument("--model", default="local")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--target-tokens", type=int, default=500_000)
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument("--decode-tokens", type=int, default=1024)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    sessions = []
    for index in range(args.sessions):
        rng = random.Random(1000 + index)
        value = str(rng.randrange(100000, 999999))
        name = PROJECTS[index % len(PROJECTS)]
        needle = (
            f"\nCritical retrieval note: the verification number for project {name} "
            f"is {value}. Retain this exact number.\n"
        )
        # About 30 tokens per filler line; oversize the pool before fitting.
        lines = filler_lines(seed=index, count=args.target_tokens // 20)
        question = QUESTION.format(name=name)
        count = fit_count(base_url, args.model, lines, needle, args.depth, question, args.target_tokens, args.timeout)
        sessions.append({"name": name, "expected": value, "lines": lines, "count": count, "needle": needle})
        print(f"session {index}: project {name}, {count} filler lines", flush=True)

    def needle_job(session: dict[str, Any]) -> Any:
        prompt = build_prompt(
            session["lines"], session["count"], session["needle"], args.depth,
            QUESTION.format(name=session["name"]),
        )
        return lambda: stream_chat(base_url, {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }, args.timeout)

    started = time.perf_counter()
    needle_results = run_parallel([needle_job(session) for session in sessions])
    phase1_s = time.perf_counter() - started

    records = []
    for index, (session, result) in enumerate(zip(sessions, needle_results)):
        got = (result.get("content") or "")[:32]
        record = {
            "session": index,
            "project": session["name"],
            "expected": session["expected"],
            "got": got,
            "passed": got == session["expected"],
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            "ttft_s": result.get("ttft_s"),
            "elapsed_s": result.get("elapsed_s"),
            "error": result.get("error"),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    decode_records = []
    if not args.skip_decode:
        def decode_job(session: dict[str, Any]) -> Any:
            prompt = build_prompt(
                session["lines"], session["count"], session["needle"], args.depth,
                QUESTION.format(name=session["name"]),
            )
            return lambda: stream_chat(base_url, {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": session["expected"]},
                    {"role": "user", "content": FOLLOW_UP},
                ],
                "temperature": 0,
                "max_tokens": args.decode_tokens,
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }, args.timeout)

        for index, result in enumerate(run_parallel([decode_job(session) for session in sessions])):
            timings = result.get("timings") or {}
            usage = result.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            completion = timings.get("predicted_n", usage.get("completion_tokens"))
            decode_tok_s = timings.get("predicted_per_second")
            if decode_tok_s is None and completion and "elapsed_s" in result:
                # vLLM reports no timings, so measure from the first streamed token.
                decode_window = result["elapsed_s"] - result["ttft_s"]
                decode_tok_s = round(completion / decode_window, 2) if decode_window > 0 else None
            cached = timings.get("cache_n", details.get("cached_tokens"))
            prompt_tokens = usage.get("prompt_tokens")
            new_prompt = timings.get("prompt_n")
            if new_prompt is None and prompt_tokens is not None and cached is not None:
                new_prompt = prompt_tokens - cached
            record = {
                "session": index,
                "cached_tokens": cached,
                "new_prompt_tokens": new_prompt,
                "ttft_s": result.get("ttft_s"),
                "completion_tokens": completion,
                "decode_tok_s": decode_tok_s,
                "error": result.get("error"),
            }
            decode_records.append(record)
            print(json.dumps(record), flush=True)

    artifact = {
        "base_url": base_url,
        "model": args.model,
        "sessions": args.sessions,
        "target_tokens": args.target_tokens,
        "depth": args.depth,
        "phase1_wall_s": round(phase1_s, 3),
        "needles": records,
        "decode_at_depth": decode_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    passed = all(record["passed"] for record in records)
    print(f"Retrieval: {'PASS' if passed else 'FAIL'}; saved {args.output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
