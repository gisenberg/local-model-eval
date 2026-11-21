#!/usr/bin/env python3
"""Check whether idle agents keep their prefix cache while other agents run.

Each session sends one long prompt, then later continues the same
conversation (its earlier reply plus a new user turn), the way an agent
harness does. The continuation should be served almost entirely from the
prefix cache. The script records cached_tokens for every continuation, so a
cache that drops an idle session shows up as a near-zero hit.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

WORDS = (
    "alpha bravo cedar delta ember fjord granite harbor island jasper kestrel "
    "lantern meadow nickel orchard pebble quartz raven summit timber umber "
    "violet willow xenon yonder zephyr"
).split()


def post(base_url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def filler(seed: int, approx_tokens: int) -> str:
    rng = random.Random(seed)
    lines = []
    # Each line is about 13 tokens with this vocabulary.
    for index in range(approx_tokens // 13):
        lines.append(f"{index}: " + " ".join(rng.choice(WORDS) for _ in range(8)))
    return "\n".join(lines)


def turn(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    extra_body: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post(
        base_url,
        {
            "model": model,
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0,
            **extra_body,
        },
        timeout,
    )
    elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "elapsed_s": round(elapsed, 2),
        "reply": response["choices"][0]["message"].get("content") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8094")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1000,
        help="Filler seed for session 0; change it to make prompts distinct from earlier runs.",
    )
    parser.add_argument("--extra-body", default='{"reasoning_effort": "none"}')
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extra_body = json.loads(args.extra_body)

    conversations = []
    events = []
    for session in range(args.sessions):
        messages = [
            {
                "role": "user",
                "content": filler(args.seed_offset + session, args.tokens)
                + f"\n\nSession {session}: reply with the word ready.",
            }
        ]
        result = turn(args.base_url, args.model, messages, extra_body, args.timeout)
        messages.append({"role": "assistant", "content": result["reply"]})
        conversations.append(messages)
        event = {"session": session, "round": 0, **result}
        events.append(event)
        print(json.dumps(event), flush=True)

    for round_index in range(1, args.rounds + 1):
        for session, messages in enumerate(conversations):
            messages.append(
                {"role": "user", "content": f"Round {round_index}: reply with the word ok."}
            )
            result = turn(args.base_url, args.model, messages, extra_body, args.timeout)
            messages.append({"role": "assistant", "content": result["reply"]})
            prompt = result["prompt_tokens"] or 0
            cached = result["cached_tokens"] or 0
            event = {
                "session": session,
                "round": round_index,
                **result,
                "hit_ratio": round(cached / prompt, 4) if prompt else None,
            }
            events.append(event)
            print(json.dumps(event), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "model": args.model,
                "sessions": args.sessions,
                "tokens": args.tokens,
                "rounds": args.rounds,
                "seed_offset": args.seed_offset,
                "extra_body": extra_body,
                "events": events,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
