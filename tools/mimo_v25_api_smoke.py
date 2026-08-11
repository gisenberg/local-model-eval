#!/usr/bin/env python3
"""Exercise MiMo-V2.5 generation, reasoning, and tool calls through chat completions."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, body: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    result["_client_elapsed_s"] = round(time.perf_counter() - started, 4)
    return result


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    return response["choices"][0]["message"]


def reasoning_text(message: dict[str, Any]) -> str:
    return (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("analysis")
        or ""
    )


def parse_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call["function"].get("arguments", {})
    if isinstance(arguments, str):
        return json.loads(arguments)
    if isinstance(arguments, dict):
        return arguments
    raise TypeError(f"unexpected tool arguments type: {type(arguments).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--reasoning",
        choices=("required", "forbidden", "optional"),
        default="required",
        help="Expected reasoning_content contract for the served preset.",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": args.reasoning != "forbidden",
            "drop_thinking": False,
        },
    }

    basic = post_json(
        endpoint,
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly MIMO_OK and no other text.",
                }
            ],
        },
    )
    basic_message = response_message(basic)
    basic_content = (basic_message.get("content") or "").strip()

    reasoning = post_json(
        endpoint,
        {
            **common,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Compute 17 * 24. Think briefly, verify it a second way, "
                        "then give the concise final answer."
                    ),
                }
            ],
        },
    )
    reasoning_message = response_message(reasoning)
    reasoning_content = reasoning_message.get("content") or ""
    reasoning_trace = reasoning_text(reasoning_message)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Look up one record by its exact identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "Exact record identifier.",
                        }
                    },
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Call lookup_record with record_id alpha-7. "
                "Do not guess the record value and do not answer directly."
            ),
        }
    ]
    tool_response = post_json(
        endpoint,
        {
            **common,
            "messages": tool_messages,
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    tool_message = response_message(tool_response)
    tool_calls = tool_message.get("tool_calls") or []

    tool_name = None
    tool_arguments: dict[str, Any] | None = None
    followup: dict[str, Any] | None = None
    followup_message: dict[str, Any] = {}
    if len(tool_calls) == 1:
        tool_name = tool_calls[0]["function"].get("name")
        tool_arguments = parse_arguments(tool_calls[0])
        tool_messages.extend(
            [
                tool_message,
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "name": tool_name,
                    "content": json.dumps(
                        {"record_id": "alpha-7", "value": 42},
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        followup = post_json(
            endpoint,
            {
                **common,
                "messages": tool_messages,
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        followup_message = response_message(followup)

    reasoning_check_name = {
        "required": "reasoning_preserved",
        "forbidden": "reasoning_suppressed",
        "optional": "reasoning_accepted",
    }[args.reasoning]
    reasoning_check = {
        "required": bool(reasoning_trace.strip()),
        "forbidden": not bool(reasoning_trace.strip()),
        "optional": True,
    }[args.reasoning]
    checks = {
        "basic_generation": basic_content == "MIMO_OK",
        "reasoning_answer": "408" in reasoning_content,
        reasoning_check_name: reasoning_check,
        "single_tool_call": len(tool_calls) == 1,
        "tool_name": tool_name == "lookup_record",
        "tool_arguments": tool_arguments == {"record_id": "alpha-7"},
        "tool_followup": (
            followup is not None
            and "42" in (followup_message.get("content") or "")
            and not (followup_message.get("tool_calls") or [])
        ),
    }
    artifact = {
        "base_url": args.base_url,
        "model": args.model,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "responses": {
            "basic": basic,
            "reasoning": reasoning,
            "tool_call": tool_response,
            "tool_followup": followup,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"TOTAL: {artifact['passed']}/{artifact['total']}")
    print(f"Saved: {args.output}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
