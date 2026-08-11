#!/usr/bin/env python3
"""Validate Laguna's reasoning and tool-call history protocol against vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def complete(port: int, body: dict) -> dict:
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=body,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--reasoning-max-tokens", type=int, default=4096)
    parser.add_argument("--tool-max-tokens", type=int, default=2048)
    args = parser.parse_args()
    raw_responses: dict[str, dict | None] = {}

    reasoning_prompt = (
        "Solve this step by step before giving the final answer: find all integers "
        "n such that n^2 + n + 41 is divisible by n + 2."
    )
    reasoning_probe = complete(
        args.port,
        {
            "model": args.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": reasoning_prompt,
                },
            ],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": args.reasoning_max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )
    raw_responses["reasoning_probe"] = reasoning_probe
    reasoning_message = reasoning_probe["choices"][0]["message"]
    reasoning_content = (
        reasoning_message.get("reasoning_content")
        or reasoning_message.get("reasoning")
        or ""
    )
    reasoning_parsed = bool(reasoning_content)

    preserved_follow_up = complete(
        args.port,
        {
            "model": args.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": reasoning_prompt,
                },
                {
                    "role": "assistant",
                    "content": reasoning_message.get("content") or "",
                    "reasoning_content": reasoning_content,
                },
                {
                    "role": "user",
                    "content": "Restate only the conclusion in one short sentence.",
                },
            ],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": 512,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )
    raw_responses["preserved_follow_up"] = preserved_follow_up
    preserved_message = preserved_follow_up["choices"][0]["message"]
    preserved_follow_up_nonempty = bool(
        preserved_message.get("content")
        or preserved_message.get("reasoning_content")
        or preserved_message.get("reasoning")
    )

    tool = {
        "type": "function",
        "function": {
            "name": "inspect_file",
            "description": "Read a repository file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
    first = complete(
        args.port,
        {
            "model": args.served_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Call exactly one tool at a time and wait for its result. "
                        "Never include multiple tool calls in one response."
                    ),
                },
                {
                    "role": "user",
                    "content": "Inspect pyproject.toml with the available tool.",
                },
            ],
            "tools": [tool],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "max_tokens": args.tool_max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )
    raw_responses["first_tool_response"] = first
    first_message = first["choices"][0]["message"]
    tool_calls = first_message.get("tool_calls") or []
    tool_reasoning_content = (
        first_message.get("reasoning_content")
        or first_message.get("reasoning")
        or ""
    )

    second = None
    second_message: dict = {}
    if len(tool_calls) == 1:
        assistant_message = {
            "role": "assistant",
            "content": first_message.get("content") or "",
            "reasoning_content": tool_reasoning_content,
            "tool_calls": tool_calls,
        }
        second = complete(
            args.port,
            {
                "model": args.served_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Call exactly one tool at a time and wait for its result. "
                            "After receiving the tool result, answer in one sentence without another tool call."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Inspect pyproject.toml with the available tool.",
                    },
                    assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": "[project]\nname = 'local-model-eval'\n",
                    },
                ],
                "tools": [tool],
                "tool_choice": "none",
                "parallel_tool_calls": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 20,
                "max_tokens": args.tool_max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                },
            },
        )
        second_message = second["choices"][0]["message"]
    raw_responses["tool_follow_up"] = second
    tool_follow_up_nonempty = bool(
        second_message.get("content")
        or second_message.get("reasoning_content")
        or second_message.get("reasoning")
    )

    protocol_pass = (
        reasoning_parsed
        and preserved_follow_up_nonempty
        and len(tool_calls) == 1
        and tool_follow_up_nonempty
    )
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(raw_responses, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "protocol_pass": protocol_pass,
                "generation_content_chars": len(reasoning_message.get("content") or ""),
                "generation_reasoning_chars": len(reasoning_content),
                "generation_finish_reason": reasoning_probe["choices"][0].get(
                    "finish_reason"
                ),
                "generation_completion_tokens": reasoning_probe.get("usage", {}).get(
                    "completion_tokens"
                ),
                "reasoning_parsed": reasoning_parsed,
                "reasoning_history_accepted": preserved_follow_up_nonempty,
                "preserved_follow_up_finish_reason": preserved_follow_up["choices"][0].get(
                    "finish_reason"
                ),
                "preserved_follow_up_completion_tokens": preserved_follow_up.get(
                    "usage", {}
                ).get("completion_tokens"),
                "preserved_follow_up_content_chars": len(
                    preserved_message.get("content") or ""
                ),
                "preserved_follow_up_reasoning_chars": len(
                    preserved_message.get("reasoning_content")
                    or preserved_message.get("reasoning")
                    or ""
                ),
                "first_tool_calls": len(tool_calls),
                "first_tool_reasoning_chars": len(tool_reasoning_content),
                "tool_history_accepted": tool_follow_up_nonempty,
                "follow_up_finish_reason": (
                    second["choices"][0].get("finish_reason") if second else None
                ),
                "follow_up_tool_calls": len(second_message.get("tool_calls") or []),
                "follow_up_content_chars": len(second_message.get("content") or ""),
                "follow_up_reasoning_chars": len(
                    second_message.get("reasoning_content")
                    or second_message.get("reasoning")
                    or ""
                ),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
