#!/usr/bin/env python3
"""Stream one long tool call and time the server's /health meanwhile.

The request forces a write_file tool call whose content argument is a long
file, streams it, and reassembles the argument deltas. A second thread polls
/health every 0.5 s. A tool parser that re-parses the whole argument on each
delta shows up as long gaps between streamed chunks and slow /health replies
while the argument grows; the reassembled argument must also parse as JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file to disk, replacing it if it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "The complete file contents."},
            },
            "required": ["path", "content"],
        },
    },
}

PROMPT = (
    "Use write_file once to create inventory.py: a complete, self-contained Python module of about "
    "{lines} lines implementing an in-memory inventory system with dataclasses, a repository class, "
    "CSV import and export, search, validation errors, and a unittest suite at the bottom with at "
    "least 30 test methods. Put the whole file in a single write_file call."
)


def poll_health(base: str, stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        t = time.perf_counter()
        try:
            urllib.request.urlopen(base + "/health", timeout=60).read()
            out.append(time.perf_counter() - t)
        except Exception:  # noqa: BLE001 - a failed probe counts as the timeout
            out.append(60.0)
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="server root, without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lines", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT.format(lines=args.lines)}],
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "write_file"}},
        "max_tokens": args.max_tokens,
        "stream": True,
        "reasoning_effort": "none",
    }
    health: list[float] = []
    stop = threading.Event()
    poller = threading.Thread(target=poll_health, args=(base, stop, health), daemon=True)
    poller.start()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    args_text, gaps, finish = "", [], None
    start = last = time.perf_counter()
    first_arg = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            now = time.perf_counter()
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for call in (choice.get("delta") or {}).get("tool_calls") or []:
                    piece = (call.get("function") or {}).get("arguments") or ""
                    if piece:
                        if first_arg is None:
                            first_arg = now - start
                        else:
                            gaps.append(now - last)
                        args_text += piece
                        last = now
    total = time.perf_counter() - start
    stop.set()
    poller.join()
    try:
        parsed = json.loads(args_text)
        valid = isinstance(parsed, dict) and isinstance(parsed.get("content"), str)
        content_chars = len(parsed.get("content", "")) if valid else 0
    except json.JSONDecodeError:
        valid, content_chars = False, 0
    result = {
        "total_s": round(total, 2),
        "first_argument_s": round(first_arg, 2) if first_arg is not None else None,
        "argument_chars": len(args_text),
        "content_chars": content_chars,
        "arguments_valid_json": valid,
        "finish_reason": finish,
        "argument_deltas": len(gaps) + (1 if first_arg is not None else 0),
        "max_gap_s": round(max(gaps), 3) if gaps else None,
        "p99_gap_s": round(sorted(gaps)[int(0.99 * (len(gaps) - 1))], 3) if gaps else None,
        "health_probes": len(health),
        "health_p50_s": round(statistics.median(health), 3) if health else None,
        "health_max_s": round(max(health), 3) if health else None,
        "argument_chars_per_s": round(len(args_text) / (total - (first_arg or 0)), 1) if args_text else 0,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
