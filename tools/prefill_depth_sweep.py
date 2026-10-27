#!/usr/bin/env python3
"""Measure prefill throughput against context depth on a vLLM server.

Each cold request is a distinct prompt with no cached prefix. After each cold
request, a fresh ~16K-token block is appended to the now-cached prompt to
measure the marginal prefill rate an agent pays per turn at that depth.
Point --base-url at the vLLM server itself: llama-swap does not proxy /tokenize.
"""

import argparse
import json
import random
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--sizes", default="8192,32768,131072,262144,524288,786432")
parser.add_argument("--append-tokens", type=int, default=16_384)
parser.add_argument("--output", required=True)
args = parser.parse_args()
BASE = args.base_url.rstrip("/")
MODEL = args.model
GOODS = ("valves", "cable", "lenses", "bearings", "fabric", "resin", "tiles", "gaskets")


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def lines(seed, n):
    rng = random.Random(seed)
    return [f"Entry {rng.randrange(100000, 999999)}: {rng.randrange(2, 400)} crates of {rng.choice(GOODS)} "
            f"inspected on day {rng.randrange(1, 365)}, bay {rng.randrange(1, 90)}.\n" for _ in range(n)]


def ntok(text):
    return len(post("/tokenize", {"model": MODEL, "prompt": text})["tokens"])


def text_of(seed, target):
    pool = lines(seed, target // 15)
    lo, hi = 0, len(pool)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ntok("".join(pool[:mid])) <= target:
            lo = mid
        else:
            hi = mid
    return "".join(pool[:lo])


def ttft(messages):
    body = {"model": MODEL, "messages": messages, "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.perf_counter()
    d = post("/v1/chat/completions", body)
    dt = time.perf_counter() - t
    u = d["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return dt, u["prompt_tokens"], cached


results = {"cold": [], "append": []}
# Warm the route.
ttft([{"role": "user", "content": "hi"}])

for i, size in enumerate(int(value) for value in args.sizes.split(",")):
    content = text_of(100 + i, size) + "\nSummarize in one word."
    dt, pt, cached = ttft([{"role": "user", "content": content}])
    rec = {"prompt_tokens": pt, "cached": cached, "ttft_s": round(dt, 2), "tok_s": round((pt - cached) / dt)}
    results["cold"].append(rec)
    print("cold", rec, flush=True)

    # Append a fresh 16K block on top of the now-cached prefix: the marginal cost at this depth.
    if size >= 32_768:
        extra = text_of(900 + i, args.append_tokens)
        dt2, pt2, cached2 = ttft([{"role": "user", "content": content + "\n" + extra + "\nSummarize in one word."}])
        rec2 = {"depth": cached2, "new_tokens": pt2 - cached2, "ttft_s": round(dt2, 2), "tok_s": round((pt2 - cached2) / dt2)}
        results["append"].append(rec2)
        print("append", rec2, flush=True)

with open(args.output, "w") as handle:
    json.dump(results, handle, indent=2)
