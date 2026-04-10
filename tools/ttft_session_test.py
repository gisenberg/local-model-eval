#!/usr/bin/env python3
"""
Test whether requests.Session() (connection pooling/reuse) eliminates the
~2 second per-request overhead we measured on Windows.

Compares three modes against the same Windows llama-server:
  1. Fresh requests.post() per request (current ttft_isolation_test.py behavior)
  2. requests.Session() with explicit reuse (single connection across all requests)
  3. urllib3 PoolManager directly (bypasses requests entirely)

If mode 2 or 3 is fast, the issue is connection-establishment overhead
in requests.post() on Windows, not anything fundamental to Python sockets.
"""

import json
import statistics
import time

import requests
import urllib3

URL = "http://localhost:8080"
PROMPT = (
    "Write a Python function to compute the factorial of n recursively. "
    "Include type hints, a docstring, and one pytest test."
)
PAYLOAD = {
    "model": "local-model",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": True,
}
WARMUP = 2
RUNS = 5


def measure_with_fresh_post():
    """Each request makes a fresh requests.post() call (new connection each time)."""
    t_send = time.perf_counter()
    resp = requests.post(
        f"{URL}/v1/chat/completions",
        json=PAYLOAD,
        stream=True,
        timeout=120,
    )
    return _consume_first_token(resp, t_send)


def measure_with_session(session):
    """Reuse a single requests.Session() across all requests."""
    t_send = time.perf_counter()
    resp = session.post(
        f"{URL}/v1/chat/completions",
        json=PAYLOAD,
        stream=True,
        timeout=120,
    )
    return _consume_first_token(resp, t_send)


def measure_with_urllib3(http):
    """Use urllib3.PoolManager directly, bypassing requests."""
    t_send = time.perf_counter()
    resp = http.request(
        "POST",
        f"{URL}/v1/chat/completions",
        body=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        preload_content=False,
    )
    first_token_at = None
    for raw in resp.read_chunked(decode_content=True):
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        for sub in line.split("\n"):
            sub = sub.strip()
            if not sub.startswith("data: "):
                continue
            payload = sub[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                first_token_at = time.perf_counter()
                resp.release_conn()
                return first_token_at - t_send
    return None


def _consume_first_token(resp, t_send):
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if choices and choices[0].get("delta", {}).get("content"):
            ttft = time.perf_counter() - t_send
            resp.close()
            return ttft
    return None


def run_mode(name, measure_fn, *fn_args):
    print(f"\n=== {name} ===")
    for i in range(WARMUP):
        try:
            t = measure_fn(*fn_args)
            print(f"  warmup {i+1}/{WARMUP}: {t:.3f}s")
        except Exception as e:
            print(f"  warmup {i+1}/{WARMUP}: ERROR {e}")

    times = []
    for i in range(RUNS):
        try:
            t = measure_fn(*fn_args)
            print(f"  run {i+1}/{RUNS}:    {t:.3f}s")
            times.append(t)
        except Exception as e:
            print(f"  run {i+1}/{RUNS}:    ERROR {e}")

    if times:
        print(f"  -> mean={statistics.mean(times):.3f}s  median={statistics.median(times):.3f}s  "
              f"min={min(times):.3f}s  max={max(times):.3f}s")
    return times


def main():
    print("=== TTFT Session Test (Windows) ===")
    print(f"Server: {URL}")
    print(f"Warmup: {WARMUP} req | Timed: {RUNS} req per mode")

    # Mode 1: fresh requests.post each call
    fresh = run_mode("Mode 1: fresh requests.post() each call", measure_with_fresh_post)

    # Mode 2: shared requests.Session
    session = requests.Session()
    sess = run_mode("Mode 2: shared requests.Session()", measure_with_session, session)
    session.close()

    # Mode 3: urllib3 PoolManager directly
    http = urllib3.PoolManager(num_pools=1, maxsize=1)
    pool = run_mode("Mode 3: urllib3.PoolManager direct", measure_with_urllib3, http)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (mean TTFT)")
    print("=" * 60)
    if fresh:
        print(f"  Mode 1 (fresh requests.post):  {statistics.mean(fresh):.3f}s")
    if sess:
        print(f"  Mode 2 (requests.Session):     {statistics.mean(sess):.3f}s")
    if pool:
        print(f"  Mode 3 (urllib3 PoolManager):  {statistics.mean(pool):.3f}s")

    if fresh and sess:
        delta = statistics.mean(fresh) - statistics.mean(sess)
        print(f"\n  Session saves: {delta:.3f}s ({delta / statistics.mean(fresh) * 100:.0f}% reduction)")


if __name__ == "__main__":
    main()
