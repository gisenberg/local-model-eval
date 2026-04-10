#!/usr/bin/env python3
"""Hard coding benchmarks: LRU Cache with TTL + A* Pathfinding."""

import json
import os
import time
import requests

BASE_URL = "http://localhost:1234"

BENCHMARKS = {
    "lru_cache": {
        "name": "LRU Cache with TTL",
        "prompt": """\
Implement an LRU (Least Recently Used) cache in Python with time-based expiration. Requirements:

1. Class `TTLCache` with `__init__(self, capacity: int, default_ttl: float)` where capacity is max items and default_ttl is seconds until expiry
2. `get(key: str) -> Optional[Any]` — return value if exists and not expired, else None. Accessing a key makes it most-recently-used.
3. `put(key: str, value: Any, ttl: Optional[float] = None)` — insert/update. If at capacity, evict the least-recently-used non-expired item. If all items are expired, clear them all first. Custom ttl overrides default.
4. `delete(key: str) -> bool` — remove key, return True if it existed
5. `size() -> int` — return count of non-expired items (lazy cleanup: expired items removed on access)
6. All operations must be O(1) average time. Use a doubly-linked list + hash map internally — do NOT use OrderedDict.
7. Use `time.monotonic()` for time tracking
8. Include type hints throughout and a brief docstring on each method
9. Write 6 pytest tests covering: basic get/put, capacity eviction (LRU order), TTL expiry, custom per-key TTL, delete, and size with mixed expired/valid items. Use `unittest.mock.patch` to mock `time.monotonic` for deterministic time control in tests — do NOT use `time.sleep`.""",
    },
    "astar": {
        "name": "A* Pathfinding",
        "prompt": """\
Implement A* pathfinding on a weighted 2D grid in Python. Requirements:

1. Class `AStarGrid` with `__init__(self, grid: List[List[int]])` where grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell)
2. `find_path(start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]` — return shortest path as list of (row, col) coordinates from start to end inclusive, or None if no path exists
3. Support 4-directional movement (up, down, left, right) — no diagonals
4. Use Manhattan distance as the heuristic
5. Handle edge cases: start == end (return [start]), start or end is a wall (return None), start or end out of bounds (raise ValueError)
6. The path must be optimal (minimum total cost)
7. Use a min-heap (heapq) for the open set
8. Include type hints throughout and a brief docstring on each method
9. Write 6 pytest tests covering: simple path on uniform grid, path around obstacles, weighted grid (path prefers lower-cost cells), no path exists (fully blocked), start equals end, and invalid coordinates. Assert both path validity and optimality (total cost).""",
    },
}

MODELS = [
    "google/gemma-4-26b-a4b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-9b",
]

OUTPUT_DIR = "experiments/hard_bench"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

for model_id in MODELS:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_id}")
    print(f"{'=' * 70}")

    # Load
    print("  Loading...", end="", flush=True)
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/models/load", json={
            "model": model_id, "context_length": 32768, "flash_attention": True,
        }, timeout=600)
        if resp.status_code == 200:
            print(" done")
        else:
            print(f" warn: {resp.status_code}, continuing")
    except Exception as e:
        print(f" warn: {e}, continuing")

    for bench_key, bench in BENCHMARKS.items():
        bench_name = bench["name"]
        prompt = bench["prompt"]
        print(f"\n  --- {bench_name} ---")
        print(f"  Generating...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16384,
                "temperature": 0,
            }, timeout=600)
            elapsed = time.perf_counter() - t0
            data = resp.json()
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            finish = data["choices"][0].get("finish_reason", "?")
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            comp_tokens = usage.get("completion_tokens", 0)
            tps = comp_tokens / elapsed if elapsed > 0 else 0

            print(
                f" {tps:.1f} tok/s | {comp_tokens} tok | {elapsed:.1f}s | "
                f"think: {len(reasoning)}c | content: {len(content)}c | "
                f"finish={finish}"
            )

            safe_model = model_id.replace("/", "_")
            fname = f"{OUTPUT_DIR}/{safe_model}_{bench_key}"
            with open(f"{fname}.md", "w", encoding="utf-8") as f:
                f.write(f"# {model_id} - {bench_name}\n\n")
                if reasoning:
                    f.write(f"## Thinking ({len(reasoning)} chars)\n\n{reasoning}\n\n")
                f.write(f"## Output\n\n{content}\n")
            with open(f"{fname}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_id, "benchmark": bench_key,
                    "elapsed_s": round(elapsed, 2), "tokens": comp_tokens,
                    "tok_per_sec": round(tps, 1), "finish_reason": finish,
                    "thinking_chars": len(reasoning), "content_chars": len(content),
                    "reasoning": reasoning, "content": content,
                }, f, indent=2, ensure_ascii=False)

            results.append({
                "model": model_id, "benchmark": bench_name,
                "tok_per_sec": round(tps, 1), "tokens": comp_tokens,
                "elapsed_s": round(elapsed, 1), "finish": finish,
                "thinking": len(reasoning), "content": len(content),
            })

        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "model": model_id, "benchmark": bench_name,
                "error": str(e),
            })

    # Unload
    print(f"\n  Unloading...", end="", flush=True)
    try:
        requests.post(f"{BASE_URL}/api/v1/models/unload",
                       json={"instance_id": model_id}, timeout=60)
        print(" done")
    except:
        print(" warn")

# Summary
print(f"\n{'=' * 100}")
print("GENERATION SUMMARY")
print(f"{'=' * 100}")
print(f"{'Model':35s} | {'Benchmark':25s} | {'Tok/s':>7s} | {'Tokens':>7s} | {'Time':>7s} | {'Think':>8s} | {'Content':>8s}")
print("-" * 100)
for r in results:
    if "error" in r:
        print(f"{r['model']:35s} | {r['benchmark']:25s} | ERROR: {r['error'][:40]}")
    else:
        print(
            f"{r['model']:35s} | {r['benchmark']:25s} | "
            f"{r['tok_per_sec']:>7.1f} | {r['tokens']:>7d} | "
            f"{r['elapsed_s']:>6.1f}s | {r['thinking']:>7d}c | {r['content']:>7d}c"
        )

print(f"\nOutputs saved to {OUTPUT_DIR}/")
