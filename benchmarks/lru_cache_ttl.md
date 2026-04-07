# Benchmark: LRU Cache with TTL

**Difficulty:** Hard
**Expected tests:** 6
**Skills tested:** Data structures (doubly-linked list + hash map), time-based expiry, lazy cleanup, test mocking

## Prompt

```
Implement an LRU (Least Recently Used) cache in Python with time-based expiration. Requirements:

1. Class TTLCache with __init__(self, capacity: int, default_ttl: float) where capacity is max items and default_ttl is seconds until expiry
2. get(key: str) -> Optional[Any] — return value if exists and not expired, else None. Accessing a key makes it most-recently-used.
3. put(key: str, value: Any, ttl: Optional[float] = None) — insert/update. If at capacity, evict the least-recently-used non-expired item. If all items are expired, clear them all first. Custom ttl overrides default.
4. delete(key: str) -> bool — remove key, return True if it existed
5. size() -> int — return count of non-expired items (lazy cleanup: expired items removed on access)
6. All operations must be O(1) average time. Use a doubly-linked list + hash map internally — do NOT use OrderedDict.
7. Use time.monotonic() for time tracking
8. Include type hints throughout and a brief docstring on each method
9. Write 6 pytest tests covering: basic get/put, capacity eviction (LRU order), TTL expiry, custom per-key TTL, delete, and size with mixed expired/valid items. Use unittest.mock.patch to mock time.monotonic for deterministic time control in tests — do NOT use time.sleep.
```

## What Makes This a Good Benchmark

This is the hardest benchmark in the suite because it requires coordinating multiple interacting systems:

- **Doubly-linked list** for O(1) move-to-front and eviction from tail
- **Hash map** for O(1) key lookup
- **Time tracking** with proper expiry semantics
- **Lazy cleanup** — expired items aren't removed until accessed or eviction is needed
- **size()** must return count of non-expired items, which requires scanning or tracking
- **Test mocking** — must use `unittest.mock.patch` on `time.monotonic`, not `time.sleep`

The interaction between LRU eviction and TTL expiry is where most models fail — when capacity is reached but some items are expired, the eviction logic must handle both cases correctly.

## Common Failure Modes Observed

1. **Missing constructor args** (Nemotron 30B): Sentinel nodes created as `_Node("", "")` missing required `expire_at` parameter — every test crashes
2. **Lazy cleanup in size()** (Gemma Q6_K): `size()` returns `len(cache)` which includes expired-but-unaccessed items that haven't been cleaned up
3. **Broken eviction** (Qwen 9B): `_evict()` checks `node in self._map` (object identity against dict values) instead of `node.key in self._map` — eviction silently fails
4. **Stale count** (Qwen 9B): `size()` returns `len(self._map)` without filtering expired entries

## Evaluation Criteria

- Does the code import without errors?
- Do all 6 tests pass when run with `pytest -v`?
- Time mocking must use `unittest.mock.patch`, not `time.sleep`
- No manual fixes allowed

## Why Only Qwen 35B Passed

The Qwen 35B was the only model to achieve 6/6. It used a `_valid_count` counter that tracks non-expired items separately from the dict size, and a `MockTime` helper class with `side_effect` for clean time advancement. This shows stronger architectural reasoning — anticipating the expired-item counting edge case before writing the implementation.
