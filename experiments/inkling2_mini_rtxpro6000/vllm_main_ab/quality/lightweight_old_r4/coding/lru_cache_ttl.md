Here's a complete implementation. The cache uses a sentinel-based doubly-linked list (MRU at the head, LRU at the tail) plus a hash map for O(1) average get/put/delete. Expired entries are cleaned lazily and removed from the map when encountered, so the repeated-purge work is amortized O(1) per operation.

**`ttl_cache.py`**

```python
"""An LRU cache with per-entry TTL expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node holding one cache entry."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """An O(1)-average LRU cache with per-entry TTL expiration.

    Entries are ordered by recency: the head sentinel is the most-recently-used
    side, the tail sentinel is the least-recently-used side.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` items, each expiring
        `default_ttl` seconds after insertion/update (unless overridden)."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head = _Node("", None, 0.0)  # sentinel, MRU side
        self._tail = _Node("", None, 0.0)  # sentinel, LRU side
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired; otherwise None.
        A successful lookup marks the key as most-recently-used."""
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove(node)  # lazy cleanup of expired entry
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`, expiring `ttl` seconds from now
        (or `default_ttl` if `ttl` is None). When at capacity, expired entries
        are dropped first; if every entry is expired they are all cleared,
        otherwise the least-recently-used non-expired entry is evicted."""
        now = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl

        node = self._map.get(key)
        if node is not None:
            # Update existing entry (even if expired) in place.
            node.value = value
            node.expires_at = now + effective_ttl
            self._move_to_front(node)
            return

        if len(self._map) >= self.capacity:
            self._purge_expired(now)
            # Still full (some/all entries were non-expired): evict the LRU one.
            while len(self._map) >= self.capacity:
                self._remove(self._tail.prev)

        node = _Node(key, value, now + effective_ttl)
        self._insert_front(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """Remove `key` if present; return True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items (expired items are dropped)."""
        now = time.monotonic()
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            if node.expires_at <= now:
                self._remove(node)
            node = nxt
        return len(self._map)

    # ---- internal helpers ----

    def _insert_front(self, node: _Node) -> None:
        """Place `node` right after the head sentinel (most-recently-used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_front(self, node: _Node) -> None:
        """Promote `node` to the most-recently-used position."""
        self._unlink(node)
        self._insert_front(node)

    def _remove(self, node: _Node) -> None:
        """Unlink `node` and drop it from the hash map."""
        self._unlink(node)
        del self._map[node.key]

    def _purge_expired(self, now: float) -> None:
        """Drop every expired node from the list/map (amortized O(1) each)."""
        node = self._tail.prev
        while node is not self._head:
            prev = node.prev
            if node.expires_at <= now:
                self._remove(node)
            node = prev
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. Time is controlled by patching time.monotonic."""

from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@patch("time.monotonic")
def test_basic_get_put(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("missing") is None
    cache.put("a", 10)  # update existing key
    assert cache.get("a") == 10
    assert cache.size() == 2


@patch("time.monotonic")
def test_capacity_evicts_lru_order(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(3, 60.0)
    cache.put("a", 1)  # LRU
    cache.put("b", 2)
    cache.put("c", 3)  # MRU
    cache.get("a")     # a -> MRU; LRU order is now: b, c, a
    cache.put("d", 4)  # evicts b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.size() == 3


@patch("time.monotonic")
def test_ttl_expiry(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    clock.advance(11.0)
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.size() == 0


@patch("time.monotonic")
def test_custom_ttl_overrides_default(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(3, 100.0)
    cache.put("short", "x", ttl=5.0)
    cache.put("long", "y")  # default ttl = 100.0
    clock.advance(10.0)
    assert cache.get("short") is None
    assert cache.get("long") == "y"


@patch("time.monotonic")
def test_delete(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False
    assert cache.size() == 0


@patch("time.monotonic")
def test_size_counts_only_non_expired(mock_now):
    clock = FakeClock()
    mock_now.side_effect = clock
    cache = TTLCache(5, 10.0)
    cache.put("a", 1, ttl=20.0)  # valid
    cache.put("b", 2, ttl=5.0)   # expires
    cache.put("c", 3, ttl=20.0)  # valid
    clock.advance(10.0)
    assert cache.size() == 2
    assert cache.get("b") is None
    assert cache.size() == 2
```

**Design notes**

- **Linked list + map, not OrderedDict**: sentinel head/tail nodes make insertion, unlinking, and re-ordering O(1) with no edge cases; the hash map gives O(1) lookup.
- **Lazy cleanup**: expired nodes are removed on `get`, `size`, and at capacity-check time in `put`. Each node is removed at most once, so the cleanup work amortizes to O(1) per operation.
- **`put` at capacity**: first drops all expired nodes (if everything is expired, this clears the cache, per the spec); then, if still full, evicts exactly the LRU non-expired node and inserts the new entry.
- **`get` ordering**: a successful lookup unlinks and re-inserts the node at the head, marking it MRU; expired lookups remove the node and return `None` without re-ordering.
- **Time**: everything uses `time.monotonic()` (never wall clock), so the mocked `time.monotonic` drives the tests deterministically — no `time.sleep` anywhere.
- `get(key)` on an expired entry is treated as a miss, and `put` on an expired key overwrites it (refreshing both value and expiry) rather than triggering an eviction.

Run tests with: `pytest test_ttl_cache.py -v`