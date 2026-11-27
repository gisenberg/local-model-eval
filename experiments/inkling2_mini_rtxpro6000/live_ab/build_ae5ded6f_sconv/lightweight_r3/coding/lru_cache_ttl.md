Here's a complete implementation. It uses a hash map plus a sentinel-based doubly-linked list (head = LRU, tail = MRU), so `get`/`put`/`delete` are O(1) average.

**`ttl_cache.py`**

```python
"""LRU cache with time-based expiration.

Internally: a hash map (key -> node) + a doubly-linked list
(sentinel head = least-recently-used, sentinel tail = most-recently-used).
All time is tracked via time.monotonic().
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """A single cache entry in the doubly-linked list."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """An LRU cache where entries expire after a TTL (seconds).

    get/put/delete are O(1) average. size() performs lazy cleanup of
    expired entries and is O(n) in the number of stored entries.
    Not thread-safe.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` entries.

        Args:
            capacity: Maximum number of items allowed (>= 1).
            default_ttl: Seconds until an entry expires (> 0).
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        # Sentinels: head = LRU end, tail = MRU end.
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unlink(self, node: _Node) -> None:
        """Remove `node` from the linked list (map entry left untouched)."""
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]

    def _insert_mru(self, node: _Node) -> None:
        """Insert `node` as the new most-recently-used entry."""
        node.prev = self._tail.prev
        node.next = self._tail
        self._tail.prev.next = node  # type: ignore[union-attr]
        self._tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink `node` and remove it from the hash map."""
        self._unlink(node)
        del self._map[node.key]

    def _move_to_mru(self, node: _Node) -> None:
        """Move `node` to the MRU end of the list."""
        self._unlink(node)
        self._insert_mru(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and not expired, else None.

        On a hit, the key becomes most-recently-used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_mru(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key` with `value`.

        Args:
            key: Entry key.
            value: Value to store.
            ttl: Optional per-key TTL in seconds; overrides the default.

        If at capacity, expired entries are dropped first (this clears
        everything when all items are expired); if still at capacity, the
        least-recently-used non-expired item is evicted.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl <= 0:
            raise ValueError("ttl must be > 0")

        node = self._map.get(key)
        if node is not None:
            # Update existing entry: overwrite value/expiry, refresh recency.
            node.value = value
            node.expires_at = now + effective_ttl
            self._move_to_mru(node)
            return

        if len(self._map) >= self._capacity:
            # Drop expired entries from the LRU end first. If every stored
            # entry is expired, this clears the whole cache.
            while (
                self._head.next is not self._tail
                and self._head.next.expires_at <= now  # type: ignore[union-attr]
            ):
                self._remove_node(self._head.next)  # type: ignore[arg-type]
            # Still full: evict the least-recently-used non-expired item.
            if (
                len(self._map) >= self._capacity
                and self._head.next is not self._tail
            ):
                self._remove_node(self._head.next)  # type: ignore[arg-type]

        new_node = _Node(key, value, now + effective_ttl)
        self._map[key] = new_node
        self._insert_mru(new_node)

    def delete(self, key: str) -> bool:
        """Remove `key` from the cache.

        Returns True if the key existed, False otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items (expired ones are lazily removed)."""
        now = time.monotonic()
        node = self._head.next
        while node is not self._tail:
            next_node = node.next  # type: ignore[union-attr]
            if node.expires_at <= now:
                self._remove_node(node)  # type: ignore[arg-type]
            node = next_node  # type: ignore[assignment]
        return len(self._map)
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. Time is controlled deterministically via
unittest.mock.patch on time.monotonic (no time.sleep)."""

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


def test_basic_get_put() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=5, default_ttl=60.0)
        assert cache.get("a") is None      # missing key
        cache.put("a", 1)
        assert cache.get("a") == 1
        cache.put("a", 2)                   # update value
        assert cache.get("a") == 2
        assert cache.size() == 1


def test_capacity_eviction_lru_order() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1          # "a" -> MRU, "b" -> LRU
        cache.put("c", 3)                   # evicts least-recently-used: "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        clock.advance(11.0)
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size() == 0


def test_custom_per_key_ttl() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=5, default_ttl=100.0)
        cache.put("default_ttl", 1)
        cache.put("short_ttl", 2, ttl=5.0)
        clock.advance(10.0)
        assert cache.get("short_ttl") is None
        assert cache.get("default_ttl") == 1
        assert cache.size() == 1


def test_delete() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=5, default_ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.delete("a") is False  # already gone
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_and_valid() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=10, default_ttl=100.0)
        cache.put("short", 1, ttl=5.0)
        cache.put("long", 2, ttl=100.0)
        cache.put("mid", 3, ttl=10.0)
        clock.advance(6.0)                 # t=6: short expired, mid/long valid
        assert cache.size() == 2
        clock.advance(5.0)                 # t=11: mid expired, long valid
        assert cache.size() == 1
        assert cache.get("long") == 2


def test_put_clears_all_expired_items_when_all_expired() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        clock.advance(11.0)                # both expired
        cache.put("c", 3)                  # all expired -> cleared first
        assert cache.size() == 1
        assert cache.get("c") == 3
```

**Notes**

- `get`/`put`/`delete` are O(1) average; `size()` is O(n) in stored entries because it must verify every entry's expiry for an exact non-expired count (lazy cleanup), though it only does work proportional to expired entries removed plus the final valid scan. If you need strictly O(1) `size()`, you'd need a separate min-heap by expiry time — I kept the required list + map design instead.
- Update semantics: `put` on an existing key overwrites value, resets the TTL, and refreshes recency.
- Expired entries are dropped eagerly from the LRU end during `put` when needed, and lazily on `get`/`delete`/`size` otherwise.
- Run with: `pytest test_ttl_cache.py -v`