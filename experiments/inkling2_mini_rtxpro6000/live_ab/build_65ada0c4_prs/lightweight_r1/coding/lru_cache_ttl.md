Here's a complete implementation using a hash map + doubly-linked list (no `OrderedDict`), with `time.monotonic()` for time tracking and 6 deterministic pytest tests.

## `ttl_cache.py`

```python
"""A least-recently-used (LRU) cache with per-item time-to-live (TTL) expiration."""

from __future__ import annotations

import time
from typing import Any, Optional


class _Node:
    """A doubly-linked-list node holding one cache entry.

    The list is ordered from most-recently-used (head) to
    least-recently-used (tail).
    """

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional["_Node"] = None  # toward head (MRU)
        self.next: Optional["_Node"] = None  # toward tail (LRU)


class TTLCache:
    """An LRU cache with per-entry TTL expiration.

    Internally uses a hash map (key -> node) plus a doubly-linked list
    ordered by recency (head = most recently used, tail = least recently used).

    get / put / delete are O(1) average (amortized). size() is O(n)
    worst-case: lazy expiry means expired nodes may linger, so size() must
    sweep the map to report an exact non-expired count.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialise the cache.

        Args:
            capacity: maximum number of items retained (>= 1).
            default_ttl: seconds until an entry expires by default (> 0).

        Raises:
            ValueError: if capacity < 1 or default_ttl <= 0.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity: int = capacity
        self.default_ttl: float = float(default_ttl)
        self._map: dict[str, _Node] = {}
        self._head: Optional[_Node] = None
        self._tail: Optional[_Node] = None

    # ---------------- internal helpers ----------------

    def _unlink(self, node: _Node) -> None:
        """Remove `node` from the linked list (does not touch the hash map)."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = None
        node.next = None

    def _link_front(self, node: _Node) -> None:
        """Insert `node` at the head (most-recently-used position)."""
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink `node` and drop it from the hash map."""
        self._unlink(node)
        del self._map[node.key]

    # ---------------- public API ----------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key`, or None if missing/expired.

        A successful access marks the key as most-recently-used.
        Expired entries are removed (lazy cleanup).
        """
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            return None
        if node is not self._head:
            self._unlink(node)
            self._link_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`.

        `ttl` overrides the default TTL for this entry. When at capacity,
        expired tail items are cleaned up first; the least-recently-used
        non-expired item is then evicted. If all items are expired they are
        cleared first. The new/updated key becomes most-recently-used.
        """
        now = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl
        if effective_ttl <= 0:
            raise ValueError("ttl must be > 0")
        expires_at = now + effective_ttl

        # Handle update of an existing key (even if expired).
        old = self._map.get(key)
        if old is not None:
            self._remove_node(old)

        # Make room for the new item.
        if len(self._map) >= self.capacity:
            # Clean up expired tail items first (they are worthless).
            node = self._tail
            while node is not None and node.expires_at <= now:
                prev = node.prev
                self._remove_node(node)
                node = prev
            # Still full: evict the LRU non-expired item (tail is valid now).
            if len(self._map) >= self.capacity:
                node = self._tail
                if node is not None:
                    self._remove_node(node)

        node = _Node(key, value, expires_at)
        self._map[key] = node
        self._link_front(node)

    def delete(self, key: str) -> bool:
        """Remove `key`. Returns True if it existed (even if expired)."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items.

        Expired items are removed (lazy cleanup).
        """
        now = time.monotonic()
        expired = [node for node in self._map.values() if node.expires_at <= now]
        for node in expired:
            self._remove_node(node)
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache. Deterministic time via unittest.mock.patch on
time.monotonic — no time.sleep, no wall-clock flakiness."""

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


class FakeClock:
    """Controllable replacement for time.monotonic()."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_basic_get_put() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=3, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        # update overwrites the value
        cache.put("a", 99)
        assert cache.get("a") == 99


def test_capacity_eviction_lru_order() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")        # "a" becomes MRU; "b" is now the LRU item
        cache.put("c", 3)     # evicts "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=3, default_ttl=10)
        cache.put("a", 1)
        assert cache.get("a") == 1
        clock.advance(11)
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_per_key_ttl() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=5, default_ttl=100)
        cache.put("short", 1, ttl=5)
        cache.put("long", 2, ttl=50)
        cache.put("default", 3)          # uses default_ttl=100
        clock.advance(6)
        assert cache.get("short") is None
        assert cache.get("long") == 2
        assert cache.get("default") == 3


def test_delete() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=3, default_ttl=60)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False  # already gone


def test_size_with_mixed_expired_and_valid_items() -> None:
    clock = FakeClock()
    with patch("time.monotonic", new=clock):
        cache = TTLCache(capacity=10, default_ttl=100)
        cache.put("a", 1, ttl=5)
        cache.put("b", 2, ttl=50)
        cache.put("c", 3, ttl=100)
        clock.advance(10)                # only "a" has expired
        assert cache.size() == 2
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2
```

Run with `pytest -v`.

## Design notes

- **Data structures**: `_map: dict[str, _Node]` for O(1) lookup + a doubly-linked list ordered head→tail (MRU→LRU) for O(1) re-linking on access and O(1) eviction of the tail.
- **Expiry**: stored per-node as an absolute `expires_at` timestamp (`time.monotonic()` + ttl). Expired nodes are removed lazily on access (`get`, `put`, `delete`, `size`) — there are no background threads or timers.
- **Capacity eviction in `put`**: if full, expired tail items are cleaned up first; if still full, the LRU non-expired item (the tail) is evicted. If *everything* is expired, the tail-sweep removes them all before the new insert.
- **Honesty about complexity**: `get`/`put`/`delete` are O(1) average (amortized — each node is removed at most once). `size()` is O(n) worst-case because lazy expiry means expired nodes may linger anywhere in the map, so an exact count requires a sweep. This is the standard trade-off of lazy expiry; a strictly O(1) size would require an expiry-index (heap), which would go beyond the required list+hash-map structure. If you want, I can add an optional min-heap of `(expires_at, key)` alongside to make size O(log n) amortized — just say the word.