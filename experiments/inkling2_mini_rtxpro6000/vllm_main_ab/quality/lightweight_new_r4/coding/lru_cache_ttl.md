## Implementation

**`ttl_cache.py`**

```python
"""Time-based LRU cache implemented with a doubly-linked list + hash map."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """A single doubly-linked-list entry (one cached key)."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """An LRU cache where entries expire after a TTL.

    The list is kept most-recently-used first (head) to least-recently-used
    last (tail). Lookups, inserts and deletes are O(1) average; expiry is
    handled lazily (expired entries are dropped when encountered).

    NOTE: size() walks the list once to drop expired entries, so a single
    call is O(n) worst case, but each entry is removed at most once, making
    it amortized O(1) across a sequence of operations.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache with a max entry count and a default TTL (seconds)."""
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most recently used
        self._tail: Optional[_Node] = None  # least recently used

    def get(self, key: str) -> Optional[Any]:
        """Return the value for ``key`` if present and unexpired, else None.

        A hit promotes the key to most-recently-used. An expired entry is
        dropped and reported as missing.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if now >= node.expires_at:
            self._map.pop(key, None)
            self._unlink(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update ``key``, using ``ttl`` (seconds) or the default.

        If at capacity, expired entries at the tail are dropped first (so an
        all-expired cache is cleared), then the least-recently-used non-expired
        entry is evicted.
        """
        now = time.monotonic()
        expires_at = now + (self._default_ttl if ttl is None else ttl)
        node = self._map.get(key)
        if node is not None:  # update existing entry
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return
        if len(self._map) >= self._capacity:
            self._drop_expired_tail(now)
            if len(self._map) >= self._capacity:
                self._evict_tail()
        node = _Node(key, value, expires_at)
        self._map[key] = node
        self._add_to_front(node)

    def delete(self, key: str) -> bool:
        """Remove ``key`` if present; return True if it existed (even if expired)."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._unlink(node)
        return True

    def size(self) -> int:
        """Return the count of non-expired entries, dropping expired ones."""
        now = time.monotonic()
        node = self._tail
        while node is not None:
            prev = node.prev
            if now >= node.expires_at:
                self._map.pop(node.key, None)
                self._unlink(node)
            node = prev
        return len(self._map)

    # ---- internal helpers (all O(1)) ----

    def _add_to_front(self, node: _Node) -> None:
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _unlink(self, node: _Node) -> None:
        prev, next_ = node.prev, node.next
        if prev is not None:
            prev.next = next_
        else:
            self._head = next_
        if next_ is not None:
            next_.prev = prev
        else:
            self._tail = prev

    def _move_to_front(self, node: _Node) -> None:
        if node is self._head:
            return
        self._unlink(node)
        self._add_to_front(node)

    def _evict_tail(self) -> None:
        """Evict the least-recently-used entry (must be non-expired)."""
        node = self._tail
        if node is None:
            return
        self._map.pop(node.key, None)
        self._unlink(node)

    def _drop_expired_tail(self, now: float) -> None:
        """Drop expired entries from the tail (least-recently-used first)."""
        while self._tail is not None and now >= self._tail.expires_at:
            node = self._tail
            self._map.pop(node.key, None)
            self._unlink(node)
```

## Tests

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. Time is controlled by mocking time.monotonic."""

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


class _FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    clk = _FakeClock()
    with patch("time.monotonic", clk):
        yield clk


def test_basic_get_put(clock):
    cache = TTLCache(3, 60.0)
    assert cache.get("missing") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    cache.put("a", 2)  # update existing key
    assert cache.get("a") == 2
    assert cache.size() == 1


def test_capacity_eviction_lru_order(clock):
    cache = TTLCache(2, 60.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")       # "a" becomes most-recently-used
    cache.put("c", 3)    # capacity reached -> evict LRU non-expired ("b")
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.size() == 2


def test_ttl_expiry(clock):
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    clock.advance(10.0)
    assert cache.get("a") is None
    assert cache.size() == 0
    cache.put("b", 2)  # expired entry cleared before insert
    assert cache.size() == 1
    assert cache.get("b") == 2


def test_custom_ttl(clock):
    cache = TTLCache(3, 100.0)
    cache.put("short", 1, ttl=5.0)
    cache.put("long", 2)
    clock.advance(5.0)
    assert cache.get("short") is None
    assert cache.get("long") == 2


def test_delete(clock):
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False
    assert cache.delete("nonexistent") is False
    assert cache.size() == 1


def test_size_mixed_expired_and_valid(clock):
    cache = TTLCache(5, 100.0)
    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2)                 # default ttl (100s)
    cache.put("c", 3, ttl=5.0)
    clock.advance(6.0)                # "a" and "c" expired, "b" still valid
    assert cache.size() == 1
    assert cache.get("b") == 2
    assert cache.get("a") is None
    assert cache.get("c") is None
```

Run with:

```bash
python -m pytest test_ttl_cache.py -v
```

## Design notes

- **Structure**: hash map `key -> _Node` gives O(1) lookup; a doubly-linked list (head = MRU, tail = LRU) gives O(1) removal/insertion at the ends — no `OrderedDict` used.
- **Lazy expiry**: nothing is evicted by a background timer. Expired entries are dropped whenever they're encountered (`get`, `put`, `size`). Note that `time.monotonic()` is read at call time so patching `time.monotonic` controls the clock everywhere.
- **Eviction order in `put`**: when at capacity, tail entries are dropped if expired (so an all-expired cache is cleared first), then the tail (the LRU non-expired entry) is evicted, then the new entry is inserted at the head.
- **Update semantics**: `put` on an existing key refreshes the value *and* expiry and promotes the key to MRU (it's an access).
- **Complexity**: `get`/`put`/`delete` are O(1) average (tail-expired cleanup is amortized — each node is dropped at most once). `size()` does a single lazy-cleanup pass, so a single call is O(n) worst case but amortized O(1) over a sequence of operations.