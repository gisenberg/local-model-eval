Here's a complete implementation. The cache uses a hash map (`key -> node`) plus a doubly-linked list with sentinel head/tail nodes (front = MRU, back = LRU), so all core operations are O(1).

**`ttl_cache.py`**

```python
"""A least-recently-used (LRU) cache with time-based (TTL) expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node representing one cache entry."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with per-item TTL expiration.

    Internal layout: hash map (key -> node) + doubly-linked list ordered
    most-recently-used (front) -> least-recently-used (back), with sentinel
    head/tail nodes so no boundary special-casing is required. Expired items
    are removed lazily on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache with a maximum capacity and a default TTL (seconds)."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head = _Node("", None, 0.0)  # sentinel, front (MRU)
        self._tail = _Node("", None, 0.0)  # sentinel, back (LRU)
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired.

        Accessing a live key marks it most-recently-used. Expired keys are
        removed and yield None.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`.

        `ttl` overrides the cache's default TTL for this item. If the cache is
        full, the least-recently-used non-expired item is evicted (if every
        item is expired, all of them are cleared first).
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        node = self._map.get(key)
        if node is not None:  # update existing key
            node.value = value
            node.expires_at = now + effective_ttl
            self._move_to_front(node)
            return
        if len(self._map) >= self.capacity:
            self._evict_for_insertion(now)
        node = _Node(key, value, now + effective_ttl)
        self._insert_front(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """Remove `key` if present. Returns True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items (lazily removes expired ones)."""
        now = time.monotonic()
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            if node.expires_at <= now:
                self._unlink(node)
                self._map.pop(node.key, None)
            node = nxt
        return len(self._map)

    # ---- internal helpers ----

    def _insert_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        node.next.prev = node  # type: ignore[union-attr]
        self._head.next = node

    def _unlink(self, node: _Node) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _remove(self, node: _Node) -> None:
        self._unlink(node)
        self._map.pop(node.key, None)

    def _move_to_front(self, node: _Node) -> None:
        if node.prev is self._head:
            return  # already most-recently-used
        self._unlink(node)
        self._insert_front(node)

    def _evict_for_insertion(self, now: float) -> None:
        # Drop expired nodes from the LRU (tail) side...
        node = self._tail.prev
        while node is not self._head:
            if node.expires_at <= now:
                prev = node.prev
                self._remove(node)
                node = prev
            else:
                break
        # ...then evict the LRU non-expired node (if any). If everything was
        # expired, the map is now empty and no eviction is needed.
        if node is not self._head:
            self._remove(node)
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache — time is mocked via unittest.mock.patch on time.monotonic."""

from unittest.mock import patch

from ttl_cache import TTLCache


class _Clock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def test_basic_get_put():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        cache.put("a", 100)  # overwrite existing key
        assert cache.get("a") == 100


def test_capacity_eviction_lru_order():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1   # 'a' is now MRU -> 'b' is the LRU
        cache.put("c", 3)             # evicts 'b'
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)              # expires at t=10
        clock.advance(5.0)
        cache.put("b", 2)              # expires at t=15
        clock.advance(6.0)             # now = 11
        assert cache.get("a") is None  # expired
        assert cache.get("b") == 2     # still valid


def test_custom_per_key_ttl():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("a", 1, ttl=5.0)     # expires at t=5
        cache.put("b", 2)              # default ttl -> expires at t=100
        clock.advance(6.0)             # now = 6
        assert cache.get("a") is None
        assert cache.get("b") == 2
        # custom ttl also applies to updates
        cache.put("b", 20, ttl=2.0)    # expires at t=8
        clock.advance(3.0)             # now = 9
        assert cache.get("b") is None


def test_delete():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.delete("a") is False   # already gone
        assert cache.delete("missing") is False
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_and_valid():
    clock = _Clock(0.0)
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("a", 1)              # expires at t=10
        cache.put("b", 2)              # expires at t=10
        clock.advance(4.0)
        cache.put("c", 3)              # expires at t=14
        clock.advance(7.0)             # now = 11
        assert cache.size() == 1       # a, b expired; c valid
        assert cache.get("a") is None
        assert cache.get("c") == 3
        assert cache.size() == 1       # expired items already removed
```

**Design notes**

- The doubly-linked list uses sentinel head/tail nodes, so insertion at the front and unlinking are branch-free O(1) operations; the hash map gives O(1) lookup. `get`/`put`/`delete` are strict O(1). `size()` and insertion-time eviction scan expired nodes, but each node is removed only once, so the amortized cost per operation stays O(1).
- Eviction scans from the LRU (tail) end: it removes expired nodes first (which is what handles the "all items expired" case — the whole list is cleared) and then evicts the first non-expired node found.
- A custom `ttl` overrides the default for that key on both insert and update; updating an existing key refreshes both its value and its recency position.
- `delete` treats presence in the cache as "existed", regardless of expiry — an expired entry is simply removed.
- Tests never use `time.sleep`; each test installs a `_Clock` as the replacement for `time.monotonic` via `patch`, advancing it manually.

Run with:

```
python -m pytest test_ttl_cache.py -v
```