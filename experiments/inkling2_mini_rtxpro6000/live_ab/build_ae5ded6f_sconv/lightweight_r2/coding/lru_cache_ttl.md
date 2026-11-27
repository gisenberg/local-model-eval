# LRU Cache with TTL

Here's a complete implementation. The core idea: a doubly-linked list ordered by recency (most-recently-used at the head) plus a hash map for O(1) lookups, with lazy expiry sweeps.

**`ttl_cache.py`**

```python
"""TTLCache: a fixed-capacity LRU cache with time-based expiration.

Data structures:
  - a doubly-linked list keeps items in recency order (MRU at the front)
  - a hash map (key -> node) gives O(1) lookups

Expiry is enforced lazily: expired items are dropped when they are
encountered (on get / size / insert-at-capacity).

Complexity: get / delete / update are O(1). Insert removes each node at
most once (amortized O(1) removal work) but may scan the list to drop
expired items. size() scans the list to return the exact count of
non-expired items.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

__all__ = ["TTLCache"]


class _Node:
    """A single cached item held by the doubly-linked list."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev = None  # type: Optional[_Node]
        self.next = None  # type: Optional[_Node]


class TTLCache:
    """A fixed-capacity LRU cache whose items expire after a TTL.

    The head sentinel is the most-recently-used end; the tail sentinel is
    the least-recently-used end.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache.

        :param capacity: maximum number of items the cache may hold.
        :param default_ttl: seconds until an item expires by default.
        """
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be a positive number of seconds")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}

        # Sentinels so list operations never touch None.
        self._head = _Node("", None, float("inf"))
        self._tail = _Node("", None, float("inf"))
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for ``key`` if it exists and is not expired.

        Accessing a live key marks it most-recently-used. Expired keys are
        removed and reported as missing (None).
        """
        node = self._map.get(key)
        if node is None:
            return None
        if time.monotonic() >= node.expires_at:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update ``key``.

        ``ttl`` overrides the default TTL for this item. Inserting a new key
        when at capacity removes expired items first; if everything is
        expired it is all cleared, otherwise the least-recently-used
        non-expired item is evicted.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl

        # Existing key: refresh in place (no capacity change).
        existing = self._map.get(key)
        if existing is not None:
            existing.value = value
            existing.expires_at = now + effective_ttl
            self._move_to_front(existing)
            return

        if len(self._map) >= self._capacity:
            self._sweep_expired(now)
            if len(self._map) >= self._capacity:
                # Evict the least-recently-used non-expired item (tail end).
                self._remove_node(self._tail.prev)

        node = _Node(key, value, now + effective_ttl)
        self._map[key] = node
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def delete(self, key: str) -> bool:
        """Remove ``key`` from the cache. Return True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items.

        Expired items are lazily removed from the cache before counting.
        """
        self._sweep_expired(time.monotonic())
        return len(self._map)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_node(self, node: _Node) -> None:
        """Unlink ``node`` from the list and from the hash map."""
        node.prev.next = node.next
        node.next.prev = node.prev
        del self._map[node.key]

    def _move_to_front(self, node: _Node) -> None:
        """Mark ``node`` as most-recently-used (move it to the front)."""
        if node.prev is self._head:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _sweep_expired(self, now: float) -> None:
        """Remove every expired item from the cache."""
        node = self._tail.prev
        while node is not self._head:
            prev_node = node.prev
            if now >= node.expires_at:
                self._remove_node(node)
            node = prev_node
```

**`test_ttl_cache.py`**

```python
"""Tests for the TTLCache implementation.

Time is controlled deterministically by patching time.monotonic with a
manually advanced fake clock — no time.sleep() is used.
"""

from contextlib import contextmanager
from typing import Iterator

import pytest
from unittest.mock import patch

from ttl_cache import TTLCache


class _FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def fake_time() -> Iterator[_FakeClock]:
    """Patch time.monotonic with a manually advanced clock."""
    clock = _FakeClock()
    with patch("time.monotonic", clock):
        yield clock


def test_basic_get_put():
    with fake_time():
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        cache.put("b", 2)
        assert cache.get("a") == 1           # still valid, no side effects
        cache.put("a", 42)                   # update existing key
        assert cache.get("a") == 42
        assert cache.size() == 2


def test_capacity_eviction_lru_order():
    with fake_time():
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1           # a becomes MRU, b is now LRU
        cache.put("c", 3)                    # evicts b (least recently used)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_capacity_eviction_all_expired():
    with fake_time() as clock:
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        clock.advance(11.0)                  # both items expire
        cache.put("c", 3)                    # all expired -> clear, then insert
        assert cache.size() == 1
        assert cache.get("c") == 3
        assert cache.get("a") is None


def test_ttl_expiry():
    with fake_time() as clock:
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("a", 1)
        clock.advance(9.9)
        assert cache.get("a") == 1
        clock.advance(0.1)                   # reach the expiry boundary
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with fake_time() as clock:
        cache = TTLCache(capacity=5, default_ttl=100.0)
        cache.put("a", 1, ttl=5.0)           # short-lived item
        cache.put("b", 2)                    # default TTL (100s)
        clock.advance(6.0)
        assert cache.get("a") is None        # expired via custom TTL
        assert cache.get("b") == 2           # default TTL still valid


def test_delete():
    with fake_time():
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False    # already removed
        assert cache.get("a") is None
        assert cache.size() == 0


def test_size_with_mixed_expired_and_valid_items():
    with fake_time() as clock:
        cache = TTLCache(capacity=10, default_ttl=10.0)
        cache.put("short", 1, ttl=5.0)
        cache.put("default", 2)
        cache.put("long", 3, ttl=30.0)
        clock.advance(6.0)                   # "short" expires
        assert cache.size() == 2             # expired item lazily cleaned
        assert cache.get("default") == 2
        assert cache.get("long") == 3
        assert cache.get("short") is None
        assert cache.size() == 2
```

Run with:

```bash
pytest test_ttl_cache.py -v
```

A few notes:

- **LRU ordering** — `get` on a live key moves it to the front (MRU). `put` on a new key inserts at the front. Eviction removes the tail (LRU). Expired items are swept first, so they never count against capacity.
- **Expiry boundary** — an item expires when `now >= expires_at`, so `advance(10.0)` on a 10s TTL is treated as expired.
- **Complexity** — `get`/`delete`/update are strictly O(1). `size()` and the insert path perform a lazy sweep of the list, so they are O(n) in the worst case (though each node is removed at most once, so removal work is amortized O(1) per item). A truly O(1) `size()` would require an additional expiry-ordered structure on top of the mandated list+map, which I left out to keep the implementation simple and faithful to the spec.
- The tests cover the six required scenarios (basic, capacity/LRU order, TTL expiry, custom per-key TTL, delete, mixed size) plus one extra for the all-expired-at-capacity eviction path.