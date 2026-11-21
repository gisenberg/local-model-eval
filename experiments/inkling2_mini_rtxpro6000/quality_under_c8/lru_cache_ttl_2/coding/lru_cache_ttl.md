# LRU Cache with TTL Expiration

## `ttl_cache.py`

```python
"""LRU cache with per-item TTL expiration.

Design: a doubly-linked list (head = most recently used, tail = least recently
used) plus a hash map from key -> node. Expiration is lazy — expired items are
removed when they are touched, never by a background thread.
"""

import time
from typing import Any, Dict, Optional


class _Node:
    """Internal doubly-linked-list node."""

    __slots__ = ("key", "value", "expire_at", "prev", "next")

    def __init__(self, key: Optional[str], value: Any, expire_at: float) -> None:
        self.key = key
        self.value = value
        # Item is expired when time.monotonic() >= expire_at.
        self.expire_at = expire_at
        self.prev: Optional["_Node"] = None  # toward head (more recently used)
        self.next: Optional["_Node"] = None  # toward tail (least recently used)


class TTLCache:
    """An LRU cache whose items expire after a time-to-live (TTL).

    Args:
        capacity: maximum number of items the cache may hold.
        default_ttl: seconds after which an item expires (used when a caller
                     does not supply a per-key ``ttl``).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}

        # Sentinels so list operations never special-case an empty list.
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    # -- internal list helpers ----------------------------------------------

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the list."""
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _insert_front(self, node: _Node) -> None:
        """Insert `node` immediately after the head (as the MRU item)."""
        assert self._head.next is not None
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Promote `node` to most-recently-used."""
        if node is self._head.next:
            return  # already at the front
        self._unlink(node)
        self._insert_front(node)

    def _remove_node(self, node: _Node) -> None:
        """Unlink `node` and drop it from the hash map."""
        self._unlink(node)
        assert node.key is not None
        del self._map[node.key]

    # -- public API ----------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired, else None.

        A successful access promotes the key to most-recently-used. Expired
        entries are removed lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if now >= node.expire_at:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`.

        The item expires `ttl` seconds from now (or `default_ttl` if `ttl` is
        not given). If the cache is at capacity, the least-recently-used
        non-expired item is evicted; if all items are expired, they are cleared
        first.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl
        expire_at = now + effective_ttl

        existing = self._map.get(key)
        if existing is not None:
            existing.value = value
            existing.expire_at = expire_at
            self._move_to_front(existing)
            return

        if len(self._map) >= self._capacity:
            # 1. Drop expired items from the LRU end (lazy cleanup).
            node = self._tail.prev
            while node is not self._head and now >= node.expire_at:
                prev = node.prev
                self._remove_node(node)
                node = prev
            # 2. Still at capacity: evict the least-recently-used non-expired item.
            if len(self._map) >= self._capacity:
                node = self._tail.prev
                if node is not self._head:
                    self._remove_node(node)

        new_node = _Node(key, value, expire_at)
        self._map[key] = new_node
        self._insert_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove `key` if it is present. Return True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items.

        Expired items are removed lazily.
        """
        now = time.monotonic()
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            if now >= node.expire_at:
                self._unlink(node)
                assert node.key is not None
                del self._map[node.key]
            node = nxt
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache — time is mocked for determinism (no time.sleep)."""

from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic replacement for time.monotonic()."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_basic_get_put():
    clock = FakeClock()
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        cache.put("a", 10)  # update existing key
        assert cache.get("a") == 10
        assert cache.size() == 2


def test_capacity_eviction_lru_order():
    clock = FakeClock()
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") == 1  # 'a' becomes MRU; LRU order: c, b, a
        cache.put("d", 4)  # evicts LRU non-expired item ('b')
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert cache.size() == 3


def test_ttl_expiry():
    clock = FakeClock()
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        clock.advance(6.0)  # both items expire
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    clock = FakeClock()
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("short", 1, ttl=5.0)  # custom per-key TTL overrides default
        cache.put("long", 2)            # uses default_ttl
        clock.advance(6.0)
        assert cache.get("short") is None
        assert cache.get("long") == 2
        assert cache.size() == 1


def test_delete():
    clock = FakeClock()
    with patch("time.monotonic", clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete