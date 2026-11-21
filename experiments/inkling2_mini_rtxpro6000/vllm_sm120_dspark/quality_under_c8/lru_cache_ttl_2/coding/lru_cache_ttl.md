Here's a complete implementation. I've split it into an implementation module and a test module.

## `ttl_cache.py`

```python
"""TTLCache: an LRU cache with per-item time-to-live expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """A node in the cache's doubly-linked list (sentinel nodes reuse this type)."""

    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with per-item TTL expiration.

    Internally uses a hash map (key -> node) plus a doubly-linked list ordered
    most-recently-used (front) to least-recently-used (back). get/put/delete
    run in O(1) average time; size() performs lazy cleanup and is O(n) where
    n is the number of tracked items.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: maximum number of items the cache may hold.
            default_ttl: default number of seconds until an item expires.
        """
        if capacity < 1:
            raise ValueError('capacity must be at least 1')
        if default_ttl < 0:
            raise ValueError('default_ttl must be non-negative')
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        # Sentinel-based list: MRU items sit just after the head, LRU items
        # just before the tail. Sentinels are never in the map.
        self._head = _Node('', None, 0.0)
        self._tail = _Node('', None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired, else None.

        Marks the key as most-recently-used. Expired entries are removed lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`, resetting its TTL and recency.

        A custom `ttl` overrides the default. If the cache is at capacity, the
        least-recently-used non-expired item is evicted; if every item is
        expired, expired items are cleared first (freeing space with no eviction).
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at = now + effective_ttl
        node = self._map.get(key)
        if node is not None:
            # Existing key: update value and expiry, refresh recency.
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return
        # New key: lazily drop trailing expired nodes from the LRU end.
        node = self._tail.prev
        while node is not self._head and node.expires_at <= now:
            prev_node = node.prev
            self._remove_node(node)
            node = prev_node
        # If still full, evict the LRU non-expired item. If the list is now
        # empty (all items were expired), nothing is evicted -- they were
        # already cleared in the loop above.
        if len(self._map) >= self._capacity and node is not self._head:
            self._remove_node(node)
        self._insert_front(key, value, expires_at)

    def delete(self, key: str) -> bool:
        """Remove `key` if present. Return True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items.

        Expired items are removed lazily as part of the scan.
        """
        now = time.monotonic()
        node = self._head.next
        while node is not self._tail:
            if node.expires_at <= now:
                next_node = node.next
                self._remove_node(node)
                node = next_node
            else:
                node = node.next
        return len(self._map)

    # -- internal helpers ----------------------------------------------------

    def _move_to_front(self, node: _Node) -> None:
        """Unlink `node` and re-insert it right after the head sentinel."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink `node` from the list and remove it from the map."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        del self._map[node.key]
        node.prev = None
        node.next = None

    def _insert_front(self, key: str, value: Any, expires_at: float) -> None:
        """Create a new node and insert it right after the head sentinel."""
        node = _Node(key, value, expires_at)
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        self._map[key] = node
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache (deterministic via mocked time.monotonic)."""

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic replacement for time.monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, dt: float) -> None:
        self.now += dt

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    """Patch time.monotonic with a controllable fake clock."""
    fake = FakeClock()
    with patch('time.monotonic') as mock_now:
        mock_now.side_effect = lambda: fake.now
        yield fake


def test_basic_get_put(clock):
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)

    assert cache.get('a') == 1
    assert cache.get('b') == 2
    assert cache.get('missing') is None

    cache.put('a', 100)  # update existing key
    assert cache.get('a') == 100
    assert cache.size() == 2


def test_capacity_eviction_lru_order(clock):
    cache = TTLCache(capacity=3, default_ttl=10.0