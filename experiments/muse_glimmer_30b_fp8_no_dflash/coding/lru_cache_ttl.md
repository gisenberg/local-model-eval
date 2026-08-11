```python
# ttl_cache.py
import time
from typing import Any, Optional


class _Node:
    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self):
        self.key = None
        self.value = None
        self.expiry = 0.0
        self.prev = None
        self.next = None


class TTLCache:
    """LRU cache with per-item TTL using a doubly linked list + hash map."""

    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, _Node] = {}
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    # ----- helpers -----
    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        return self._now() >= node.expiry

    def _add_to_head(self, node: _Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_expired_node(self, node: _Node) -> None:
        self._remove_node(node)
        self.cache.pop(node.key, None)

    # ----- public API -----
    def get(self, key: str) -> Optional[Any]:
        """Return value if present and not expired, else None. Moves to MRU."""
        node = self.cache.get(key)
        if not node:
            return None
        if self._is_expired(node):
            self._remove_expired_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update. Evicts LRU non-expired on capacity."""
        now = self._now()
        ttl_eff = ttl if ttl is not None else self.default_ttl
        expiry = now + ttl_eff

        node = self.cache.get(key)
        if node:
            if self._is_expired(node):
                self._remove_expired_node(node)
                node = None
            else:
                node.value = value
                node.expiry = expiry
                self._move_to_head(node)
                return

        # make room
        while len(self.cache) >= self.capacity:
            lru = self.tail.prev
            while lru is not self.head and self._is_expired(lru):
                prev = lru.prev
                self._remove_expired_node(lru)
                lru = prev
            if lru is self.head:
                self.cache.clear()
                self.head.next = self.tail
                self.tail.prev = self.head
                break
            self._remove_expired_node(lru)

        new_node = _Node()
        new_node.key = key
        new_node.value = value
        new_node.expiry = expiry
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """Remove key, return True if it existed."""
        node = self.cache.pop(key, None)
        if not node:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Number of non-expired items. Lazy cleanup."""
        now = self._now()
        cur = self.head.next
        while cur is not self.tail:
            nxt = cur.next
            if now >= cur.expiry:
                self._remove_node(cur)
                del self.cache[cur.key]
            cur = nxt
        return len(self.cache)
```

```python
# test_ttl_cache.py
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def make_cache(capacity=2, ttl=10):
    cur = [0.0]
    m = patch('time.monotonic', side_effect=lambda: cur[0])
    m.start()
    cache = TTLCache(capacity, ttl)
    return cache, cur, m


def test_basic_get_put():
    cache, cur, m = make_cache()
    try:
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.get('missing') is None
    finally:
        m.stop()


def test_capacity_eviction_lru():
    cache, cur, m = make_cache(capacity=2, ttl=100)
    try:
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')          # a becomes MRU
        cache.put('c', 3)       # evicts b
        assert cache.get('b') is None
        assert cache.get('a') == 1
        assert cache.get('c') == 3
    finally:
        m.stop()


def test_ttl_expiry():
    cache, cur, m = make_cache(capacity=2, ttl=10)
    try:
        cache.put('a', 1)
        assert cache.get('a') == 1
        cur[0] = 5
        assert cache.get('a') == 1
        cur[0] = 11
        assert cache.get('a') is None
    finally:
        m.stop()


def test_custom_per_key_ttl():
    cache, cur, m = make_cache(capacity=2, ttl=100)
    try:
        cache.put('a', 1, ttl=5)
        cache.put('b', 2)
        cur[0] = 6
        assert cache.get('a') is None
        assert cache.get('b') == 2
    finally:
        m.stop()


def test_delete():
    cache, cur, m = make_cache()
    try:
        cache.put('a', 1)
        assert cache.delete('a') is True
        assert cache.get('a') is None
        assert cache.delete('a') is False
    finally:
        m.stop()


def test_size_mixed_expired_valid():
    cache, cur, m = make_cache(capacity=10, ttl=100)
    try:
        cache.put('a', 1, ttl=5)
        cache.put('b', 2, ttl=100)
        assert cache.size() == 2
        cur[0] = 6
        assert cache.size() == 1
        assert cache.get('a') is None
        assert cache.get('b') == 2
    finally:
        m.stop()
```

The implementation uses a sentinel doubly-linked list for O(1) MRU moves and eviction, `time.monotonic()` for TTL, and lazy removal of expired entries on access.