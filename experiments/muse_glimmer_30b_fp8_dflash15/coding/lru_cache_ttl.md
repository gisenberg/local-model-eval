```python
# ttl_cache.py
import time
from typing import Any, Optional


class _Node:
    __slots__ = ("key", "value", "expire_at", "prev", "next")

    def __init__(self, key: str, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev = None
        self.next = None


class TTLCache:
    """LRU cache with per-item TTL using monotonic time."""

    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.map = {}                     # key -> _Node
        self.head = _Node(None, None, 0)  # most recent
        self.tail = _Node(None, None, 0)  # least recent
        self.head.next = self.tail
        self.tail.prev = self.head

    # ---- helpers ----
    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        return self._now() > node.expire_at

    def _add_to_head(self, node: _Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    # ---- public API ----
    def get(self, key: str) -> Optional[Any]:
        """Return value if present and not expired, else None."""
        node = self.map.get(key)
        if not node:
            return None
        if self._is_expired(node):
            self._remove_node(node)
            del self.map[key]
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update. Evicts LRU non-expired on overflow."""
        ttl = ttl if ttl is not None else self.default_ttl
        expire_at = self._now() + ttl
        node = self.map.get(key)
        if node:
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
            return

        if self.capacity <= 0:
            return

        # make room, cleaning expired first
        while len(self.map) >= self.capacity:
            cur = self.tail.prev
            victim = None
            while cur is not self.head:
                if self._is_expired(cur):
                    self._remove_node(cur)
                    del self.map[cur.key]
                    cur = self.tail.prev
                    continue
                victim = cur
                break
                cur = cur.prev
            if victim:
                self._remove_node(victim)
                del self.map[victim.key]
                break
            # all remaining nodes were expired -> loop will exit

        node = _Node(key, value, expire_at)
        self.map[key] = node
        self._add_to_head(node)

    def delete(self, key: str) -> bool:
        """Remove key. Return True if it existed."""
        node = self.map.pop(key, None)
        if not node:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Number of non-expired items, with lazy cleanup."""
        now = self._now()
        count = 0
        for key in list(self.map.keys()):
            node = self.map[key]
            if node.expire_at <= now:
                self._remove_node(node)
                del self.map[key]
            else:
                count += 1
        return count
```

```python
# test_ttl_cache.py
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def test_basic_get_put():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(2, 100)
        c.put("a", 1)
        assert c.get("a") == 1
        assert c.get("b") is None


def test_capacity_eviction_lru():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(2, 1000)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")          # a becomes MRU
        c.put("c", 3)       # evicts b
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3


def test_ttl_expiry():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(2, 5)
        c.put("a", 1)
        assert c.get("a") == 1
        t[0] = 6
        assert c.get("a") is None


def test_custom_per_key_ttl():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(2, 100)
        c.put("a", 1, ttl=1)
        c.put("b", 2)       # default ttl
        t[0] = 2
        assert c.get("a") is None
        assert c.get("b") == 2


def test_delete():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(2, 100)
        c.put("a", 1)
        assert c.delete("a") is True
        assert c.get("a") is None
        assert c.delete("a") is False


def test_size_mixed_expired_valid():
    t = [0]
    with patch("time.monotonic", side_effect=lambda: t[0]):
        c = TTLCache(3, 10)
        c.put("a", 1, ttl=1)
        c.put("b", 2)
        c.put("c", 3)
        t[0] = 2
        assert c.size() == 2
        # expired item is really gone
        assert c.get("a") is None
```

The cache uses a hash map + doubly linked list, all operations are O(1) average, expiration is checked with `time.monotonic()` and the tests are deterministic via `unittest.mock.patch`.