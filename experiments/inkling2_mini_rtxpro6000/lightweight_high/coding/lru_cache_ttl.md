Here's a complete implementation with the linked-list + hash-map design (no `OrderedDict`), plus the tests.

## `ttl_cache.py`

```python
"""LRU cache with per-entry time-based expiration."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node; head is most-recently-used, tail is least-recently-used."""
    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with per-entry TTL.

    Internals: doubly-linked list (head = MRU, tail = LRU) + hash map
    from key -> node. Expired entries are removed lazily (on access).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache with max `capacity` entries and a default TTL in seconds."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most-recently-used
        self._tail: Optional[_Node] = None  # least-recently-used

    # ---------- internal helpers ----------

    def _unlink(self, node: _Node) -> None:
        """Remove node from the list, leaving the hash map entry intact."""
        prev, nxt = node.prev, node.next
        if prev is not None:
            prev.next = nxt
        else:
            self._head = nxt
        if nxt is not None:
            nxt.prev = prev
        else:
            self._tail = prev
        node.prev = node.next = None

    def _link_front(self, node: _Node) -> None:
        """Insert node at the head (MRU position), leaving the hash map intact."""
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node and drop it from the hash map."""
        self._unlink(node)
        self._map.pop(node.key, None)

    def _purge_expired(self, now: float) -> None:
        """Remove every expired entry from the list and map."""
        node = self._tail
        while node is not None:
            prev = node.prev
            if node.expires_at <= now:
                self._remove_node(node)
            node = prev

    # ---------- public API ----------

    def get(self, key: str) -> Optional[Any]:
        """Return the value if present and unexpired (and mark it MRU), else None."""
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            return None
        self._unlink(node)
        self._link_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert/update a key. `ttl` overrides default_ttl. Evicts LRU non-expired
        entry when full; expired entries are cleared first."""
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = now + effective_ttl

        node = self._map.get(key)
        if node is not None:  # upsert: refresh value, expiry, and recency
            node.value = value
            node.expires_at = expires_at
            self._unlink(node)
            self._link_front(node)
            return

        if len(self._map) >= self.capacity:
            self._purge_expired(now)          # clear all expired items first
            if len(self._map) >= self.capacity:
                self._remove_node(self._tail)  # evict LRU non-expired item

        new_node = _Node(key, value, expires_at)
        self._link_front(new_node)
        self._map[key] = new_node

    def delete(self, key: str) -> bool:
        """Remove key if present. Return True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items (expired items are lazily removed)."""
        self._purge_expired(time.monotonic())
        return len(self._map)
```

## `test_ttl_cache.py`

```python
from unittest.mock import patch

from ttl_cache import TTLCache


def test_basic_get_put():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=3, default_ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None


def test_capacity_eviction_lru_order():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        mock_time.return_value = 1.0
        assert cache.get("a") == 1  # a -> MRU, b -> LRU
        cache.put("c", 3)            # evicts b (least-recently-used)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        mock_time.return_value = 5.0
        assert cache.get("a") == 1
        mock_time.return_value = 11.0
        assert cache.get("a") is None  # expired
        assert cache.size() == 0       # lazily removed


def test_custom_per_key_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("short", "s", ttl=5.0)
        cache.put("long", "l")  # falls back to default_ttl
        mock_time.return_value = 6.0
        assert cache.get("short") is None
        assert cache.get("long") == "l"


def test_delete():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.delete("a") is False  # already gone
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_and_valid():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=10, default_ttl=100.0)
        cache.put("short", 1, ttl=5.0)
        cache.put("mid", 2, ttl=50.0)
        cache.put("long", 3)  # default_ttl=100.0
        mock_time.return_value = 10.0
        assert cache.size() == 2  # short expired
        mock_time.return_value = 60.0
        assert cache.size() == 1  # short and mid expired
        assert cache.get("long") == 3
```

Run with:

```bash
pytest test_ttl_cache.py -v
```

**Notes on design:**

- Head = most-recently-used, tail = least-recently-used; `get`/`put`/`delete` are O(1) average via the hash map + constant-time list surgery.
- The lazy sweep in `size()` (and the capacity-triggered cleanup in `put`) is O(n) worst-case — that's inherent to "expired items removed on access" since items can expire without any access occurring. All other operations are O(1) average.
- Expiry uses `expires_at <= now`, so an item expires exactly at its TTL boundary.
- Tests patch `time.monotonic` directly (the cache calls it via the `time` module, so the patch applies) and control time by reassigning `mock_time.return_value` — no `time.sleep`, fully deterministic.