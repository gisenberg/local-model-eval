Here's a complete implementation. One honest note up front: get/put-update/delete are strictly O(1); `size()` and the at-capacity branch of `put()` are O(n) worst case, because an *exact* count of non-expired items can't be known without inspecting each node's `expires_at` (lazy expiry means expired nodes sit there undetected until someone looks). The linked-list + hash-map structure gives O(1) average for everything else, and each node is only ever purged once.

## `ttl_cache.py`

```python
"""A least-recently-used (LRU) cache with time-based expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Internal doubly-linked-list node."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with per-key TTL expiration.

    Nodes are ordered in a doubly-linked list from least-recently-used
    (head) to most-recently-used (tail), with a hash map for O(1) lookups.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most ``capacity`` items; items expire ``default_ttl`` seconds after insertion/update."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # least recently used
        self._tail: Optional[_Node] = None  # most recently used

    def get(self, key: str) -> Optional[Any]:
        """Return the value for ``key`` if present and unexpired, else None.

        A hit promotes the key to most-recently-used.
        """
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_mru(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update ``key``; ``ttl`` (seconds) overrides the default TTL.

        On insert into a full cache: expired items are cleared first, then the
        least-recently-used non-expired item is evicted.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at = now + effective_ttl

        existing = self._map.get(key)
        if existing is not None:
            if existing.expires_at <= now:
                self._remove_node(existing)  # stale entry -> fresh insert below
            else:
                existing.value = value
                existing.expires_at = expires_at
                self._move_to_mru(existing)
                return

        if len(self._map) >= self._capacity:
            self._purge_expired(now)
            while len(self._map) >= self._capacity and self._head is not None:
                self._remove_node(self._head)  # LRU non-expired item

        node = _Node(key, value, expires_at)
        self._insert_mru(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """Remove ``key`` and return True if it existed (even if already expired)."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items (expired items are lazily removed)."""
        now = time.monotonic()
        self._purge_expired(now)
        return len(self._map)

    # -- internal helpers ------------------------------------------------

    def _insert_mru(self, node: _Node) -> None:
        """Append ``node`` at the most-recently-used (tail) end."""
        node.prev = self._tail
        node.next = None
        if self._tail is not None:
            self._tail.next = node
        else:
            self._head = node
        self._tail = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink ``node`` from the list and remove it from the map."""
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
        self._map.pop(node.key, None)

    def _move_to_mru(self, node: _Node) -> None:
        """Promote ``node`` to the most-recently-used (tail) end."""
        if node is self._tail:
            return
        prev = node.prev
        nxt = node.next  # always not None: node is not the tail
        if prev is not None:
            prev.next = nxt
        else:
            self._head = nxt
        nxt.prev = prev
        node.prev = self._tail
        node.next = None
        self._tail.next = node
        self._tail = node

    def _purge_expired(self, now: float) -> None:
        """Remove every node whose expiry has passed."""
        node = self._head
        while node is not None:
            nxt = node.next
            if node.expires_at <= now:
                self._remove_node(node)
            node = nxt
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache. Time is mocked via unittest.mock.patch on time.monotonic."""

from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_basic_get_put():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None

        cache.put("a", 2)  # update existing key
        assert cache.get("a") == 2

        clock.advance(5)
        assert cache.get("a") == 2  # still valid
        assert cache.size() == 1


def test_capacity_eviction_lru_order():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=2, default_ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1  # 'a' becomes MRU; 'b' is now LRU
        cache.put("c", 3)           # evicts least-recently-used ('b')
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        clock.advance(9.9)
        assert cache.get("a") == 1
        clock.advance(0.2)  # now = 10.1 > ttl
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=60.0)
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=50.0)
        cache.put("c", 3)  # default ttl
        clock.advance(6)
        assert cache.get("a") is None  # expired after 5s
        assert cache.get("b") == 2     # still valid
        assert cache.get("c") == 3     # still valid
        assert cache.size() == 2


def test_delete():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False
        assert cache.get("a") is None
        assert cache.size() == 0

        # deleting an expired key still reports it existed
        cache.put("b", 2, ttl=1.0)
        clock.advance(2)
        assert cache.delete("b") is True
        assert cache.size() == 0


def test_size_mixed_expired_valid_items():
    clock = FakeClock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=5, default_ttl=60.0)
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=100.0)
        cache.put("c", 3, ttl=20.0)
        clock.advance(6)
        assert cache.size() == 2  # 'a' expired
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2
```

## Run

```bash
python -m pytest test_ttl_cache.py -v
```

## Design notes

- **Internals:** a `_Node`-based doubly-linked list (head = LRU, tail = MRU) plus a `Dict[str, _Node]` — no `OrderedDict`, per your requirement. All list operations (insert at tail, unlink, move-to-MRU) are O(1).
- **Expiry semantics:** an item is expired when `expires_at <= now`, so a `ttl=0` item is immediately expired. Expired items are purged lazily — on access (`get`, `delete`, `size`, `put`).
- **Eviction:** on insert into a full cache, `put` first purges all expired items (this is the "clear them all first" case), then evicts the head (LRU) node, which is guaranteed non-expired after the purge. Updates to an existing key do not count as evictions and refresh both value and TTL.
- **Complexity caveat:** `get`, update-`put`, and `delete` are O(1) average. `size()` and the full-cache branch of `put` are O(n) worst case, since an exact count of non-expired items requires checking each node — the price of exact expiry semantics with lazy cleanup. If you ever need strictly O(1) `size()`, the usual answer is a min-heap keyed by `expires_at` (at the cost of O(log n) `put`), or an approximate size.
- The cache is **not thread-safe** — add a lock if you need concurrent access.