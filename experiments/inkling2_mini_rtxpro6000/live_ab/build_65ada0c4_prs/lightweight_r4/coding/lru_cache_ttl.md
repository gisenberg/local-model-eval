Here's a complete implementation. One honest note up front: because expiry is **lazy**, `size()` must sweep the map to know exactly how many items are still live, so it's O(n). `get`/`put`/`delete`/eviction are O(1) average (`put` only does a sweep when the cache is full). I've documented that in the docstrings.

## `ttl_cache.py`

```python
"""Time-based TTL LRU cache.

Internal structure (no OrderedDict):
- A doubly-linked list holding items in recency order
  (head = most-recently-used, tail = least-recently-used).
- A hash map from key -> list node.

Expiration is lazy: expired items are removed when they are encountered,
so no background thread is needed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class _Node:
    """One entry in the recency list."""

    __slots__ = ("key", "value", "expire_at", "prev", "next")

    def __init__(self, key: str, value: Any, expire_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expire_at: float = expire_at
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """An LRU cache with per-key TTL expiration."""

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create the cache.

        Args:
            capacity: Max number of items. Must be >= 1.
            default_ttl: Seconds until an item expires by default. Must be > 0.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity: int = capacity
        self.default_ttl: float = float(default_ttl)
        self._nodes: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most-recently-used
        self._tail: Optional[_Node] = None  # least-recently-used

    # ---------- public API ----------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for ``key`` if it exists and is not expired, else None.

        A live hit marks the key as most-recently-used. An expired entry is
        removed (lazy cleanup).
        """
        node = self._nodes.get(key)
        if node is None:
            return None
        if node.expire_at <= time.monotonic():
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update ``key``.

        ``ttl`` (seconds) overrides ``default_ttl`` when provided. If the cache is
        at capacity, expired items are purged first (an all-expired cache is
        cleared), then the least-recently-used non-expired item is evicted.
        """
        now = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl

        node = self._nodes.get(key)
        if node is not None:
            if node.expire_at > now:  # live: update in place
                node.value = value
                node.expire_at = now + effective_ttl
                self._move_to_front(node)
                return
            self._remove_node(node)  # stale: treat as new insert

        if len(self._nodes) >= self.capacity:
            self._purge_expired(now)  # clears all-expired caches too
        if len(self._nodes) >= self.capacity:
            lru = self._tail  # non-expired after purge
            if lru is not None:
                self._remove_node(lru)

        new_node = _Node(key, value, now + effective_ttl)
        self._nodes[key] = new_node
        self._insert_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove ``key``. Returns True if it existed (live or expired), else False."""
        node = self._nodes.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items.

        Expired items are removed lazily. Because expiry is lazy, this sweeps the
        map and is O(n); other operations are O(1) average.
        """
        now = time.monotonic()
        self._purge_expired(now)
        return len(self._nodes)

    # ---------- internals ----------

    def _purge_expired(self, now: float) -> None:
        expired = [n for n in self._nodes.values() if n.expire_at <= now]
        for n in expired:
            self._remove_node(n)

    def _detach(self, node: _Node) -> None:
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

    def _insert_front(self, node: _Node) -> None:
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _move_to_front(self, node: _Node) -> None:
        if node is self._head:
            return
        self._detach(node)
        self._insert_front(node)

    def _remove_node(self, node: _Node) -> None:
        self._detach(node)
        del self._nodes[node.key]
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache. time.monotonic is patched for deterministic time control."""

from unittest.mock import patch

from ttl_cache import TTLCache


def test_basic_get_put():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        cache.put("a", 42)  # overwrite existing
        assert cache.get("a") == 42
        assert cache.size() == 2


def test_capacity_eviction_lru_order():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")      # a becomes most-recently-used; b is LRU
        cache.put("c", 3)   # cache full -> evict b
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("a", 1)
        mock_time.return_value = 4.9
        assert cache.get("a") == 1   # still valid
        mock_time.return_value = 6.0
        assert cache.get("a") is None  # expired
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("short", 1, ttl=2.0)  # custom override
        cache.put("long", 2)            # default ttl
        mock_time.return_value = 3.0
        assert cache.get("short") is None  # custom ttl expired
        assert cache.get("long") == 2      # default ttl still live


def test_delete():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False   # already gone
        assert cache.get("a") is None
        assert cache.size() == 0


def test_size_with_mixed_expired_valid():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2)            # default ttl
        cache.put("c", 3, ttl=20.0)
        mock_time.return_value = 6.0
        assert cache.size() == 2     # "a" expired; "b" and "c" valid
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
```

Run with:

```bash
pytest test_ttl_cache.py -v
```

## Design notes / decisions

- **List order**: head = most-recently-used, tail = least-recently-used. `get` on a live key detaches/re-inserts at the head; eviction removes the tail.
- **Expiry boundary**: an item is expired when `now >= expire_at` (i.e. `expire_at <= now`).
- **Eviction path** (`put` at capacity): purge all expired items (this is what clears an all-expired cache), then evict the tail — the LRU non-expired item — then insert.
- **`delete` on an expired key**: the stale entry is removed and `True` is returned (it "existed"); this is documented in the docstring.
- **No `time.sleep`**: every test controls time via `patch("time.monotonic")` and advancing `mock_time.return_value`.
- **Complexity**: `get`/`delete` O(1) average; `put` O(1) average (O(n) only when full, since it may purge expired items); `size` O(n) because exact live-count accounting requires a sweep. If you need `size()` in O(1) average, the standard fix is a secondary min-heap keyed by `expire_at` — happy to add it, but it goes beyond the requested list+map structure.