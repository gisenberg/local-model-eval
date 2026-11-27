Here's a complete implementation with the required design.

## `ttl_cache.py`

```python
"""Time-based TTL cache with LRU eviction.

Internal structure:
  - Hash map: key -> node            (O(1) lookup)
  - Doubly-linked list: recency order (head = most recently used)
                                     (O(1) move-to-head / evict-from-tail)

Expiry is lazy: expired nodes are removed when observed (on get, or during
the cleanup sweeps in put/size), never by a background timer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node holding one cached entry."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """A fixed-capacity LRU cache with per-entry time-to-live."""

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache with a maximum number of entries and a
        default TTL (seconds) applied to entries inserted without one."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most recently used
        self._tail: Optional[_Node] = None  # least recently used

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and not expired, else None.
        A successful lookup marks `key` most-recently-used."""
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`. `ttl` (seconds) overrides the cache-level
        default. If the cache is at capacity, expired entries are cleared
        first; if it is still full, the least-recently-used non-expired
        entry is evicted."""
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl

        node = self._map.get(key)
        if node is not None:
            if node.expires_at > now:
                # Refresh value + TTL, mark most-recently-used.
                node.value = value
                node.expires_at = now + effective_ttl
                self._move_to_head(node)
                return
            # Expired entry: treat the put as a fresh insert.
            self._remove_node(node)

        if len(self._map) >= self._capacity:
            self._prune_expired(now)
            if len(self._map) >= self._capacity:
                # LRU non-expired entry is the tail (after pruning).
                assert self._tail is not None
                self._remove_node(self._tail)

        new_node = _Node(key, value, now + effective_ttl)
        self._insert_head(new_node)
        self._map[key] = new_node

    def delete(self, key: str) -> bool:
        """Remove `key` if present. Return True if it existed, else False."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired entries (lazy cleanup sweep)."""
        self._prune_expired(time.monotonic())
        return len(self._map)

    # ---- internals ----

    def _insert_head(self, node: _Node) -> None:
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _unlink(self, node: _Node) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

    def _remove_node(self, node: _Node) -> None:
        self._unlink(node)
        self._map.pop(node.key, None)

    def _move_to_head(self, node: _Node) -> None:
        if node is self._head:
            return
        self._unlink(node)
        self._insert_head(node)

    def _prune_expired(self, now: float) -> None:
        node = self._head
        while node is not None:
            nxt = node.next
            if node.expires_at <= now:
                self._unlink(node)
                self._map.pop(node.key, None)
            node = nxt
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache — deterministic time via mocked time.monotonic."""

from unittest.mock import patch

from ttl_cache import TTLCache


def test_basic_get_put():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        assert cache.size() == 1


def test_capacity_eviction_lru_order():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1  # mark "a" as most-recently-used
        cache.put("c", 3)           # should evict "b" (least-recently-used)
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("a", 1)
        mock_time.return_value = 5.0  # expires exactly at t=5.0
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("short", 1, ttl=1.0)  # expires at t=1.0
        cache.put("long", 2)            # expires at t=100.0
        mock_time.return_value = 2.0
        assert cache.get("short") is None
        assert cache.get("long") == 2


def test_delete():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False


def test_size_with_mixed_expired_and_valid_items():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=2.0)
        cache.put("a", 1)               # expires at t=2.0
        mock_time.return_value = 1.0
        cache.put("b", 2)               # expires at t=3.0
        mock_time.return_value = 2.0
        cache.put("c", 3, ttl=3.0)      # expires at t=5.0
        mock_time.return_value = 4.0
        assert cache.size() == 1        # only "c" remains
        assert cache.get("c") == 3
```

Run with:

```bash
python -m pytest test_ttl_cache.py -v
```

**Design notes**

- `get`/`delete` are strictly O(1): dict lookup + O(1) list surgery (move-to-head / unlink).
- `put` is O(1) for inserts and updates at non-capacity; when at capacity it does a lazy sweep removing expired entries first (so "all items are expired → clear them all"), then evicts the tail (LRU non-expired). Each entry is removed at most once, so cleanup is amortized O(1) per entry.
- `size()` does a lazy cleanup sweep over the list and returns the remaining count.
- Expiry uses `time.monotonic()` (never `time.time()`, never `sleep`), and an entry expires when `expires_at <= now` (boundary is inclusive).
- Updates on an existing key refresh both value and TTL, and mark the key most-recently-used — same as a fresh insert.
- Tests patch `time.monotonic` via `unittest.mock.patch` and control the clock by setting `mock_time.return_value`, so they're deterministic with no `time.sleep`.