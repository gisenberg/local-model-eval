Here's a complete implementation. The design uses a doubly-linked list (MRU → LRU) with sentinel nodes, a hash map for O(1) lookup, and a min-heap of `(expires_at, version, key)` for lazy expiry purges (so `size()` and capacity checks don't scan the whole map).

## `ttl_cache.py`

```python
"""Time-to-live (TTL) least-recently-used cache.

Internal structure:
  - doubly-linked list with sentinel head/tail nodes, ordered MRU -> LRU
  - hash map: key -> node
  - min-heap of (expires_at, version, key) for lazy expiry purges

Time comes from time.monotonic(), which is immune to wall-clock changes.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _Node:
    """A node in the doubly-linked LRU list."""

    key: str
    value: Any
    expires_at: float
    version: int
    prev: Optional["_Node"] = None  # toward MRU (head)
    next: Optional["_Node"] = None  # toward LRU (tail)


class TTLCache:
    """A capacity-bounded LRU cache with per-item TTL expiration.

    Items are evicted when they are either expired (lazily removed on
    access) or pushed beyond capacity (least-recently-used first).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of live (non-expired) items.
            default_ttl: Seconds until an item expires when no custom
                per-item TTL is supplied.
        """
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._heap: List[Tuple[float, int, str]] = []
        self._version = 0
        # Sentinels: head = MRU end, tail = LRU end.
        self._head = _Node("\x00head", None, 0.0, 0)
        self._tail = _Node("\x00tail", None, 0.0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    # -- internal helpers -------------------------------------------------

    def _insert_front(self, node: _Node) -> None:
        """Insert node at the MRU end of the list. O(1)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the list. O(1)."""
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_front(self, node: _Node) -> None:
        """Mark node as most-recently-used. O(1)."""
        self._remove_node(node)
        self._insert_front(node)

    def _purge_expired(self, now: float) -> None:
        """Remove all nodes whose expiry has passed. Amortized O(log n) per removed item."""
        while self._heap and self._heap[0][0] <= now:
            expires_at, version, key = heapq.heappop(self._heap)
            node = self._map.get(key)
            # Discard stale heap entries (old versions, deleted keys).
            if node is not None and node.version == version and node.expires_at == expires_at:
                self._remove_node(node)
                del self._map[key]

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for key if present and unexpired, else None.

        A hit marks key as most-recently-used. Expired keys are removed
        and reported as absent.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            del self._map[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key, marking it most-recently-used.

        Expired items are cleared first; if the cache is still at capacity,
        the least-recently-used non-expired item is evicted.

        Args:
            ttl: Custom time-to-live in seconds; overrides default_ttl.
                A ttl of 0 (or negative) means the item expires immediately.
        """
        now = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = now + effective_ttl

        node = self._map.get(key)
        if node is not None:
            if node.expires_at > now:
                # Refresh a live entry.
                node.value = value
                node.expires_at = expires_at
                node.version += 1
                heapq.heappush(self._heap, (expires_at, node.version, key))
                self._move_to_front(node)
                return
            # Expired entry: drop it and fall through to a fresh insert.
            self._remove_node(node)
            del self._map[key]

        self._purge_expired(now)

        if len(self._map) >= self.capacity:
            lru = self._tail.prev  # least-recently-used non-expired node
            self._remove_node(lru)
            del self._map[lru.key]

        self._version += 1
        new_node = _Node(key, value, expires_at, self._version)
        self._map[key] = new_node
        self._insert_front(new_node)
        heapq.heappush(self._heap, (expires_at, self._version, key))

    def delete(self, key: str) -> bool:
        """Remove key and return True if it existed and was not expired.

        Missing or expired keys return False (expired entries are removed).
        """
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        del self._map[key]
        if node.expires_at <= time.monotonic():
            return False  # expired entries are treated as absent
        return True

    def size(self) -> int:
        """Return the count of non-expired items.

        Lazily removes expired items before counting.
        """
        self._purge_expired(time.monotonic())
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache. time.monotonic is mocked via unittest.mock.patch."""

from unittest.mock import patch

from ttl_cache import TTLCache


def test_basic_get_put():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        cache.put("a", 2)  # overwrite
        assert cache.get("a") == 2
        assert cache.get("missing") is None
        assert cache.size() == 1


def test_capacity_eviction_lru_order():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1  # a -> MRU, b is now LRU
        cache.put("c", 3)  # should evict b, not a
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        mock_time.return_value = 1005.0
        assert cache.get("a") == 1  # still valid
        mock_time.return_value = 1010.0
        assert cache.get("a") is None  # expires exactly at 1010
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=4, default_ttl=100.0)
        cache.put("short", 1, ttl=5.0)
        cache.put("long", 2)  # uses default_ttl
        mock_time.return_value = 1006.0
        assert cache.get("short") is None  # expired at 1005
        assert cache.get("long") == 2  # expires at 1100


def test_delete():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False  # already gone
        assert cache.delete("missing") is False
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_and_valid_items():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=4, default_ttl=100.0)
        cache.put("expired1", 1, ttl=5.0)
        cache.put("valid1", 2)
        cache.put("expired2", 3, ttl=5.0)
        mock_time.return_value = 1006.0
        assert cache.size() == 1
        assert cache.get("valid1") == 2
        assert cache.size() == 1


def test_eviction_clears_expired_items_first():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2)
        mock_time.return_value = 1006.0
        cache.put("c", 3)  # expired 'a' cleared; 'b' must NOT be evicted
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2
```

Run with:

```bash
pytest test_ttl_cache.py -v
```

## Notes

- **Complexity:** `get`, `delete`, and live-entry `put` are O(1); `put`/`size` are amortized O(log n) because of the expiry-heap purge (each item is pushed and popped at most once, so purges amortize away; the LRU list itself is strictly O(1)).
- **Expiry boundary:** an item is expired when `now >= expires_at` (inclusive).
- **Lazy cleanup:** expired items are removed on access (`get`/`delete`/`put`/`size`), never eagerly in a background thread — consistent with the lazy-cleanup requirement.
- **`delete` semantics:** a key that exists but is expired is treated as absent — it's removed and `False` is returned.
- **Stale heap entries:** a key updated via `put` leaves an older `(expires_at, version, key)` entry in the heap; a per-node `version` counter lets the purge discard those entries in O(1) per stale entry instead of leaking them.