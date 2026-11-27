# TTL-based LRU Cache

Two files: `ttl_cache.py` (implementation) and `test_ttl_cache.py` (tests).

## `ttl_cache.py`

```python
"""A TTL (time-to-live) cache with LRU eviction.

Internally uses a doubly-linked list (head = most-recently-used,
tail = least-recently-used) plus a hash map keyed by item key,
giving O(1) average get/put/delete. Expiration is handled lazily:
expired entries are removed when accessed or when capacity forces
eviction.
"""

import time
from typing import Any, Dict, Optional


class _Node:
    """A single entry in the doubly-linked list."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """An LRU cache with per-entry TTL expiration.

    Head of the list is most-recently-used; tail is least-recently-used.
    Time is tracked via time.monotonic().
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of live (non-expired) items.
            default_ttl: Seconds an item lives by default.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # MRU
        self._tail: Optional[_Node] = None  # LRU

    # -- internal helpers -------------------------------------------------

    def _detach(self, node: _Node) -> None:
        """Unlink node from the list (leaves it in the map)."""
        prev, nxt = node.prev, node.next
        if prev is not None:
            prev.next = nxt
        else:
            self._head = nxt
        if nxt is not None:
            nxt.prev = prev
        else:
            self._tail = prev
        node.prev = None
        node.next = None

    def _insert_front(self, node: _Node) -> None:
        """Insert node as most-recently-used (at the head)."""
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _move_to_front(self, node: _Node) -> None:
        """Promote node to most-recently-used position."""
        if node is self._head:
            return
        self._detach(node)
        self._insert_front(node)

    def _remove_node(self, node: _Node) -> None:
        """Remove node from both list and map."""
        self._detach(node)
        del self._map[node.key]

    def _evict_tail(self) -> None:
        """Remove the least-recently-used entry."""
        if self._tail is not None:
            self._remove_node(self._tail)

    def _purge_expired(self, now: float) -> None:
        """Remove all entries whose TTL has elapsed (O(n), used lazily)."""
        node = self._head
        while node is not None:
            next_node = node.next
            if node.expires_at <= now:
                self._detach(node)
                del self._map[node.key]
            node = next_node

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return value for key if present and unexpired; else None.

        On a hit, the key is promoted to most-recently-used.
        """
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key with value; refresh TTL and LRU position.

        Args:
            key: Item key.
            value: Item value.
            ttl: Per-key TTL in seconds; falls back to default_ttl.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = now + effective_ttl

        node = self._map.get(key)
        if node is not None:
            # Update existing entry (even if expired) and refresh it.
            self._detach(node)
            node.value = value
            node.expires_at = expires_at
            self._insert_front(node)
            return

        # New key: make room if at capacity. Purge expired items first
        # (this also covers the "all items expired" case), then evict
        # the LRU non-expired entry (the tail) if still at capacity.
        if len(self._map) >= self._capacity:
            self._purge_expired(now)
            if len(self._map) >= self._capacity:
                self._evict_tail()

        new_node = _Node(key, value, expires_at)
        self._insert_front(new_node)
        self._map[key] = new_node

    def delete(self, key: str) -> bool:
        """Remove key; return True if it existed (regardless of expiry)."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items (lazily removes expired ones)."""
        now = time.monotonic()
        self._purge_expired(now)
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Deterministic tests for TTLCache (time.monotonic is mocked)."""

from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """A controllable stand-in for time.monotonic."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_clock() -> FakeClock:
    return FakeClock(0.0)


def test_basic_get_put():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        assert cache.size() == 2


def test_capacity_eviction_lru_order():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")          # promote a to MRU; b is now LRU
        cache.put("c", 3)       # forces eviction of b
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3


def test_ttl_expiry():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=10)
        cache.put("a", 1)
        clock.advance(5)
        assert cache.get("a") == 1      # still valid
        clock.advance(6)                 # t=11 > ttl 10
        assert cache.get("a") is None


def test_custom_per_key_ttl():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=100)
        cache.put("a", 1, ttl=10)        # expires at t=10
        cache.put("b", 2)                # default ttl=100
        clock.advance(15)
        assert cache.get("a") is None
        assert cache.get("b") == 2


def test_delete():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=3, default_ttl=10)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.delete("a") is False   # already removed
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_valid():
    clock = make_clock()
    with patch("time.monotonic", side_effect=clock):
        cache = TTLCache(capacity=5, default_ttl=10)
        cache.put("a", 1, ttl=10)        # expires t=10
        cache.put("b", 2, ttl=30)        # expires t=30
        cache.put("c", 3, ttl=20)        # expires t=20
        clock.advance(12)                # t=12
        assert cache.size() == 2         # a expired; b, c valid
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        clock.advance(10)                # t=22
        assert cache.size() == 1         # a and c expired; only b valid
```

## Notes

- **Design**: head = MRU, tail = LRU. The hash map provides O(1) lookup; the linked list provides O(1) eviction/promotion. `get` promotes hits to MRU; `put` on a new key evicts the tail (the LRU).
- **Eviction semantics**: on a new-key `put` at capacity, expired entries are purged first (this handles the "all items expired → clear them all" case); if still at capacity, the LRU non-expired entry (the tail) is evicted. Capacity therefore counts live items.
- **Complexity honesty**: `get`, `delete`, and the update path of `put` are strictly O(1). `size()` and the capacity-triggered path in `put` do a lazy expiry scan, so they're O(n) worst-case — that's inherent to lazy TTL cleanup without a background timer; they stay O(1) average when no expired items are present.
- **Time**: everything flows through `time.monotonic()`, and tests patch it with a `FakeClock` (advancing via `advance()`), so no `time.sleep` is used and results are fully deterministic.

Run with:

```bash
pytest test_ttl_cache.py -v
```