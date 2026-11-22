Here's a complete implementation. The core is a doubly-linked list (ordered MRU→LRU) plus a hash map from key → node, with lazy expiry checks.

## `ttl_cache.py`

```python
"""A time-based expiring LRU cache.

Data structure: a doubly-linked list ordered from most-recently-used (head)
to least-recently-used (tail), plus a hash map from key -> node for O(1)
lookup. Time is tracked with ``time.monotonic()``.

Complexity (average): ``get``/``put``/``delete`` are O(1). ``put``'s insert
path is O(1) amortized (stale tail entries are removed at most once each).
``size()`` performs lazy cleanup of expired entries, which is O(n) worst
case per call — each expired entry is removed at most once.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Internal doubly-linked-list node (not part of the public API)."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        # List order: head (MRU) on the left, tail (LRU) on the right.
        # prev -> toward the head (more recently used)
        # next -> toward the tail (less recently used)
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Fixed-capacity LRU cache with per-entry TTL expiration.

    Args:
        capacity: maximum number of entries (>= 1).
        default_ttl: seconds until an entry expires when no custom ttl
            is supplied to :meth:`put`.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Raises:
            ValueError: if ``capacity`` is less than 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most-recently-used
        self._tail: Optional[_Node] = None  # least-recently-used

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_front(self, node: _Node) -> None:
        """Insert ``node`` as the most-recently-used entry. O(1)."""
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _unlink(self, node: _Node) -> None:
        """Unlink ``node`` from the list, updating head/tail and neighbors. O(1)."""
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

    def _move_to_front(self, node: _Node) -> None:
        """Mark ``node`` as most-recently-used. O(1)."""
        if node is self._head:
            return
        self._unlink(node)
        self._insert_front(node)

    def _evict_for_insert(self, now: float) -> None:
        """Free space for one new entry: drop expired tail entries (this
        clears the whole cache when all entries are expired), then evict
        the least-recently-used non-expired entry if still at capacity.
        O(1) amortized.
        """
        while self._tail is not None and self._tail.expires_at <= now:
            tail = self._tail
            self._unlink(tail)
            del self._map[tail.key]
        if len(self._map) >= self._capacity and self._tail is not None:
            tail = self._tail  # non-expired LRU entry
            self._unlink(tail)
            del self._map[tail.key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for ``key`` if present and not expired, and mark
        it most-recently-used; otherwise return None. O(1) average."""
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            self._unlink(node)
            del self._map[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update ``key`` with ``value``.

        If ``key`` exists and is not expired, its value and expiry are
        replaced and it is marked most-recently-used. Otherwise it is
        inserted as new; if the cache is at capacity, expired entries are
        dropped first, then the least-recently-used non-expired entry is
        evicted. ``ttl`` overrides the default TTL for this entry.
        O(1) average.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl

        node = self._map.get(key)
        if node is not None:
            if node.expires_at <= now:
                # Stale entry: drop it and treat this as a fresh insert.
                self._unlink(node)
                del self._map[key]
                node = None
            else:
                node.value = value
                node.expires_at = now + effective_ttl
                self._move_to_front(node)
                return

        if len(self._map) >= self._capacity:
            self._evict_for_insert(now)

        node = _Node(key, value, now + effective_ttl)
        self._map[key] = node
        self._insert_front(node)

    def delete(self, key: str) -> bool:
        """Remove ``key`` if present; return True if it existed. O(1)."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._unlink(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired entries, lazily removing expired
        ones. O(n) worst case (each expired entry is removed at most once).
        """
        now = time.monotonic()
        node = self._head
        while node is not None:
            nxt = node.next
            if node.expires_at <= now:
                self._unlink(node)
                del self._map[node.key]
            node = nxt
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache.

Time is controlled by patching ``time.monotonic`` with a deterministic fake
clock — no real sleeping is used.
"""

from typing import Any
from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self.now: float = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@patch("time.monotonic")
def test_basic_get_put(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1)
    assert cache.get("a") == 1

    assert cache.get("missing") is None

    cache.put("a", 2)  # update existing key
    assert cache.get("a") == 2
    assert cache.size() == 1


@patch("time.monotonic")
def test_capacity_eviction_lru_order(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")        # "a" is now most-recently-used
    cache.put("c", 3)     # at capacity -> evict least-recently-used ("b")

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size() == 2


@patch("time.monotonic")
def test_ttl_expiry(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1)
    assert cache.get("a") == 1

    clock.advance(10.0)  # exactly at expiry -> expired
    assert cache.get("a") is None
    assert cache.size() == 0

    # New entries are independent of the expired one.
    cache.put("b", 2)
    clock.advance(5.0)
    assert cache.get("b") == 2
    assert cache.size() == 1


@patch("time.monotonic")
def test_custom_per_key_ttl(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1)               # default TTL -> expires at t=10
    cache.put("b", 2, ttl=5.0)      # custom TTL -> expires at t=5

    clock.advance(6.0)

    assert cache.get("b") is None   # custom TTL elapsed
    assert cache.get("a") == 1      # default TTL still valid
    assert cache.size() == 1


@patch("time.monotonic")
def test_delete(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("a") is False        # already removed
    assert cache.delete("never-existed") is False

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.size() == 1


@patch("time.monotonic")
def test_size_with_mixed_expired_and_valid(mock_monotonic: Any) -> None:
    clock = FakeClock()
    mock_monotonic.side_effect = clock
    cache = TTLCache(capacity=5, default_ttl=10.0)

    cache.put("a", 1)               # expires at t=10
    cache.put("b", 2)               # expires at t=10
    clock.advance(5.0)
    cache.put("c", 3)               # expires at t=15

    clock.advance(6.0)               # t=11 -> a and b expired, c valid

    assert cache.size() == 1         # only "c" survives
    assert cache.get("c") == 3
    assert cache.get("a") is None
    assert cache.size() == 1         # "a" was lazily removed
```

## Notes on design

- **List convention**: head = most-recently-used, tail = least-recently-used. `prev` points toward the head, `next` toward the tail. Every operation that touches an entry (`get`, `put` update) calls `_move_to_front`, which unlinks and re-inserts in O(1).
- **Expiry is lazy**: entries are only removed when touched (`get`, `put`, `delete`) or when `size()` sweeps the list. That's why `size()` is O(n) worst case — items that expired *between* calls can't be detected without a scan (with mixed per-key TTLs there's no recency/expiry ordering to exploit). Each stale entry is removed at most once, so total cleanup work is bounded by the number of insertions.
- **Eviction in `put`**: tail entries that are expired are dropped first (this handles the "all items expired → clear them all" case), then the LRU non-expired entry — the new tail — is evicted. After the cleanup, `len(map)` is guaranteed `< capacity`, so the insert always succeeds.
- **Time source**: `time.monotonic()` everywhere, and tests patch it with `@patch("time.monotonic")` plus a `FakeClock` with `side_effect`, so no `time.sleep()` and no flakiness.

Run with:

```bash
pytest test_ttl_cache.py -v
```