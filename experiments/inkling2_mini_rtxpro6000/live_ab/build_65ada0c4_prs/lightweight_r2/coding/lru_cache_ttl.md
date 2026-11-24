Here's the implementation and test suite.

## `ttl_cache.py`

```python
"""A least-recently-used (LRU) cache with time-based expiration."""

import time
from typing import Any, Dict, Optional


class _Node:
    """A single entry in the doubly-linked list."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(
        self,
        key: Optional[str] = None,
        value: Any = None,
        expires_at: float = 0.0,
    ) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """An LRU cache with per-entry TTL expiration.

    Internally uses a hash map (key -> node) for O(1) lookups and a
    doubly-linked list ordered most-recently-used (head) to
    least-recently-used (tail) for O(1) recency updates and eviction.
    Expiration is lazy: expired entries are removed when they are
    accessed or via size().
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` entries, each expiring
        `default_ttl` seconds after insertion."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        # Sentinel nodes: head is the MRU side, tail is the LRU side.
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the linked list (leaves it in the hash map)."""
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _insert_front(self, node: _Node) -> None:
        """Insert `node` at the head (most-recently-used position)."""
        assert self._head.next is not None
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove(self, node: _Node) -> None:
        """Remove `node` from both the linked list and the hash map."""
        assert node.key is not None
        self._unlink(node)
        del self._map[node.key]

    def _move_to_front(self, node: _Node) -> None:
        """Mark `node` as most-recently-used."""
        self._unlink(node)
        self._insert_front(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Return True if `node` has expired at time `now`."""
        return now >= node.expires_at

    def _evict(self, now: float) -> None:
        """Clear expired entries from the LRU end, then evict the LRU entry."""
        # Phase 1: drop expired entries (lazy cleanup), LRU-first.
        while True:
            node = self._tail.prev
            if node is self._head:
                break
            if self._is_expired(node, now):
                self._remove(node)
            else:
                break
        # Phase 2: evict the least-recently-used non-expired entry.
        while len(self._map) > self._capacity:
            node = self._tail.prev
            if node is self._head:
                break
            self._remove(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and not expired, else None.

        A successful lookup marks the entry as most-recently-used.
        Expired entries are removed as a side effect.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if self._is_expired(node, now):
            self._remove(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key` with `value`.

        `ttl` overrides the cache's default TTL (in seconds); if omitted, the
        default is used. The entry becomes most-recently-used. If the cache is
        full, expired entries are cleared from the LRU end first, then the
        least-recently-used non-expired entry is evicted.
        """
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0")
        now = time.monotonic()
        existing = self._map.get(key)
        if existing is not None:
            # Replace the old entry: drop it and re-insert as most-recently-used.
            self._unlink(existing)
            del self._map[key]
        effective_ttl = ttl if ttl is not None else self._default_ttl
        node = _Node(key=key, value=value, expires_at=now + effective_ttl)
        self._insert_front(node)
        self._map[key] = node
        if len(self._map) > self._capacity:
            self._evict(now)

    def delete(self, key: str) -> bool:
        """Remove `key` if present. Returns True if it existed, False otherwise."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired entries.

        Expired entries are removed as a side effect (lazy cleanup).
        Note: this is O(n) in the worst case because expiry can only be
        detected by checking each entry's timestamp.
        """
        now = time.monotonic()
        node = self._tail.prev
        while node is not self._head:
            prev_node = node.prev
            if self._is_expired(node, now):
                self._remove(node)
            node = prev_node
        return len(self._map)
```

## `test_ttl_cache.py`

```python
"""Tests for TTLCache. All time-dependent behavior uses a mocked clock."""

from typing import Any, Callable, Dict

import pytest
from unittest.mock import patch

from ttl_cache import TTLCache


class FakeClock:
    """Manually advanced stand-in for time.monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    """Yield a FakeClock; time.monotonic() is mocked to return its time."""
    fake = FakeClock()
    with patch("time.monotonic", side_effect=lambda: fake.now):
        yield fake


def test_basic_get_and_put(clock):
    cache = TTLCache(capacity=3, default_ttl=60.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    cache.put("a", 2)  # update overwrites the value
    assert cache.get("a") == 2
    assert cache.get("missing") is None
    assert cache.size() == 1


def test_capacity_evicts_least_recently_used(clock):
    cache = TTLCache(capacity=2, default_ttl=60.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # 'a' becomes MRU; 'b' is now LRU
    cache.put("c", 3)           # forces eviction of 'b' (LRU)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size() == 2


def test_ttl_expiry(clock):
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    clock.advance(9.0)
    assert cache.get("a") == 1
    clock.advance(2.0)  # t = 11.0, past the 10.0s TTL
    assert cache.get("a") is None
    assert cache.size() == 0  # expired entry was cleaned up lazily


def test_custom_per_key_ttl(clock):
    cache = TTLCache(capacity=3, default_ttl=100.0)
    cache.put("short", "s", ttl=5.0)
    cache.put("long", "l")  # uses the default TTL
    clock.advance(6.0)
    assert cache.get("short") is None  # expired at t = 5.0
    assert cache.get("long") == "l"    # still valid (expires at t = 100.0)
    clock.advance(95.0)  # t = 101.0
    assert cache.get("long") is None


def test_delete(clock):
    cache = TTLCache(capacity=3, default_ttl=60.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False           # already removed
    assert cache.delete("never-existed") is False


def test_size_with_mixed_expired_and_valid(clock):
    cache = TTLCache(capacity=5, default_ttl=60.0)
    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=5.0)
    cache.put("c", 3, ttl=60.0)
    clock.advance(6.0)
    assert cache.size() == 1  # only 'c' survives
    assert cache.get("c") == 3
    clock.advance(60.0)  # t = 66.0
    assert cache.size() == 0  # 'c' expired too
```

## Notes

- **O(1) average:** `get`, `put`, and `delete` are O(1) average/amortized (the eviction loop removes one entry per iteration, and each expired entry is purged at most once). `size()` is O(n) worst case — an exact non-expired count fundamentally requires checking every entry's timestamp, so it's the one operation that isn't O(1).
- **Eviction order:** on `put`, expired entries are cleared from the LRU end first (which fully clears the cache when everything is expired), then the least-recently-used *non-expired* entry is evicted.
- **`delete` semantics:** returns `True` if the key existed in the cache (even if already expired), since "existed" is about presence, not validity.
- **No `time.sleep`** anywhere — the `clock` fixture patches `time.monotonic` with a `side_effect` lambda and advances it manually, giving deterministic time.

Run with:

```bash
pytest test_ttl_cache.py -v
```