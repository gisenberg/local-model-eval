

```python
"""
TTLCache: An LRU cache with time-based expiration implemented using
a doubly-linked list and hash map for O(1) average-time operations.
"""

from typing import Any, Optional
import time


class _Node:
    """Internal doubly-linked list node storing cache entry metadata."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with per-entry time-to-live (TTL) expiration.

    Uses a doubly-linked list (most-recently-used at head, least-recently-used at tail)
    combined with a hash map for O(1) average-time get, put, and delete operations.
    Expired entries are lazily cleaned up during access operations.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTL cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict[str, _Node] = {}

        # Sentinel nodes to simplify linked list manipulation
        self._head: _Node = _Node("__head__", None, 0.0)  # MRU sentinel
        self._tail: _Node = _Node("__tail__", None, 0.0)  # LRU sentinel
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal linked-list helpers
    # ------------------------------------------------------------------

    def _remove_node(self, node: _Node) -> None:
        """Unlink a node from the doubly-linked list in O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _insert_at_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (MRU position) in O(1)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _is_expired(self, node: _Node) -> bool:
        """Check whether a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for *key* if it exists and has not expired.

        Accessing a key promotes it to most-recently-used position.
        Returns None if the key is absent or expired (expired entries are removed).

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if not found / expired.
        """
        node: Optional[_Node] = self._map.get(key)
        if node is None:
            return None

        if self._is_expired(node):
            # Lazy cleanup: remove expired entry
            self._remove_node(node)
            del self._map[key]
            return None

        # Promote to MRU
        self._remove_node(node)
        self._insert_at_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a cache entry.

        If the key already exists, its value and TTL are refreshed and it is
        promoted to MRU. If the cache is at capacity, the least-recently-used
        non-expired item is evicted. If every existing item is expired, they
        are all cleared before insertion.

        Args:
            key: The cache key.
            value: The value to store.
            ttl: Optional per-entry TTL in seconds; overrides the default TTL.
        """
        effective_ttl: float = ttl if ttl is not None else self._default_ttl
        expiry: float = time.monotonic() + effective_ttl

        # Update existing key
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            # Promote to MRU
            self._remove_node(node)
            self._insert_at_head(node)
            return

        # Evict if at capacity
        if len(self._map) >= self._capacity:
            self._evict()

        # Insert new node at head (MRU)
        new_node = _Node(key, value, expiry)
        self._insert_at_head(new_node)
        self._map[key] = new_node

    def delete(self, key: str) -> bool:
        """
        Remove *key* from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if the key existed and was removed, False otherwise.
        """
        node: Optional[_Node] = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        del self._map[key]
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items currently in the cache.

        Performs a lazy cleanup pass: expired items are removed from both the
        hash map and the linked list during this call.

        Returns:
            Count of entries that have not yet expired.
        """
        current_time: float = time.monotonic()
        expired_keys: list[str] = []

        node: _Node = self._head.next
        while node is not self._tail:
            if current_time > node.expiry:
                expired_keys.append(node.key)
            node = node.next

        for k in expired_keys:
            n = self._map[k]
            self._remove_node(n)
            del self._map[k]

        return len(self._map)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """
        Evict the least-recently-used non-expired item.

        Walks from the LRU end (tail) toward MRU. Expired nodes are purged
        along the way. If the entire cache consists of expired items, they
        are all cleared.
        """
        current_time: float = time.monotonic()
        node: Optional[_Node] = self._tail.prev  # start from LRU side
        expired_nodes: list[_Node] = []
        victim: Optional[_Node] = None

        while node is not self._head:
            next_node: Optional[_Node] = node.prev
            if current_time > node.expiry:
                expired_nodes.append(node)
            else:
                # First non-expired item found is our eviction victim
                victim = node
                break
            node = next_node

        # Remove all expired nodes encountered during the walk
        for expired in expired_nodes:
            self._remove_node(expired)
            del self._map[expired.key]

        if victim is None:
            # All items are expired — clear everything
            self._map.clear()
            self._head.next = self._tail
            self._tail.prev = self._head
            return

        # Evict the LRU non-expired victim
        self._remove_node(victim)
        del self._map[victim.key]
```

```python
"""
pytest tests for TTLCache — uses unittest.mock.patch to control time.monotonic()
deterministically without time.sleep().
"""

import pytest
from unittest.mock import patch
from typing import Optional

# We patch time.monotonic globally so the module-level code and cache internals
# all see the same controlled clock.
_clock_value: float = 1000.0


def _fake_monotonic() -> float:
    return _clock_value


def _advance(seconds: float) -> None:
    global _clock_value
    _clock_value += seconds


@pytest.fixture(autouse=True)
def reset_clock():
    """Reset the fake clock before every test."""
    global _clock_value
    _clock_value = 1000.0
    yield


# ---------------------------------------------------------------------------
# Test 1: Basic get / put
# ---------------------------------------------------------------------------

class TestBasicGetPut:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_put_and_get_returns_value(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=5, default_ttl=60.0)
        cache.put("alpha", 42)
        assert cache.get("alpha") == 42

    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_get_missing_key_returns_none(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=5, default_ttl=60.0)
        assert cache.get("nonexistent") is None

    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_put_overwrites_existing_value(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=5, default_ttl=60.0)
        cache.put("k", "first")
        cache.put("k", "second")
        assert cache.get("k") == "second"


# ---------------------------------------------------------------------------
# Test 2: Capacity eviction in LRU order
# ---------------------------------------------------------------------------

class TestCapacityEviction:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_lru_eviction_order(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=3, default_ttl=300.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" so it becomes MRU; "b" is now LRU
        assert cache.get("a") == 1
        # Insert "d" → should evict "b" (the LRU non-expired item)
        cache.put("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4


# ---------------------------------------------------------------------------
# Test 3: TTL expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_item_expires_after_ttl(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=10, default_ttl=5.0)
        cache.put("x", "hello")
        assert cache.get("x") == "hello"

        _advance(5.1)
        assert cache.get("x") is None

    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_item_valid_before_ttl(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=10, default_ttl=5.0)
        cache.put("y", "world")
        _advance(3.0)
        assert cache.get("y") == "world"


# ---------------------------------------------------------------------------
# Test 4: Custom per-key TTL overrides default
# ---------------------------------------------------------------------------

class TestCustomTTL:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_custom_ttl_shorter_than_default(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=10, default_ttl=100.0)
        cache.put("short", "gone_soon", ttl=2.0)
        cache.put("long", "stays", ttl=None)  # uses default 100s

        _advance(3.0)
        assert cache.get("short") is None
        assert cache.get("long") == "stays"

    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_custom_ttl_longer_than_default(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=10, default_ttl=2.0)
        cache.put("persist", "yes", ttl=50.0)

        _advance(3.0)
        assert cache.get("persist") == "yes"


# ---------------------------------------------------------------------------
# Test 5: Delete
# ---------------------------------------------------------------------------

class TestDelete:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_delete_existing_returns_true_and_removes(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=5, default_ttl=60.0)
        cache.put("kill", "me")
        assert cache.delete("kill") is True
        assert cache.get("kill") is None

    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_delete_missing_returns_false(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=5, default_ttl=60.0)
        assert cache.delete("ghost") is False


# ---------------------------------------------------------------------------
# Test 6: size() with mixed expired / valid items
# ---------------------------------------------------------------------------

class TestSizeMixedExpiry:
    @patch("time.monotonic", side_effect=lambda: _clock_value)
    def test_size_reflects_only_valid_items(self, _mock: object) -> None:
        cache: TTLCache = TTLCache(capacity=20, default_ttl=10.0)
        # Insert items with staggered TTLs
        cache.put("a", 1, ttl=3.0)
        cache.put("b", 2, ttl=6.0)
        cache.put("c", 3, ttl=10.0)
        cache.put("d", 4, ttl=20.0)

        # At t=0 all 4 are valid
        assert cache.size() == 4

        # Advance to t=4 → "a" expired
        _advance(4.0)
        assert cache.size() == 3  # b, c, d

        # Advance to t=7 → "b" also expired
        _advance(3.0)
        assert cache.size() == 2  # c, d

        # Advance to t=12 → "c" expired
        _advance(5.0)
        assert cache.size() == 1  # d

        # Advance to t=22 → "d" expired
        _advance(10.0)
        assert cache.size() == 0
```

**How it works — design summary:**

| Component | Purpose |
|---|---|
| `_Node` (doubly-linked list) | Stores `key`, `value`, absolute `expiry`, and `prev`/`next` pointers. Head-side = MRU, tail-side = LRU. |
| `_map: dict[str, _Node]` | O(1) key → node lookup. |
| Sentinel head/tail nodes | Eliminate null-check edge cases in link/unlink. |
| `time.monotonic()` | Monotonic clock; never goes backwards, immune to system clock changes. |
| Lazy expiry | Expired nodes are removed only when touched (`get`, `put` eviction walk, `size` scan). No background thread. |
| `_evict()` | Walks from LRU end; purges expired nodes it passes, then removes the first *valid* node. If *everything* is expired it clears the cache in one shot. |

All `get`/`put`/`delete` operations are **O(1)** average (hash lookup + constant-time node unlink/insert). `size()` is a linear scan but is the only place where every expired item must be accounted for, matching the "lazy cleanup" contract.