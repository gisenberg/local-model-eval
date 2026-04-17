Looking at the requirements carefully, I need to:

1. Build a doubly-linked list with sentinel nodes + hash map for O(1) operations
2. Handle lazy cleanup of expired entries (only removed on access)
3. Eviction must skip expired items and evict the LRU *non-expired* item; if all expired, clear all
4. Mock `time.monotonic` for deterministic tests

Let me walk through the invariants:
- **LRU order**: head = MRU, tail = LRU; moving accessed items to head
- **Capacity**: `size()` (non-expired count) must never exceed `capacity` after a `put`
- **Lazy cleanup**: expired nodes stay in the data structures until an operation touches them
- **Eviction**: walk from tail, skip expired, evict first non-expired; if none found, clear all

```python
"""TTLCache: LRU cache with time-based expiration using a doubly-linked list + hash map."""

import time
from typing import Any, Optional


class _Node:
    """Doubly-linked list node storing a cache entry."""
    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """Least-Recently-Used cache with per-entry time-to-live expiration.

    Uses a sentinel-based doubly-linked list (head=MRU, tail=LRU) paired with
    a hash map for O(1) average-time operations.  Expired entries are cleaned
    up lazily on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialise the cache.

        Args:
            capacity: Maximum number of *non-expired* items the cache may hold.
            default_ttl: Default time-to-live in seconds for entries without an
                         explicit TTL.
        """
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict[str, _Node] = {}
        # Sentinel nodes — never removed, never counted
        self._head = _Node("", None, float("inf"))
        self._tail = _Node("", None, float("inf"))
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size: int = 0  # number of entries in _map (non-expired + expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, node: _Node) -> bool:
        """Return True if *node* has passed its expiry time."""
        return time.monotonic() >= node.expiry

    def _remove(self, node: _Node) -> None:
        """Unlink *node* from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* immediately after the head sentinel (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Reposition *node* to the MRU position (head)."""
        self._remove(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Evict the least-recently-used **non-expired** item.

        Walks from the tail (LRU) toward the head, skipping expired entries.
        If every entry is expired, clears the entire cache.
        """
        current = self._tail.prev
        found_non_expired = False

        while current != self._head:
            if self._is_expired(current):
                current = current.prev
                continue
            # Found the LRU non-expired entry — evict it
            self._remove(current)
            del self._map[current.key]
            self._size -= 1
            found_non_expired = True
            break

        if not found_non_expired:
            self._clear_all()

    def _clear_all(self) -> None:
        """Remove every entry and reset the linked list."""
        self._map.clear()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for *key*, or ``None`` if absent or expired.

        Accessing a key promotes it to the most-recently-used position.
        Expired entries are removed lazily.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        if self._is_expired(node):
            self._remove(node)
            del self._map[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* → *value*.

        * If *key* already exists, its value and expiry are updated and it
          becomes MRU.
        * If the cache is at capacity, the LRU **non-expired** item is evicted.
          If every item is expired, the cache is cleared first.
        * A per-call *ttl* overrides ``self._default_ttl``.
        """
        effective_ttl: float = ttl if ttl is not None else self._default_ttl
        expiry: float = time.monotonic() + effective_ttl

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        if self._size >= self._capacity:
            self._evict_lru()

        node = _Node(key, value, expiry)
        self._map[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed (and was removed), ``False``
        otherwise.
        """
        if key not in self._map:
            return False

        node = self._map[key]
        self._remove(node)
        del self._map[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """Return the number of **non-expired** items.

        Performs lazy cleanup: expired entries encountered during the scan
        are removed from both the linked list and the hash map.
        """
        current = self._head.next
        count = 0
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove(current)
                del self._map[current.key]
                self._size -= 1
            else:
                count += 1
            current = next_node
        return count
```

Now the test suite — each test encodes a specific semantic contract:

```python
"""Tests for TTLCache — deterministic via mocked time.monotonic."""

import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def _mock_time(base: float = 0.0):
    """Return (mock_func, advance_func) for deterministic time control."""
    clock = [base]

    def mock_monotonic() -> float:
        return clock[0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    return mock_monotonic, advance


class TestTTLCache:
    """Six tests covering the core contracts of TTLCache."""

    # ------------------------------------------------------------------
    # 1. Basic get / put
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_basic_get_put(self, mock_monotonic):
        """A freshly inserted key is retrievable; missing keys return None."""
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)

        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None

    # ------------------------------------------------------------------
    # 2. Capacity eviction — LRU order
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_capacity_eviction_lru_order(self, mock_monotonic):
        """When at capacity, the *least-recently-used* item is evicted.

        Accessing a key promotes it to MRU, so the *other* item becomes LRU.
        """
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)

        cache.put("a", 1)
        cache.put("b", 2)

        # Access "a" → "a" becomes MRU, "b" becomes LRU
        cache.get("a")

        # Insert "c" → evicts "b" (LRU)
        cache.put("c", 3)

        assert cache.get("a") == 1   # still present (was accessed)
        assert cache.get("b") is None  # evicted (was LRU)
        assert cache.get("c") == 3

    # ------------------------------------------------------------------
    # 3. TTL expiry
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_ttl_expiry(self, mock_monotonic):
        """After the default TTL elapses, get() returns None."""
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=5.0)

        cache.put("a", 1)

        # Advance past the TTL
        mock_monotonic.return_value = 5.1
        assert cache.get("a") is None

    # ------------------------------------------------------------------
    # 4. Custom per-key TTL
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_custom_per_key_ttl(self, mock_monotonic):
        """A per-call TTL overrides the default; items expire independently."""
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)

        cache.put("short", 1, ttl=3.0)
        cache.put("long", 2, ttl=15.0)

        # "short" has expired, "long" has not
        mock_monotonic.return_value = 4.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

        # "long" also expires
        mock_monotonic.return_value = 16.0
        assert cache.get("long") is None

    # ------------------------------------------------------------------
    # 5. Delete
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_delete(self, mock_monotonic):
        """delete() returns True for existing keys, False otherwise, and
        the key is no longer retrievable afterward."""
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)

        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False  # already gone
        assert cache.get("a") is None

    # ------------------------------------------------------------------
    # 6. Size with mixed expired / valid items
    # ------------------------------------------------------------------
    @patch("ttl_cache.time.monotonic")
    def test_size_with_mixed_expired_valid(self, mock_monotonic):
        """size() counts only non-expired entries and lazily removes expired
        ones from the internal structures."""
        mock_monotonic.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=5.0)

        cache.put("a", 1)          # expires at 5
        cache.put("b", 2, ttl=10)  # expires at 10
        cache.put("c", 3)          # expires at 5

        # At t=6: "a" and "c" are expired, "b" is valid
        mock_monotonic.return_value = 6.0
        assert cache.size() == 1

        # After size() the expired entries should have been cleaned up
        assert cache.get("a") is None
        assert cache.get("c") is None
        assert cache.get("b") == 2
```

**Key design decisions and invariants preserved:**

| Requirement | How it's handled |
|---|---|
| O(1) get/put/delete | Hash map lookup + sentinel-linked-list insert/remove/move-to-head — all constant-time pointer manipulations |
| Lazy cleanup | Expired nodes stay in `_map` and the linked list until an operation (`get`, `put`, `delete`, `size`) encounters them |
| Eviction skips expired | `_evict_lru` walks from tail, `continue`s on expired nodes, evicts the first non-expired one |
| All-expired → clear all | If the walk reaches the head sentinel without finding a non-expired node, `_clear_all` is called |
| `size()` excludes expired | Iterates the list, removes expired nodes on the fly, counts only valid ones |
| `time.monotonic()` | Used exclusively for expiry computation; mocked in tests via `@patch("ttl_cache.time.monotonic")` |
| No `OrderedDict` | Custom `_Node` class with `prev`/`next` pointers and sentinel head/tail nodes |