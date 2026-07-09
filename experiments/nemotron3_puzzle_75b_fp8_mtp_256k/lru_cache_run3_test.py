"""
TTLCache – an LRU cache with optional per‑item TTL.

The implementation uses a hash map (dictionary) for O(1) look‑ups and a
doubly‑linked list to keep track of usage order.  Expired entries are
removed lazily – they are cleaned up only when they are encountered during
get, put, delete or when making room for a new item.

Time is obtained via ``time.monotonic()`` so the cache is immune to
system‑clock changes.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Generic, TypeVar

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type


class _Node(Generic[K, V]):
    """
    A node of the doubly‑linked list used by TTLCache.

    Attributes
    ----------
    key: K
        The key stored in the cache.
    value: V
        The associated value.
    expiry: float | None
        Absolute timestamp (from ``time.monotonic``) when the entry expires.
        ``None`` means the entry never expires.
    prev: _Node | None
        Previous node in the list (more recent).
    next: _Node | None
        Next node in the list (older).
    """

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(
        self,
        key: K,
        value: V,
        expiry: Optional[float],
    ) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node[K, V]] = None
        self.next: Optional[_Node[K, V]] = None

    def is_expired(self, now: float) -> bool:
        """Return True if the node has expired relative to ``now``."""
        return self.expiry is not None and now >= self.expiry


class TTLCache(Generic[K, V]):
    """
    LRU cache with optional TTL.

    Parameters
    ----------
    capacity:
        Maximum number of items the cache can hold. Must be > 0.
    default_ttl:
        Default time‑to‑live in seconds for items inserted without an explicit
        TTL. If ``None``, items never expire unless a custom TTL is supplied.
    """

    def __init__(self, capacity: int, default_ttl: Optional[float] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity: int = capacity
        self.default_ttl: Optional[float] = default_ttl

        self._map: Dict[K, _Node[K, V]] = {}

        # Dummy head and tail nodes to simplify list operations
        self._head: _Node[K, V] = _Node(None, None, None)  # type: ignore[arg-type]
        self._tail: _Node[K, V] = _Node(None, None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------ #
    # Internal helpers for list manipulation
    # ------------------------------------------------------------------ #
    def _remove(self, node: _Node[K, V]) -> None:
        """Detach ``node`` from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = node.next = None

    def _add_to_head(self, node: _Node[K, V]) -> None:
        """Insert ``node`` right after the dummy head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next is not None:
            self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node[K, V]) -> None:
        """Move an existing node to the most‑recent position."""
        self._remove(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node[K, V]:
        """Remove and return the least‑recently used node (before dummy tail)."""
        node = self._tail.prev
        if node is None or node is self._head:
            raise RuntimeError("Cache is empty")
        self._remove(node)
        return node

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key‑value pair.

        If the key already exists, its value and TTL are refreshed and the
        node becomes most‑recently used.

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to store.
        ttl:
            Time‑to‑live in seconds. If ``None``, ``default_ttl`` from the
            constructor is used. If both are ``None`` the item never expires.
        """
        now = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl
        expiry: Optional[float] = None if effective_ttl is None else now + effective_ttl

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        # Need to make room if at capacity
        if len(self._map) >= self.capacity:
            while len(self._map) >= self.capacity:
                lru = self._tail.prev
                # lru will never be the head because we check size >= capacity > 0
                if lru.is_expired(now):
                    self._delete_node(lru)
                else:
                    # Evict this LRU entry
                    self._delete_node(lru)
                    break

        node = _Node(key, value, expiry)
        self._map[key] = node
        self._add_to_head(node)

    def get(self, key: K) -> Optional[V]:
        """
        Return the value associated with ``key`` if present and not expired.

        Parameters
        ----------
        key:
            Cache key to look up.

        Returns
        -------
        The cached value, or ``None`` if the key is missing or the entry has
        expired (in which case the expired entry is removed).
        """
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return None

        if node.is_expired(now):
            self._delete_node(node)
            return None

        self._move_to_head(node)
        return node.value

    def delete(self, key: K) -> bool:
        """
        Remove ``key`` from the cache.

        Returns
        -------
        True if the key was present and removed, False otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove(node)
        return True

    def size(self) -> int:
        """Current number of items stored (may include expired entries that
        have not yet been lazily removed)."""
        return len(self._map)

    # ------------------------------------------------------------------ #
    # Internal helper used by put/delete when we know the node is being
    # discarded.
    # ------------------------------------------------------------------ #
    def _delete_node(self, node: _Node[K, V]) -> None:
        """Unlink ``node`` from the list and delete it from the map."""
        del self._map[node.key]
        self._remove(node)

"""
Six pytest tests for TTLCache using unittest.mock.patch to control
time.monotonic().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Assuming the implementation lives in ttlcache.py in the same directory or
# installed as a package.


def test_basic_put_get():
    """Insert a value and retrieve it; size reflects the entry."""
    cache = TTLCache[int, str](capacity=2)
    cache.put(1, "one")
    assert cache.get(1) == "one"
    assert cache.size() == 1


def test_lru_eviction():
    """When capacity is exceeded, the least‑recently used item is removed."""
    cache = TTLCache[int, int](capacity=2)
    cache.put(1, 10)   # MRU
    cache.put(2, 20)   # MRU
    cache.put(3, 30)   # Should evict key 1

    assert cache.get(1) is None   # evicted
    assert cache.get(2) == 20     # still present (LRU now)
    assert cache.get(3) == 30     # MRU
    assert cache.size() == 2


def test_ttl_expiration():
    """An entry expires after its TTL and is removed on access."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache[int, str](capacity=5, default_ttl=None)
        cache.put(1, "hello", ttl=0.5)  # expires at 1000.5
        assert cache.get(1) == "hello"
        assert cache.size() == 1

        # Advance time just before expiry
        mock_time.return_value = 1000.4
        assert cache.get(1) == "hello"
        assert cache.size() == 1

        # Advance past expiry
        mock_time.return_value = 1000.6
        assert cache.get(1) is None   # expired, removed
        assert cache.size() == 0


def test_default_ttl():
    """Items inserted without an explicit TTL use the constructor's default."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 2000.0
        cache = TTLCache[int, int](capacity=3, default_ttl=0.2)
        cache.put(1, 42)               # expires at 2000.2
        assert cache.get(1) == 42
        assert cache.size() == 1

        mock_time.return_value = 2000.1
        assert cache.get(1) == 42
        assert cache.size() == 1

        mock_time.return_value = 2000.25
        assert cache.get(1) is None    # expired
        assert cache.size() == 0


def test_update_resets_ttl_and_order():
    """Updating an existing key refreshes its TTL and makes it MRU."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 3000.0
        cache = TTLCache[int, str](capacity=2, default_ttl=0.4)

        # Insert A (TTL 0.4) then B (TTL 0.4)
        cache.put("A", "a")   # expires 3000.4
        cache.put("B", "b")   # expires 3000.4
        # Order: B (MRU), A (LRU)

        # Update A with a new TTL of 0.1 (expires sooner) – should become MRU
        mock_time.return_value = 3000.2
        cache.put("A", "a2", ttl=0.1)   # expires 3000.3
        # Order: A (MRU), B (LRU)

        # Insert C (TTL 0.4) – should evict B (LRU) because A is MRU
        mock_time.return_value = 3000.25
        cache.put("C", "c")   # expires 3000.65
        # Order: C (MRU), A (LRU)

        assert cache.get("B") is None   # evicted
        assert cache.get("A") == "a2"
        assert cache.get("C") == "c"
        assert cache.size() == 2


def test_delete_and_size():
    """Deleting a key reduces size and returns False for missing keys."""
    cache = TTLCache[int, int](capacity=3)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.size() == 2

    assert cache.delete(2) is True
    assert cache.get(2) is None
    assert cache.size() == 1

    assert cache.delete(99) is False   # key not present
    assert cache.size() == 1