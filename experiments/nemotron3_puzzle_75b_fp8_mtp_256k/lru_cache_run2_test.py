# ttl_cache.py
from __future__ import annotations

import time
from typing import Optional, Dict


class _Node:
    """
    Internal doubly‑linked list node used by TTLCache.
    """

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(
        self,
        key: int,
        value: object,
        expiry: float,
        prev: Optional[_Node] = None,
        next: Optional[_Node] = None,
    ) -> None:
        self.key: int = key
        self.value: object = value
        self.expiry: float = expiry
        self.prev: Optional[_Node] = prev
        self.next: Optional[_Node] = next


class TTLCache:
    """
    LRU cache with optional per‑item TTL (time‑to‑live).

    Parameters
    ----------
    capacity : int
        Maximum number of items the cache can hold. Must be > 0.
    default_ttl : float
        TTL (in seconds) used when ``put`` is called without an explicit ``ttl``.
        Must be >= 0. A value of 0 means the item expires immediately.

    The cache provides *O(1)* average time complexity for ``get``, ``put`` and
    ``delete`` by combining a hash map (key → node) with a doubly‑linked list
    that tracks usage order. Expired entries are removed lazily – only when
    they are accessed or when the cache needs to make room for a new entry.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if default_ttl < 0:
            raise ValueError("default_ttl must be >= 0")

        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._now = time.monotonic  # allow easy monkeypatching in tests

        # Hash map for O(1) look‑up
        self._cache: Dict[int, _Node] = {}

        # Sentinel nodes for the doubly‑linked list
        self._head = _Node(key=-1, value=None, expiry=0.0)  # most‑recent
        self._tail = _Node(key=-1, value=None, expiry=0.0)  # least‑recent
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------ #
    # Internal linked‑list helpers
    # ------------------------------------------------------------------ #
    def _add_to_front(self, node: _Node) -> None:
        """Insert ``node`` right after the head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Detach ``node`` from the list."""
        prev, nxt = node.prev, node.next
        if prev:
            prev.next = nxt
        if nxt:
            nxt.prev = prev
        node.prev = node.next = None

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the most‑recent position."""
        self._remove_node(node)
        self._add_to_front(node)

    def _pop_lru(self) -> _Node:
        """
        Remove and return the least‑recently used node (the one right before tail).
        Assumes the list is not empty (i.e. size > 0).
        """
        lru = self._tail.prev
        assert lru is not None and lru is not self._head
        self._remove_node(lru)
        return lru

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get(self, key: int) -> Optional[object]:
        """
        Return the value associated with ``key`` if present and not expired.
        Otherwise return ``None`` and remove the expired entry (if it existed).

        The accessed node is moved to the most‑recent position (LRU update).
        """
        node = self._cache.get(key)
        if node is None:
            return None

        # Lazy expiration check
        if self._now() >= node.expiry:
            self._remove_node(node)
            del self._cache[key]
            return None

        # Not expired – promote to MRU
        self._move_to_front(node)
        return node.value

    def put(
        self, key: int, value: object, ttl: Optional[float] = None
    ) -> None:
        """
        Insert or update ``key`` with ``value``.

        If ``ttl`` is None, ``default_ttl`` is used. A non‑positive TTL results
        in an immediate expiration (the entry is not stored).

        If the cache is at capacity, the least‑recently used entry is evicted
        before inserting the new one.
        """
        # Resolve TTL
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl <= 0:
            # Treat as expired – do not store anything
            self.delete(key)  # ensure any old entry is removed
            return

        expiry = self._now() + effective_ttl

        if key in self._cache:
            # Update existing node
            node = self._cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # New entry – possibly evict LRU
        if len(self._cache) >= self._capacity:
            lru_node = self._pop_lru()
            del self._cache[lru_node.key]

        # Insert new node
        new_node = _Node(key=key, value=value, expiry=expiry)
        self._cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: int) -> None:
        """
        Remove ``key`` from the cache if it exists.
        """
        node = self._cache.pop(key, None)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """Current number of items stored in the cache."""
        return len(self._cache)

# test_ttl_cache.py
import time
from unittest.mock import patch

import pytest



@pytest.fixture
def cache():
    """A fresh TTLCache with capacity 3 and default TTL 2 seconds."""
    return TTLCache(capacity=3, default_ttl=2.0)


def test_basic_put_and_get(cache):
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1


def test_ttl_expiration(cache):
    with patch.object(time, "monotonic", side_effect=[0.0, 1.5, 3.0]):
        cache.put("b", "hello", ttl=2.0)   # expires at 2.0
        # First get – still valid
        assert cache.get("b") == "hello"
        # Second get – after expiry
        assert cache.get("b") is None
        assert cache.size() == 0


def test_lru_eviction(cache):
    # Fill cache
    cache.put(1, "one")
    cache.put(2, "two")
    cache.put(3, "three")
    assert cache.size() == 3

    # Access 1 to make it MRU
    cache.get(1)

    # Insert a fourth item – should evict the LRU (2)
    cache.put(4, "four")
    assert cache.get(2) is None          # evicted
    assert cache.get(1) == "one"         # still present (MRU)
    assert cache.get(3) == "three"
    assert cache.get(4) == "four"
    assert cache.size() == 3


def test_update_existing_key_resets_ttl_and_mru(cache):
    with patch.object(time, "monotonic", side_effect=[0.0, 1.0, 3.0]):
        cache.put("x", 100, ttl=2.0)   # expires at 2.0
        # Wait almost until expiry
        assert cache.get("x") == 100   # still valid at 1.0

        # Update with new TTL (reset expiry to 5.0)
        cache.put("x", 200, ttl=2.0)   # now expires at 5.0 (current mock time 3.0 + 2)
        assert cache.get("x") == 200   # fresh value
        # Fast‑forward past original expiry but before new expiry
        with patch.object(time, "monotonic", return_value=4.0):
            assert cache.get("x") == 200   # still valid
        # Fast‑forward past new expiry
        with patch.object(time, "monotonic", return_value=6.0):
            assert cache.get("x") is None   # now expired
        assert cache.size() == 0


def test_delete_key(cache):
    cache.put("k1", 1)
    cache.put("k2", 2)
    assert cache.size() == 2

    cache.delete("k1")
    assert cache.get("k1") is None
    assert cache.get("k2") == 2
    assert cache.size() == 1


def test_lazy_cleanup_on_put(cache):
    """
    When the cache is full, the LRU entry is evicted even if it is already
    expired (lazy cleanup). This test verifies that an expired LRU entry is
    removed without scanning the whole table.
    """
    with patch.object(time, "monotonic") as mock_time:
        # t0 = 0.0
        mock_time.return_value = 0.0
        cache.put("a", 1, ttl=1.0)   # expires at 1.0
        cache.put("b", 2, ttl=10.0)  # expires at 10.0
        # Cache now: head <-> b <-> a <-> tail   (a is LRU)

        # Advance time so that 'a' expires but we do not access it yet
        mock_time.return_value = 2.0   # a expired, b still valid

        # Insert a new item; cache is at capacity, so LRU (a) is evicted
        cache.put("c", 3, ttl=10.0)   # expires at 12.0

        # After insertion we expect: b and c present, a gone
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2