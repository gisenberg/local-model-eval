"""TTLCache – an LRU cache whose entries expire after a given time‑to‑live.

The implementation uses a hash‑map (dict) for O(1) look‑ups and a
doubly‑linked list to keep the recency order.  All public operations
(`get`, `put`, `delete`, `size`) run in O(1) average time.

Only the standard library is required.
"""

import time
from typing import Any, Dict, Optional


class _Node:
    """A node of the doubly‑linked list used by :class:`TTLCache`.

    Attributes
    ----------
    key: Any
        The cache key.
    value: Any
        The cached value.
    expiry: float
        Monotonic timestamp (seconds since epoch) when the entry expires.
    prev: _Node | None
        Link to the previous node.
    next: _Node | None
        Link to the next node.
    """

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with per‑item TTL.

    Parameters
    ----------
    capacity : int
        Maximum number of items that can be stored simultaneously.
        ``0`` means the cache cannot hold any element.
    default_ttl : float
        Default time‑to‑live in seconds if ``ttl`` is not supplied to :meth>`put`.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl

        # hash map: key → node
        self._map: Dict[Any, _Node] = {}

        # dummy head & tail simplify edge handling
        self._head = _Node(None, None)   # most recent
        self._tail = _Node(None, None)   # least recent
        self._head.next = self._tail
        self._tail.prev = self._head

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Optional[Any]:
        """Return the value for *key* if it exists and is not expired.

        If the entry has expired, it is removed lazily and ``None`` is returned.
        The accessed entry becomes the most‑recently used.

        Parameters
        ----------
        key: Any
            Key whose value should be retrieved.

        Returns
        -------
        Optional[Any]
            The cached value, or ``None`` if the key is missing or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None

        now = time.monotonic()
        if node.expiry <= now:
            # expired – remove lazily
            self._remove_node(node)
            del self._map[key]
            return None

        # still valid → move to head (most recent) and return value
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the instance’s ``default_ttl`` is used.
        The entry becomes the most‑recently used.  When the cache exceeds
        its capacity the least‑recently used entry is evicted.

        Parameters
        ----------
        key: Any
            Key to store.
        value: Any
            Value to associate with *key*.
        ttl: Optional[float]
            Expiration time‑to‑live in seconds.  If omitted, ``default_ttl`` is used.
        """
        now = time.monotonic()
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry = now + ttl_to_use

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
        else:
            node = _Node(key, value, expiry)
            self._add_to_head(node)
            self._map[key] = node

            # enforce capacity – evict LRU while we are over the limit
            while len(self._map) > self.capacity:
                self._evict_lru()

    def delete(self, key: Any) -> None:
        """Remove *key* from the cache, if it exists."""
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """Return the number of items currently stored in the cache.

        The count reflects the number of entries present in the internal map,
        regardless of whether some of them may have expired (they are removed
        lazily on the next access).
        """
        return len(self._map)

    # --------------------------------------------------------------------- #
    # Internal helpers (all O(1))
    # --------------------------------------------------------------------- #
    def _remove_node(self, node: _Node) -> None:
        """Unlink *node* from the doubly‑linked list."""
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
        node.prev = node.next = None  # help GC

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* right after the dummy head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as the most‑recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove the least‑recently used entry (the node just before the dummy tail)."""
        lru = self._tail.prev
        if lru is not self._head:
            self._remove_node(lru)
            del self._map[lru.key]

"""Pytest suite for :class:`TTLCache`.

All tests use ``unittest.mock.patch`` to replace ``time.monotonic`` with a
controlled value, allowing deterministic testing of TTL‑based behaviour.
"""

import pytest
from unittest.mock import patch

# Import the class under test.  Adjust the import path if you place the
# implementation in a different module.


class MockTime:
    """Simple mutable container that mimics ``time.monotonic``."""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_basic_get_put():
    """Get works and respects the default TTL."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(3, 10)          # 10‑second default TTL
        cache.put('x', 42)               # expires at now+10

        tm.now = 5
        assert cache.get('x') == 42

        tm.now = 15
        assert cache.get('x') is None    # expired


def test_custom_ttl():
    """Custom TTL supplied to ``put`` overrides the default."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(5, 100)         # default 100 s
        cache.put('y', 99, ttl=5)        # expires at now+5

        tm.now = 2
        assert cache.get('y') == 99

        tm.now = 7
        assert cache.get('y') is None    # expired


def test_capacity_eviction():
    """When capacity is exceeded, the least‑recently used entry is removed."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(2, 1000)        # large TTL, capacity 2
        cache.put('a', 1)
        cache.put('b', 2)

        tm.now = 1
        cache.put('c', 3)                  # 'a' is LRU → evicted

        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3


def test_size():
    """``size`` reflects the number of entries currently stored."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(3, 10)
        assert cache.size() == 0

        cache.put('k1', 1)
        cache.put('k2', 2)
        assert cache.size() == 2

        cache.put('k3', 3)
        assert cache.size() == 3

        # Adding a fourth item evicts the LRU entry, size stays 3
        cache.put('k4', 4)
        assert cache.size() == 3


def test_delete():
    """``delete`` removes a key and updates the size."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(5, 10)
        cache.put('z', 5)
        assert cache.size() == 1

        cache.delete('z')
        assert cache.get('z') is None
        assert cache.size() == 0


def test_lru_with_access():
    """Accessing items changes the LRU order; eviction follows that order."""
    tm = MockTime()
    with patch('time.monotonic', tm):
        cache = TTLCache(3, 100)        # capacity 3
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)

        tm.now = 1
        cache.get('a')   # a becomes most recent, order: a, b, c
        tm.now = 2
        cache.get('b')   # b becomes most recent, order: b, a, c

        cache.put('d', 4)  # c is LRU → evicted
        assert cache.get('c') is None
        assert cache.get('b') == 2
        assert cache.get('a') == 1
        assert cache.get('d') == 4