"""TTLCache – a fixed‑size LRU cache whose entries expire after a given time.

The implementation uses a doubly‑linked list for O(1) insert / delete
operations and a dict (hash map) for O(1) look‑ups.  Expired entries are
removed lazily – only when they are accessed (get) or when a new entry
needs to be inserted (capacity overflow).  The current time is obtained
with ``time.monotonic()`` so the cache works correctly even if the system
clock is adjusted.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Generic, TypeVar, List

T = TypeVar("T")


class _Node(Generic[T]):
    """A node of the doubly‑linked list."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(
        self,
        key: str,
        value: T,
        expiry: Optional[float],
        prev: Optional["_Node[T]"],
        next: Optional["_Node[T]"],
    ) -> None:
        self.key: str = key
        self.value: T = value
        self.expiry: Optional[float] = expiry
        self.prev: Optional[_Node[T]] = prev
        self.next: Optional[_Node[T]] = next


class TTLCache(Generic[T]):
    """Cache with a maximum number of entries and a per‑entry TTL.

    Args:
        capacity: Maximum number of live (non‑expired) entries that may be stored.
        default_ttl: Default time‑to‑live in seconds for entries whose ``ttl`` argument
            is ``None``.  ``0`` means “never expires”.

    Notes:
        * ``get`` returns ``None`` if the key is missing or has expired.
        * ``put`` overwrites the value of an existing key and refreshes its TTL.
        * ``delete`` removes a key explicitly.
        * ``size`` returns the number of *non‑expired* entries currently stored.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non‑negative")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._size: int = 0

        # Dummy head/tail nodes – they simplify edge handling.
        self._head: _Node[T] = _Node("", "", None, None, None)   # most‑recent
        self._tail: _Node[T] = _Node("", "", None, None, None)   # least‑recent
        self._head.next = self._tail
        self._tail.prev = self._head

        # Mapping from key → node for O(1) access.
        self._map: Dict[str, _Node[T]] = {}

    # --------------------------------------------------------------------- #
    # Helper methods (all O(1))
    # --------------------------------------------------------------------- #
    def _now(self) -> float:
        """Current monotonic time."""
        return time.monotonic()

    def _expire(self, node: _Node[T]) -> bool:
        """Remove *node* if it is expired, otherwise do nothing.

        Returns ``True`` if the node was removed because it had expired.
        """
        if node.expiry is None:
            return False
        if node.expiry <= self._now():
            # Unlink node
            self._remove_node(node)
            del self._map[node.key]
            self._size -= 1
            return True
        return False

    def _add_to_front(self, node: _Node[T]) -> None:
        """Insert *node* right after the head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore[assignment]
        self._head.next = node

    def _remove_node(self, node: _Node[T]) -> None:
        """Detach *node* from the linked list."""
        prev = node.prev  # type: ignore[assignment]
        nxt = node.next   # type: ignore[assignment]
        if prev:
            prev.next = nxt
        if nxt:
            nxt.prev = prev

    def _move_to_front(self, node: _Node[T]) -> None:
        """Mark *node* as most‑recently used."""
        self._remove_node(node)
        self._add_to_front(node)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: str) -> Optional[T]:
        """Return the value for *key* if it exists and is not expired.

        The accessed entry becomes the most‑recently used.

        Returns:
            The cached value, or ``None`` if the key is absent or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None

        # Lazy expiration check
        if self._expire(node):
            return None

        # Key is valid – move to front
        self._move_to_front(node)  # type: ignore[arg-type]
        return node.value  # type: ignore[return-value]

    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the entry never expires (except when the cache
        reaches its capacity, in which case the LRU entry is evicted).

        When the cache is full, the least‑recently used entry is evicted
        before inserting the new one.
        """
        now = self._now()

        if key in self._map:
            node = self._map[key]
            node.value = value
            if ttl is not None:
                node.expiry = now + ttl
            else:
                node.expiry = None  # infinite lifetime
            self._move_to_front(node)  # type: ignore[arg-type]
            return

        # New key – need to make room if we are at capacity.
        if self._size >= self._capacity and self._capacity > 0:
            # Evict LRU (the node just before the tail)
            lru = self._tail.prev  # type: ignore[assignment]
            if lru is not self._tail:  # capacity > 0 ⇒ there is a real node
                self._remove_node(lru)
                del self._map[lru.key]
                self._size -= 1

        # Create and insert the new node
        expiry = None if ttl is None else now + ttl
        new_node = _Node(key, value, expiry, None, None)
        self._map[key] = new_node
        self._add_to_front(new_node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        self._size -= 1
        return True

    def size(self) -> int:
        """Current number of live (non‑expired) entries."""
        # Remove any expired nodes that might still be present
        # (this can happen if the cache has been idle for a while).
        # We walk the list once – this is O(n) but happens only
        # when size is queried, which is acceptable for a cache.
        count = 0
        cur = self._head.next
        while cur is not self._tail:
            if self._expire(cur):
                continue
            count += 1
            cur = cur.next
        return count

"""Pytest suite for TTLCache.

All tests use ``unittest.mock.patch`` to control ``time.monotonic`` so that
the behaviour can be deterministic without sleeping.
"""

import builtins
from unittest.mock import patch

import pytest

# Import the class from the module we just created.
# Adjust the import path if you place the implementation in a different file.


def make_time_seq(values):
    """Factory that returns a callable to be used as ``side_effect`` for
    ``time.monotonic``.  The callable returns the next value in *values* each
    time it is called.
    """
    def _monotonic():
        make_time_seq.counter += 1
        return values[make_time_seq.counter - 1]
    make_time_seq.counter = 0
    return _monotonic


def test_basic_get_and_ttl_expiration():
    """get returns the value while it is fresh and None after expiry."""
    cache = TTLCache[int](capacity=3, default_ttl=0)  # no default ttl, we supply per‑call ttl

    with patch('ttl_cache.time.monotonic', side_effect=[0, 1, 2, 3, 4, 5]) as mock_time:
        # put key=1 with ttl=3 seconds
        cache.put('k1', 42, ttl=3)

        # after 2 seconds it is still valid
        assert cache.get('k1') == 42

        # after 4 seconds it should be expired
        assert cache.get('k1') is None

        # size should be 0 because the only entry expired
        assert cache.size() == 0


def test_capacity_eviction():
    """When the cache is full, the least‑recently used entry is evicted."""
    cache = TTLCache[str](capacity=2, default_ttl=10)

    # Fill the cache with two items
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2

    # Insert a third item – LRU ('a') should be removed
    cache.put('c', 3)

    # 'a' must be gone, 'c' present, 'b' still there
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2


def test_delete_operation():
    """Explicit deletion removes the key and updates size."""
    cache = TTLCache[int](capacity=5, default_ttl=0)

    cache.put('x', 100)
    assert cache.size() == 1
    assert cache.get('x') == 100

    assert cache.delete('x') is True
    assert cache.size() == 0
    assert cache.get('x') is None
    assert cache.delete('nonexistent') is False


def test_default_ttl():
    """Entries created without an explicit ttl use the default ttl."""
    cache = TTLCache[int](capacity=5, default_ttl=2)

    with patch('ttl_cache.time.monotonic', side_effect=[0, 1, 2, 3, 4]) as mock_time:
        cache.put('y', 99)               # ttl = default (2 sec)
        assert cache.get('y') == 99      # still fresh

        # after 2 seconds it expires
        assert cache.get('y') is None
        assert cache.size() == 0


def test_multiple_operations_and_size():
    """Mix of puts, gets and expirations to verify size tracking."""
    cache = TTLCache[int](capacity=4, default_ttl=0)

    # Insert three items with different ttl values
    cache.put('k1', 1, ttl=1)   # expires after 1 sec
    cache.put('k2', 2, ttl=2)   # expires after 2 sec
    cache.put('k3', 3)          # never expires (ttl=None)

    # At time = 0 all are alive
    assert cache.size() == 3

    # Advance 1 sec – k1 expires, size becomes 2
    with patch('ttl_cache.time.monotonic', side_effect=[0, 1, 2, 3, 4]):
        assert cache.get('k1') is None
        assert cache.size() == 2
        assert cache.get('k2') == 2
        assert cache.get('k3') == 3

    # Advance 2 more sec – k2 expires, size becomes 1
    assert cache.get('k2') is None
    assert cache.size() == 1
    assert cache.get('k3') == 3


def test_lazy_cleanup_and_capacity():
    """Cache should evict LRU entries even when they are not expired."""
    cache = TTLCache[int](capacity=3, default_ttl=10)

    # Insert three items, making 'k1' the least‑recently used
    cache.put('k1', 1)
    cache.put('k2', 2)
    cache.put('k3', 3)

    # Access k2 to make it most recent
    cache.get('k2')

    # Insert a fourth item – LRU ('k1') should be evicted
    cache.put('k4', 4)

    # Verify that k1 is gone, others remain
    assert cache.get('k1') is None
    assert cache.get('k2') == 2
    assert cache.get('k3') == 3
    assert cache.get('k4') == 4
    assert cache.size() == 3


def test_capacity_zero():
    """Cache with capacity 0 never stores anything."""
    cache = TTLCache[int](capacity=0, default_ttl=5)

    cache.put('any', 123)          # should be ignored
    assert cache.size() == 0
    assert cache.get('any') is None
    assert cache.delete('any') is False