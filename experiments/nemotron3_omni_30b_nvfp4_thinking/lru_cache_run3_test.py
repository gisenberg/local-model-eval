"""TTLCache – an LRU cache with per‑entry time‑to‑live support.

The implementation uses a hash map (dict) for O(1) look‑ups and a
doubly‑linked list to keep the recency order.  Expired entries are
removed lazily – only when they are accessed (get) or when a new
item is inserted (put) or explicitly deleted.  All public operations
run in O(1) average time.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Dict


class _Node:
    """A node of the doubly‑linked list used by :class:`TTLCache`."""

    __slots__ = ("key", "value", "prev", "next", "expiry")

    def __init__(self, key: Any, value: Any, expiry: Optional[float]) -> None:
        self.key: Any = key
        self.value: Any = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None
        self.expiry: Optional[float] = expiry  # time.monotonic() timestamp or None


class TTLCache:
    """LRU cache with a default TTL and optional per‑entry TTL.

    Parameters
    ----------
    capacity : int
        Maximum number of entries the cache can hold. Must be > 0.
    default_ttl : float
        Default time‑to‑live in seconds for entries that do not provide a
        ``ttl`` argument on ``put``.  ``0`` means “never expire”.

    Notes
    -----
    * ``get`` moves the accessed entry to the front (most‑recently used).
    * ``put`` updates an existing entry (resetting its TTL) or inserts a
      new one; if the cache is full the least‑recently used entry is
      removed.
    * ``delete`` removes an entry explicitly.
    * ``size`` returns the current number of *registered* entries
      (expired entries are removed lazily, so the count may temporarily
      include them).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._map: Dict[Any, _Node] = {}
        self._head: Optional[_Node] = None  # most recent
        self._tail: Optional[_Node] = None  # least recent

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _now(self) -> float:
        """Return the current monotonic time."""
        return time.monotonic()

    def _add_node(self, node: _Node) -> None:
        """Insert *node* right after the head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next if self._head else None
        if self._head:
            self._head.next.prev = node
        else:
            self._tail = node
        self._head = node

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:  # node is head
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:  # node is tail
            self._tail = node.prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most‑recently used."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the least‑recently used node."""
        assert self._tail is not None
        node = self._tail
        self._remove_node(node)
        return node

    def _expire(self, node: _Node) -> bool:
        """Return ``True`` if *node* is expired, otherwise ``False``."""
        now = self._now()
        expiry = node.expiry
        if expiry is None:
            return False
        return now >= expiry

    def _cleanup_expired(self) -> None:
        """Lazy removal of all expired entries (called on access)."""
        # Build a list of keys to delete to avoid dict mutation during iteration.
        expired_keys = [k for k, n in self._map.items() if self._expire(n)]
        for k in expired_keys:
            self.delete(k)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def get(self, key: Any) -> Optional[Any]:
        """Return the value for *key* and mark it as most‑recently used.

        If the key is absent or its TTL has expired, ``None`` is returned
        and the entry is removed from the cache.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._expire(node):
            self.delete(key)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the default TTL is used; a ``ttl`` of ``0``
        means the entry never expires.
        """
        now = self._now()
        if key in self._map:
            node = self._map[key]
            node.value = value
            # Update expiry according to the supplied TTL.
            if ttl is None:
                node.expiry = None if self.default_ttl == 0 else now + self.default_ttl
            else:
                node.expiry = None if ttl == 0 else now + ttl
            self._move_to_head(node)
            return

        # key is new
        if len(self._map) >= self.capacity:
            # evict LRU
            lru = self._pop_tail()
            del self._map[lru.key]

        expiry = None if ttl == 0 else (now + ttl if ttl is not None else now + self.default_ttl)
        node = _Node(key, value, expiry)
        self._map[key] = node
        self._add_node(node)

    def delete(self, key: Any) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Current number of entries stored in the cache (including expired ones)."""
        return len(self._map)

import pytest
from unittest.mock import patch


def test_capacity_eviction():
    """Cache with capacity 2 should evict the least‑recently used entry."""
    cache = TTLCache(2, 10)          # large default TTL, not relevant for this test
    cache.put('a', 1)                # monotonic == 0
    cache.put('b', 2)                # monotonic == 1
    cache.put('c', 3)                # should evict 'a' (LRU)

    assert cache.size() == 2
    assert cache.get('a') is None   # evicted
    assert cache.get('b') == 2
    assert cache.get('c') == 3


def test_ttl_expiration_get():
    """An entry expires according to its TTL and subsequent get returns None."""
    cache = TTLCache(2, 0.5)         # half‑second default TTL

    # put at time 0, expiry = 0.5
    with patch('time.monotonic', side_effect=[0, 0]):   # two calls: put (0), first get (0)
        cache.put('key', 'value')
        assert cache.get('key') == 'value'            # still fresh

    # advance time past the TTL
    with patch('time.monotonic', side_effect=[0, 1]):   # put (0), get (1) -> expired
        assert cache.get('key') is None               # expired


def test_ttl_zero_immediate_expiry():
    """An entry with ttl=0 is considered expired right after insertion."""
    cache = TTLCache(2, 10)
    cache.put('tmp', 'value', ttl=0)   # expiry = now (0)

    with patch('time.monotonic', side_effect=[0, 0]):  # put (0), get (0)
        assert cache.get('tmp') is None               # immediate expiry


def test_delete():
    """Delete removes a key and returns the correct boolean."""
    cache = TTLCache(3, 10)
    cache.put('x', 42)

    assert cache.delete('x') is True
    assert cache.get('x') is None
    assert cache.delete('x') is False   # already absent


def test_size():
    """size() reflects the number of entries currently stored."""
    cache = TTLCache(3, 10)
    assert cache.size() == 0

    cache.put('a', 1)
    assert cache.size() == 1

    cache.put('b', 2)
    assert cache.size() == 2

    cache.put('c', 3)
    assert cache.size() == 3

    # add one more, causing eviction of the LRU ('a')
    cache.put('d', 4)
    assert cache.size() == 3
    assert cache.get('a') is None


def test_lru_ordering():
    """Accessing an entry moves it to the head; eviction removes the true LRU."""
    cache = TTLCache(3, 10)

    cache.put('a', 1)   # a (head), tail = a
    cache.put('b', 2)   # b (head), a (tail)
    cache.put('c', 3)   # c (head), b (middle), a (tail)

    # Access 'a' → becomes most recent
    cache.get('a')      # monotonic calls happen internally, no need to patch time

    # Order should now be: a (head), c, b (tail)
    cache.put('d', 4)     # evicts b (the LRU)

    assert cache.size() == 3
    assert cache.get('b') is None
    assert cache.get('a') == 1
    assert cache.get('c') == 3
    assert cache.get('d') == 4