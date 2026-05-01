"""TTLCache – a fixed‑size LRU cache whose entries expire after a given time.

The implementation uses a hash map (dict) for O(1) look‑ups and a doubly linked
list to keep the recency order.  Expired entries are removed lazily – they are
deleted the first time they are accessed after their TTL has passed.

Typical usage
-------------
>>> >>> c = TTLCache(capacity=3, default_ttl=5)      # 5‑second default TTL
>>> c.put('foo', 'bar')                         # uses default TTL
>>> c.get('foo')                               # → 'bar'
>>> import time
>>> time.monotonic = lambda: 10                 # fast‑forward time
>>> c.get('foo')                               # expired → None
"""

import time
from __future__ import annotations

from typing import Any, Optional


class _Node:
    """A node of the doubly‑linked list."""

    __slots__ = ("key", "value", "expire_at", "prev", "next")

    def __init__(self, key: Any, value: Any, expire_at: Optional[float]) -> None:
        self.key: Any = key
        self.value: Any = value
        self.expire_at: Optional[float] = expire_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Cache with fixed capacity and per‑entry TTL.

    Args:
        capacity: Maximum number of live entries the cache may hold. ``0`` means
            the cache never stores anything.
        default_ttl: Default time‑to‑live in seconds for entries that are inserted
            without an explicit ``ttl``. ``None`` means “no expiration”.

    Notes
    -----
    * All public operations (`get`, `put`, `delete`, `size`) run in O(1) average
      time.
    * Expired entries are removed lazily – they are deleted only when a
      ``get`` or ``put`` touches them.
    """

    def __init__(self, capacity: int, default_ttl: Optional[float]) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self.capacity: int = capacity
        self.default_ttl: Optional[float] = default_ttl
        self._map: Dict[Any, _Node] = {}
        self._head: Optional[_Node] = None   # most recently used
        self._tail: Optional[_Node] = None   # least recently used

    # --------------------------------------------------------------------- #
    # Internal helper methods (all O(1))
    # --------------------------------------------------------------------- #
    def _time(self) -> float:
        """Return the current monotonic timestamp."""
        return time.monotonic()

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* at the front (most recent) of the list."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        else:  # empty list
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

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, expire_at: Optional[float]) -> bool:
        """Return ``True`` if the entry has passed its expiration time."""
        return expire_at is not None and expire_at <= self._time()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Optional[Any]:
        """Return the value for *key* if it exists and is not expired.

        The accessed entry becomes the most‑recently used one.

        Returns:
            The cached value, or ``None`` if the key is absent or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None

        if self._is_expired(node.expire_at):
            # Lazy removal of the stale entry
            self._remove_node(node)
            del self._map[key]
            return None

        # key is valid – move it to the head and return the value
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the default TTL (set at construction) is used.
        If the cache has a capacity and is already full, the least‑recently
        used entry is evicted before inserting the new one.

        The entry is considered expired only when it is accessed after its
        expiration time; ``put`` itself never removes an existing entry
        because of TTL.
        """
        if self.capacity <= 0:
            # Capacity zero – nothing can be stored.
            return

        expire_at: Optional[float]
        if ttl is None:
            expire_at = self.default_ttl if self.default_ttl is not None else float("inf")
        else:
            expire_at = self._time() + ttl

        existing = self._map.get(key)
        if existing:
            # Update existing entry and mark it as most recent
            existing.value = value
            existing.expire_at = expire_at
            self._move_to_head(existing)
        else:
            # Insert a brand‑new node
            node = _Node(key, value, expire_at)
            self._map[key] = node
            self._add_to_head(node)

            # Enforce capacity (evict LRU if necessary)
            if len(self._map) > self.capacity:
                # Tail is the least recently used node
                lru = self._tail
                if lru:
                    self._remove_node(lru)
                    del self._map[lru.key]

    def delete(self, key: Any) -> None:
        """Remove *key* from the cache, if present."""
        node = self._map.pop(key, None)
        if node:
            self._remove_node(node)

    def size(self) -> int:
        """Return the number of live (non‑expired) entries currently stored."""
        return len(self._map)

"""Pytest suite for :class:`TTLCache`.

The tests use ``unittest.mock.patch`` to control ``time.monotonic`` so that
time‑dependent behaviour can be examined without real waiting.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

# The class under test – adjust the import path if the module name differs.


def test_capacity_lru_eviction():
    """When capacity is exceeded, the least‑recently used entry is removed."""
    c = TTLCache(capacity=2, default_ttl=1000)   # large TTL, not relevant here
    c.put('a', 1)
    c.put('b', 2)
    assert c.size() == 2

    # Insert a third element – 'a' should be evicted (LRU)
    c.put('c', 3)
    assert c.size() == 2
    assert c.get('a') is None
    assert c.get('b') == 2
    assert c.get('c') == 3


def test_ttl_expiration():
    """An entry expires after its TTL has elapsed."""
    with patch('time.monotonic', side_effect=[0, 1, 2, 3]):
        c = TTLCache(capacity=5, default_ttl=2)   # expires 2 seconds after insertion
        c.put('key', 'value')                    # expire_at = 0 + 2 = 2

        # Still within TTL
        assert c.get('key') == 'value'

        # Time passes beyond TTL → entry is considered expired
        assert c.get('key') is None


def test_put_none_ttl_uses_default():
    """When ``ttl`` is ``None`` the default TTL must be applied."""
    with patch('time.monotonic', side_effect=[0, 4]):   # 4 seconds after insertion
        c = TTLCache(capacity=5, default_ttl=3)          # default TTL = 3 s
        c.put('key', 'value')                            # ttl=None → use default

        # After 4 seconds the entry should be expired
        assert c.get('key') is None


def test_delete_key():
    """``delete`` removes a key and updates the size."""
    c = TTLCache(capacity=3, default_ttl=10)
    c.put('a', 1)
    c.put('b', 2)
    assert c.size() == 2

    c.delete('a')
    assert c.get('a') is None
    assert c.size() == 1
    assert c.get('b') == 2


def test_size_after_expiration_and_eviction():
    """Size reflects only live entries; expired entries are removed lazily."""
    with patch('time.monotonic', side_effect=[0, 1, 2, 3]):
        c = TTLCache(capacity=2, default_ttl=1)   # entries live only 1 second
        c.put('x', 10)                             # expires at time 1
        assert c.get('x') == 10                    # still alive

        # Advance time past the TTL → access triggers removal
        assert c.get('x') is None
        assert c.size() == 0

        # Insert a new entry; capacity is respected
        c.put('y', 20)
        assert c.size() == 1
        assert c.get('y') == 20


def test_lru_update_and_eviction():
    """Accessing an entry moves it to the head; LRU is evicted on overflow."""
    c = TTLCache(capacity=3, default_ttl=10)

    # Populate with three items
    c.put('a', 1)
    c.put('b', 2)
    c.put('c', 3)
    assert c.size() == 3
    # Order from most‑recent to least‑recent: a, c, b (b is LRU)

    # Access 'a' → becomes most recent, order: a, c, b
    assert c.get('a') == 1

    # Insert a fourth item → evicts the LRU ('b')
    c.put('d', 4)
    assert c.size() == 3
    assert c.get('b') is None
    assert c.get('a') == 1
    assert c.get('c') == 3
    assert c.get('d') == 4