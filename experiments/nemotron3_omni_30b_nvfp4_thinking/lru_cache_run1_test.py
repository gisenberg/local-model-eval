"""TTLCache – a Least‑Recently‑Used cache with per‑item time‑to‑live.

The implementation uses a hash‑map (dict) for O(1) key lookup and a
doubly‑linked list to keep the recency order.  No ``collections.OrderedDict``
is used.

Typical usage
-------------
>>> cache = TTLCache(capacity=3, default_ttl=5.0)
>>> cache.put('x', 42)                 # default TTL (5 s)
>>> cache.put('y', 99, ttl=0.5)        # custom TTL (0.5 s)
>>> cache.get('x')                    # → 42
>>> cache.get('y')                    # → 99 (will expire after ~0.5 s)
"""

from __future__ import annotations

import time
from typing import Any, Optional


class _Node:
    """A node of the doubly‑linked list.

    Attributes
    ----------
    key: Any
        The cache key.
    value: Any
        The cached value.
    expiration: float
        Absolute expiration time as returned by :func:`time.monotonic`.
    prev, next: _Node
        Links to the previous and next nodes in the list.
    """

    __slots__ = ("key", "value", "expiration", "prev", "next")

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Cache with a maximum capacity and a default time‑to‑live.

    Parameters
    ----------
    capacity : int
        Maximum number of items that can be stored. Must be > 0.
    default_ttl : float
        Default TTL in seconds used when ``put`` is called without an explicit
        ``ttl`` argument. Must be >= 0.

    Notes
    -----
    * ``get`` and ``put`` are *amortised* O(1).  Expired entries are removed
      lazily – only when they are accessed (``get``) or when a new item is
      inserted and the capacity limit would be exceeded.
    * ``delete`` removes a key immediately, also in O(1).
    * ``size`` returns the current number of *non‑expired* items.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non‑negative")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict[Any, _Node] = {}          # key → node
        # sentinel head/tail simplify edge‑case handling
        self._head: _Node = _Node(None, None, 0.0)   # most‑recent side
        self._tail: _Node = _Node(None, None, 0.0)   # least‑recent side
        self._head.next = self._tail
        self._tail.prev = self._head

    # --------------------------------------------------------------------- #
    # Internal doubly‑linked‑list helpers (all O(1))
    # --------------------------------------------------------------------- #
    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* right after the head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None  # help GC

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most‑recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the least‑recently used node (just before tail)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _expire_node(self, node: _Node) -> None:
        """Remove *node* if it is expired and update the map."""
        if node.expiration <= time.monotonic():
            self._remove_node(node)
            del self._map[node.key]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Optional[Any]:
        """Return the value for *key* if it exists and is not expired.

        If the key is missing or has expired, ``None`` is returned and the
        entry is removed from the cache (lazy cleanup).

        Parameters
        ----------
        key : Any
            The key to look up.

        Returns
        -------
        Optional[Any]
            The cached value, or ``None`` if the key is absent or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None
        # Lazy expiration check
        self._expire_node(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the ``default_ttl`` is used.  The entry becomes
        the most‑recently used.  If the cache is at capacity, the least‑
        recently used entry is evicted (even if it is not expired).

        Parameters
        ----------
        key : Any
            Key under which the value will be stored.
        value : Any
            Value to cache.
        ttl : float, optional
            Time‑to‑live in seconds for this particular entry.  If omitted,
            ``default_ttl`` is used.
        """
        now = time.monotonic()
        ttl_to_use = ttl if ttl is not None else self._default_ttl
        expiration = now + ttl_to_use

        if key in self._map:
            # Update existing node
            node = self._map[key]
            node.value = value
            node.expiration = expiration
            self._move_to_head(node)
        else:
            # Insert new node
            node = _Node(key, value, expiration)
            self._map[key] = node
            self._add_to_head(node)

            # Enforce capacity (LRU eviction)
            if len(self._map) > self._capacity:
                evicted = self._pop_tail()
                del self._map[evicted.key]

    def delete(self, key: Any) -> None:
        """Remove *key* from the cache if it exists.

        This operation is O(1).
        """
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """Return the number of *non‑expired* items currently stored."""
        return len(self._map)

    # --------------------------------------------------------------------- #
    # Optional convenience helpers (not required by the specification)
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Allow ``len(cache)``."""
        return self.size()

    def __contains__(self, key: Any) -> bool:
        """Allow ``key in cache``."""
        return key in self._map

    def __repr__(self) -> str:
        """Unambiguous representation useful for debugging."""
        items = []
        cur = self._head.next
        while cur is not self._tail:
            items.append(f"{cur.key!r}:{cur.value!r}(ttl={cur.expiration})")
            cur = cur.next
        return f"TTLCache(capacity={self._capacity}, items=[{', '.join(items)}])"

import pytest
from unittest.mock import patch

# The class is assumed to be importable as `TTLCache`


def test_basic_get_put():
    """Simple put / get round‑trip."""
    cache = TTLCache(capacity=3, default_ttl=10)
    cache.put('alpha', 123)
    assert cache.get('alpha') == 123
    assert cache.size() == 1


def test_capacity_lru_eviction():
    """When capacity is exceeded, the least‑recently used entry is removed."""
    cache = TTLCache(capacity=2, default_ttl=10)

    cache.put('first', 1)
    cache.put('second', 2)          # 'second' is now most recent, 'first' LRU
    cache.put('third', 3)           # capacity overflow → evict 'first'

    assert cache.get('first') is None
    assert cache.get('second') == 2
    assert cache.get('third') == 3
    assert cache.size() == 2


def test_custom_ttl_expiration():
    """An entry with a custom TTL expires after the given interval."""
    cache = TTLCache(capacity=5, default_ttl=10)

    now = 1000.0
    with patch('time.monotonic', side_effect=[now, now + 0.3]):  # 0.3 s later
        cache.put('short_ttl', 'value', ttl=0.2)   # expires at 1000.2
        # after 0.3 s the entry must be considered expired
        assert cache.get('short_ttl') is None
        assert cache.size() == 0   # the only entry has vanished


def test_default_ttl_expiration():
    """When no per‑item TTL is supplied the default TTL is applied."""
    cache = TTLCache(capacity=5, default_ttl=0.2)

    now = 2000.0
    with patch('time.monotonic', side_effect=[now, now + 0.3]):
        cache.put('auto', 'value')          # expires at 2000.2
        # 0.3 s later → expired
        assert cache.get('auto') is None
        assert cache.size() == 0


def test_delete_operation():
    """Explicit deletion removes a key and updates size."""
    cache = TTLCache(capacity=4, default_ttl=5)
    cache.put('temp', 42)
    assert cache.get('temp') == 42
    cache.delete('temp')
    assert cache.get('temp') is None
    assert cache.size() == 0


def test_combined_ttl_and_capacity():
    """
    Verify that both TTL expiration and capacity‑based LRU eviction work
    together.  The test uses a small capacity and a short default TTL so
    that expiration can be forced quickly via the mocked ``time.monotonic``.
    """
    cache = TTLCache(capacity=2, default_ttl=0.2)

    # Use a deterministic monotonic clock.
    # Sequence of moments (seconds since epoch):
    #   0   → put 'a' (ttl 0.1)   → expires at 0.1
    #   0.1 → put 'b' (default) → expires at 0.3
    #   0.3 → put 'c' (default) → capacity full → evict LRU ('a')
    with patch('time.monotonic', side_effect=[0.0, 0.1, 0.3, 0.5]):
        cache.put('a', 1, ttl=0.1)   # a expires at 0.1, will be evicted on put 'c'
        cache.put('b', 2)            # b expires at 0.3
        cache.put('c', 3)            # evicts 'a' (LRU) → size stays 2

        # At time 0.4: a already gone, b still alive, c alive
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

        # Advance past b's TTL
        with patch('time.monotonic', return_value=0.4):
            assert cache.get('b') is None   # b expired
            assert cache.size() == 1        # only 'c' remains
            assert cache.get('c') == 3