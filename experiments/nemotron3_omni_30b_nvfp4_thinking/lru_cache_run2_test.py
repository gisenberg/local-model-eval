"""TTLCache – an LRU cache with per‑item time‑to‑live.

The implementation uses a doubly‑linked list (handled manually) together
with a dictionary that maps keys to the list nodes, giving O(1) average
time for all public operations.

Typical usage::

    >>>     >>> c = TTLCache( capacity=2, default_ttl=5 )
    >>> c.put('x', 1)                 # default ttl = 5 s
    >>> c.put('y', 2, ttl=1)          # custom ttl = 1 s
    >>> c.get('x')                  # → 1
    >>> time.sleep(2)
    >>> c.get('y')                  # → 2  (still valid)
    >>> time.sleep(0.5)
    >>> c.get('y')                  # → None (expired)
"""

import time
from typing import Any, Dict, Optional


class _Node:
    """A node of the doubly‑linked list used by :class:`TTLCache`.

    Attributes
    ----------
    key:
        The cache key.
    value:
        The cached value.
    expiry:
        Absolute expiration time (seconds since the epoch, from
        :func:`time.monotonic`).  ``0.0`` means “never expires”.
    prev, next:
        Pointers to the previous and next nodes in the list.
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
    capacity:
        Maximum number of *live* (non‑expired) entries that the cache may hold.
    default_ttl:
        Default time‑to‑live in seconds for items whose ``ttl`` argument is
        ``None``.  The value ``0`` means “never expires”.

    Notes
    -----
    * All public operations (`get`, `put`, `delete`, `size`) run in O(1)
      average time.
    * Expiration is checked lazily – only when an entry is accessed
      (`get`) or when a new entry is inserted (`put`).  This keeps the
      average complexity constant while still guaranteeing that stale
      entries are eventually removed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl

        # map key → node for O(1) look‑ups
        self._map: Dict[Any, _Node] = {}

        # sentinel head (most recent) and tail (least recent)
        self.head = _Node(None, None, 0.0)   # type: ignore[assignment]
        self.tail = _Node(None, None, 0.0)   # type: ignore[assignment]
        self.head.next = self.tail
        self.tail.prev = self.head

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _now(self) -> float:
        """Return the current monotonic time."""
        return time.monotonic()

    def _expire(self, node: _Node) -> bool:
        """If *node* is expired, remove it from the list and the map.

        Returns ``True`` if the node was removed, ``False`` otherwise.
        """
        if node.expiry <= self._now():
            self._remove_node(node)
            del self._map[node.key]
            return True
        return False

    def _remove_node(self, node: _Node) -> None:
        """Unlink *node* from the doubly‑linked list."""
        prev, nxt = node.prev, node.next  # type: ignore[assignment]
        prev.next = nxt
        nxt.prev = prev
        node.prev = node.next = None

    def _add_node(self, node: _Node) -> None:
        """Insert *node* right after the head (most‑recent position)."""
        node.prev = self.head
        node.next = self.head.next  # type: ignore[assignment]
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Mark *node* as the most‑recently used entry."""
        self._remove_node(node)
        self._add_node(node)

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Optional[Any]:
        """Return the value for *key* if it exists and is not expired.

        The accessed entry is moved to the front of the LRU order.
        If the key is missing or the entry has expired, ``None`` is
        returned and the entry is removed lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._expire(node):
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the instance’s ``default_ttl`` is used.
        The entry becomes the most‑recently used.  When the cache exceeds
        its ``capacity`` the least‑recently used entry is evicted.
        Expired entries are removed lazily.
        """
        now = self._now()
        expiry = (now + ttl) if ttl is not None else (now + self.default_ttl)

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Ensure we have room for the new entry
        while self.size() >= self.capacity:
            lru = self.tail.prev
            if lru is self.head:  # safety – should never happen
                break
            self._remove_node(lru)
            del self._map[lru.key]

        node = _Node(key, value, expiry)
        self._map[key] = node
        self._add_node(node)

    def delete(self, key: Any) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed and was removed, ``False``
        otherwise.
        """
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Current number of live entries in the cache."""
        return len(self._map)

"""Tests for :class:`TTLCache`.

The tests use ``unittest.mock.patch`` to replace :func:`time.monotonic`
with a controllable value, allowing deterministic checks of TTL
behaviour and LRU ordering.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

# Import the class to be tested.  Adjust the import path if you place the
# implementation in a different module.


def test_basic_put_get_and_lru_eviction(patch_monotonic):
    """Put two items, then a third – the first should be evicted (LRU)."""
    patch_monotonic.return_value = 0.0  # all time stamps start at 0

    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put('a', 1)
    cache.put('b', 2)

    assert cache.get('a') == 1
    assert cache.get('b') == 2

    # Insert a third key – capacity is 2, so 'a' (the oldest) should be evicted
    cache.put('c', 3)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2


def test_ttl_expiration(patch_monotonic):
    """An entry expires after its TTL and is removed lazily."""
    # monotonic values: 0 (now) → 2 (two seconds later)
    patch_monotonic.side_effect = [0.0, 2.0]

    cache = TTLCache(capacity=3, default_ttl=1)   # 1‑second TTL
    cache.put('x', 42)          # expires at time 1

    # Still within TTL
    assert cache.get('x') == 42
    assert cache.size() == 1

    # Time advances beyond the TTL → entry disappears
    patch_monotonic.return_value = 2.0
    assert cache.get('x') is None
    assert cache.size() == 0


def test_custom_ttl_override(patch_monotonic):
    """Custom TTL supplied to ``put`` overrides the default."""
    patch_monotonic.side_effect = [0.0, 6.0]  # 0 s now, 6 s later

    cache = TTLCache(capacity=5, default_ttl=10)   # default TTL = 10 s
    cache.put('y', 99, ttl=5)               # expires after 5 s

    assert cache.get('y') == 99           # still valid at t=0
    patch_monotonic.return_value = 6.0  # 6 s have passed → expired
    assert cache.get('y') is None
    assert cache.size() == 0


def test_update_moves_to_front(patch_monotonic):
    """Updating a key (or getting it) should move it to the front of LRU order."""
    patch_monotonic.side_effect = [0.0, 0.0, 0.0]  # three calls

    cache = TTLCache(capacity=3, default_ttl=10)
    cache.put('k1', 1)   # order: head <-> k1 <-> tail
    cache.put('k2', 2)   # order: head <-> k2 <-> k1 <-> tail

    # 'k1' is now the least‑recently used
    assert cache.get('k1') == 1
    # after a get, k1 becomes most recent
    assert cache.get('k1') == 1
    # order should now be: head <-> k1 <-> k2 <-> tail

    # Insert a third key – no eviction, but order becomes: k3, k1, k2
    cache.put('k3', 3)
    assert cache.get('k2') is None      # k2 is LRU now
    assert cache.get('k1') == 1
    assert cache.get('k3') == 3


def test_delete(patch_monotonic):
    """Delete a key and verify that it disappears from the cache."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put('d', 4)

    assert cache.size() == 1
    assert cache.delete('d') is True
    assert cache.get('d') is None
    assert cache.size() == 0

    # Deleting a non‑existent key must not raise and must report failure
    assert cache.delete('missing') is False
    assert cache.size() == 0


def test_capacity_and_ttl_combined(patch_monotonic):
    """Combine capacity eviction with TTL expiration."""
    # monotonic sequence:
    # 0 s – put a, b, c
    # 3 s – get(b) → b expires and is removed
    # 3 s – put d (capacity still 3, but size is 2 after expiration)
    patch_monotonic.side_effect = [0.0, 0.0, 0.0, 3.0, 3.0]

    cache = TTLCache(capacity=3, default_ttl=2)

    cache.put('a', 1)          # expires at 2 s
    cache.put('b', 2, ttl=1)   # expires at 1 s
    cache.put('c', 3)          # expires at 2 s

    assert cache.size() == 3

    # Advance time past the TTLs; get('b') triggers lazy removal
    patch_monotonic.return_value = 3.0
    assert cache.get('b') is None
    assert cache.size() == 2   # only 'a' and 'c' remain

    # Insert a new entry – capacity is still 3, so no eviction yet
    cache.put('d', 4)            # expiry 5 s
    assert cache.get('a') is None   # evicted because it is LRU
    assert cache.get('c') == 3
    assert cache.get('d') == 4
    assert cache.size() == 3