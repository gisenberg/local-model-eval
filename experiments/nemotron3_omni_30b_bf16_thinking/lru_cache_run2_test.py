"""LRU cache with per‑item TTL.

The implementation uses a doubly‑linked list (with dummy head/tail nodes)
and a dict that maps keys to the corresponding node.  All operations
(`get`, `put`, `delete`, `size`) run in O(1) average time.

Expiration is handled lazily: a node is considered expired when its
stored ``expiry`` timestamp is earlier than the current monotonic time.
When an expired node is encountered it is removed immediately, keeping
the structure clean without a background cleanup thread.
"""

from __future__ import annotations

import time
from typing import Optional, Dict


class _Node:
    """A node of the doubly‑linked list."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: object, expiry: float) -> None:
        self.key: str = key
        self.value: object = value
        self.expiry: float = expiry  # time.monotonic() timestamp when the item expires
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Fixed‑capacity LRU cache where each entry expires after a given TTL.

    Parameters
    ----------
    capacity:
        Maximum number of items the cache may hold.  ``0`` means the cache
        cannot store any items.
    default_ttl:
        Default time‑to‑live for items that are inserted without an explicit
        ``ttl`` argument (seconds).  The TTL is measured from the moment the
        item is inserted.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non‑negative")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl

        # Sentinel nodes to avoid edge‑case checks.
        self._head: _Node = _Node(key="", value=None, expiry=0.0)   # most‑recent
        self._tail: _Node = _Node(key="", value=None, expiry=0.0)   # least‑recent
        self._head.next = self._tail
        self._tail.prev = self._head

        # key → node mapping
        self._map: Dict[str, _Node] = {}

    # --------------------------------------------------------------------- #
    # Internal helpers (all O(1))
    # --------------------------------------------------------------------- #
    def _now(self) -> float:
        """Return the current monotonic timestamp."""
        return time.monotonic()

    def _detach(self, node: _Node) -> None:
        """Remove *node* from the linked list."""
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
        node.prev = node.next = None  # help GC

    def _attach_head(self, node: _Node) -> None:
        """Insert *node* right after the head (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most‑recently used."""
        self._detach(node)
        self._attach_head(node)

    def _purge_expired(self) -> None:
        """Remove all nodes whose expiry time is in the past."""
        now = self._now()
        cur = self._tail.prev  # start from the least‑recent side
        while cur is not self._head:
            nxt = cur.prev  # keep a reference before possible removal
            if cur.expiry <= now:
                self._detach(cur)
                del self._map[cur.key]
            cur = nxt

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: str) -> Optional[object]:
        """Return the value for *key* if it exists and is not expired.

        If the key is missing or expired, ``None`` is returned and the
        entry is removed lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None

        # Lazy expiration check
        if node.expiry <= self._now():
            self._detach(node)
            del self._map[key]
            return None

        # Key is fresh → move to head (most recent) and return value
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: object, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the ``default_ttl`` is used.  When the cache
        reaches its capacity the least‑recently used entry is evicted
        (unless ``capacity`` is ``0``).
        """
        expiry = self._now() + (ttl if ttl is not None else self.default_ttl)

        if key in self._map:
            # Update existing entry
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        # New entry
        if self.capacity == 0:
            return  # cannot store anything

        # Evict LRU if necessary
        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            if lru is not self._head:          # there is something to evict
                self._detach(lru)
                del self._map[lru.key]

        # Insert new node
        node = _Node(key, value, expiry)
        self._map[key] = node
        self._attach_head(node)

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed and was removed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._detach(node)
        return True

    def size(self) -> int:
        """Return the number of *valid* (non‑expired) items currently stored."""
        # Ensure we purge any expired entries that might have accumulated.
        self._purge_expired()
        return len(self._map)

    # --------------------------------------------------------------------- #
    # Representation helpers (optional, useful for debugging)
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        items = []
        cur = self._head.next
        while cur is not self._tail:
            items.append(f"{cur.key!r}:{cur.value!r}[ttl={cur.expiry}]")
            cur = cur.next
        return f"<TTLCache size={len(self._map)} capacity={self.capacity} {', '.join(items)}>"

"""Pytest suite for ``TTLCache`` using ``unittest.mock.patch`` on ``time.monotonic``."""

import pytest
from unittest.mock import patch



# --------------------------------------------------------------------- #
# Helper to advance simulated time
# --------------------------------------------------------------------- #
def _advance(monotonic_values, steps):
    """Advance the patched ``time.monotonic`` by ``steps`` and return the new value."""
    return monotonic_values[0] + steps


# --------------------------------------------------------------------- #
# 1. Basic get / put functionality
# --------------------------------------------------------------------- #
def test_basic_get_put():
    cache = TTLCache(capacity=3, default_ttl=10)

    cache.put("a", 1)
    assert cache.get("a") == 1

    # key not present
    assert cache.get("b") is None
    assert cache.size() == 1


# --------------------------------------------------------------------- #
# 2. TTL override per‑item
# --------------------------------------------------------------------- #
def test_ttl_override():
    cache = TTLCache(capacity=10, default_ttl=5)

    cache.put("x", "short", ttl=2)   # expires after 2 s
    assert cache.get("x") == "short"

    # Fast‑forward time by 3 s → entry expired
    with patch("time.monotonic", return_value=_advance([10], 3)):
        assert cache.get("x") is None
        assert cache.size() == 0


# --------------------------------------------------------------------- #
# 3. LRU eviction when capacity is exceeded
# --------------------------------------------------------------------- #
def test_lru_eviction():
    cache = TTLCache(capacity=2, default_ttl=100)

    cache.put("first", 1)
    cache.put("second", 2)
    assert cache.size() == 2

    # Access "first" → becomes most recent
    assert cache.get("first") == 1

    # Insert a third item → "second" (the LRU) should be evicted
    cache.put("third", 3)
    assert cache.get("second") is None          # evicted
    assert cache.get("first") == 1
    assert cache.get("third") == 3
    assert cache.size() == 2


# --------------------------------------------------------------------- #
# 4. Delete operation
# --------------------------------------------------------------------- #
def test_delete():
    cache = TTLCache(capacity=5, default_ttl=10)
    cache.put("k1", "value1")
    cache.put("k2", "value2")

    assert cache.delete("k1") is True
    assert cache.get("k1") is None
    assert cache.size() == 1

    assert cache.delete("k2") is True
    assert cache.get("k2") is None
    assert cache.size() == 0

    # Deleting a non‑existent key is a no‑op
    assert cache.delete("k3") is False


# --------------------------------------------------------------------- #
# 5. Capacity of zero (no items can be stored)
# --------------------------------------------------------------------- #
def test_zero_capacity():
    cache = TTLCache(capacity=0, default_ttl=10)

    cache.put("any", "value")
    assert cache.size() == 0
    assert cache.get("any") is None

    # Even after advancing time, nothing changes
    with patch("time.monotonic", return_value=_advance([0], 100)):
        assert cache.size() == 0


# --------------------------------------------------------------------- #
# 6. Lazy cleanup works across multiple operations
# --------------------------------------------------------------------- #
def test_lazy_cleanup_multiple_operations():
    cache = TTLCache(capacity=3, default_ttl=4)

    # Insert three items; they will expire after 4 s
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    # At time = 0 all are valid
    assert cache.size() == 3
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") == 3

    # Advance time by 5 s → all entries expired
    with patch("time.monotonic", return_value=_advance([0], 5)):
        # The first access triggers lazy removal of the expired node
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None
        # All entries have been removed lazily
        assert cache.size() == 0

    # Insert again; the cache should work after the “clean‑up” window
    cache.put("d", 4)
    assert cache.get("d") == 4
    assert cache.size() == 1