"""TTLCache – an LRU cache with per‑item time‑to‑live.

The implementation uses a hash map (key → _Node) together with a doubly linked
list.  The list keeps the usage order: the node right after the head sentinel
is the most‑recently used, the node right before the tail sentinel is the
least‑recently used.  All operations are O(1) amortised.

Expiration is handled lazily – an entry is removed from the cache the first
time it is accessed after its TTL has passed.
"""

import time
from typing import Any, Dict, Optional


class _Node:
    """A node of the doubly linked list."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry          # time.monotonic() timestamp when the entry expires
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """LRU cache with a default TTL and optional per‑item TTL.

    Parameters
    ----------
    capacity : int
        Maximum number of items that can be stored simultaneously.
    default_ttl : float
        Default time‑to‑live in seconds for items that are inserted without an
        explicit *ttl* argument.

    Methods
    -------
    get(key) -> Any
        Return the value for *key* if it exists and is still fresh; otherwise
        raise ``KeyError``.  The accessed entry becomes the most‑recently used.
    put(key, value, ttl=None) -> None
        Insert a new item or update an existing one.  If the cache is full the
        least‑recently used entry is evicted.  *ttl* overrides the default.
    delete(key) -> None
        Remove *key* from the cache, if present.
    size() -> int
        Current number of valid (non‑expired) items.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non‑negative")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._map: Dict[Any, _Node] = {}

        # sentinel nodes – they simplify edge‑case handling
        self._head = _Node(None, None, 0.0)   # most‑recent side
        self._tail = _Node(None, None, 0.0)   # least‑recent side
        self._head.next = self._tail
        self._tail.prev = self._head

        self._size: int = 0

    # ------------------------------------------------------------------ #
    # internal list helpers (all O(1))
    # ------------------------------------------------------------------ #
    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* immediately after the head sentinel."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the linked list."""
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev
        node.prev = node.next = None

    def _pop_tail(self) -> _Node:
        """Remove and return the LRU node (the one right before the tail)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get(self, key: Any) -> Any:
        """Return the value for *key* if it exists and is not expired.

        Raises
        ------
        KeyError
            If *key* is not present or has expired.  Expired entries are removed
            lazily when accessed.
        """
        node = self._map.get(key)
        if node is None:
            raise KeyError(key)

        now = time.monotonic()
        if node.expiry <= now:
            # Expired – purge it now
            self._remove_node(node)
            del self._map[key]
            raise KeyError(key)

        # Move the accessed node to the front (most recent)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the cache uses ``default_ttl``.  The entry becomes
        fresh for *ttl* seconds from the moment of insertion.  When the cache
        exceeds its *capacity* the least‑recently used entry is evicted.
        """
        now = time.monotonic()
        expiry = (now + ttl) if ttl is not None else (now + self.default_ttl)

        if key in self._map:
            # Update existing entry and mark it as most recent
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._remove_node(node)
            self._add_to_head(node)
            return

        # New entry
        node = _Node(key, value, expiry)
        self._add_to_head(node)
        self._map[key] = node
        self._size += 1

        if self._size > self.capacity:
            # Evict the LRU entry
            lru = self._pop_tail()
            del self._map[lru.key]
            self._size -= 1

    def delete(self, key: Any) -> None:
        """Remove *key* from the cache if it is present."""
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)
            self._size -= 1

    def size(self) -> int:
        """Return the number of items currently stored in the cache."""
        return self._size

"""Pytest suite for TTLCache.

All tests use ``unittest.mock.patch`` on :func:`time.monotonic` to control
the notion of time, enabling deterministic testing of TTL expiration.
"""

import time
from unittest.mock import patch

import pytest

# Import the class we want to test.  Adjust the import path if you place the
# implementation in a different module.


def test_put_get_basic():
    """Put a value and retrieve it – basic functionality."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1


def test_capacity_lru_eviction():
    """Cache with capacity 2: ensure LRU entry is evicted."""
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    # Access 'a' → becomes most recent, 'b' becomes LRU
    cache.get("a")

    # Insert 'c' → 'b' (the LRU) should be removed
    cache.put("c", 3)

    # 'b' must be gone
    with pytest.raises(KeyError):
        cache.get("b")

    # 'a' and 'c' remain
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size() == 2


def test_ttl_expiration():
    """An entry expires after its TTL and is removed lazily."""
    cache = TTLCache(capacity=5, default_ttl=0.1)   # 100 ms TTL

    cache.put("temp", 42)
    assert cache.get("temp") == 42

    # Fast‑forward time beyond the TTL
    with patch("time.monotonic", return_value=0.2):
        with pytest.raises(KeyError):
            cache.get("temp")   # entry has expired and been removed

    # No items remain
    assert cache.size() == 0


def test_default_ttl_used_when_not_provided():
    """When *ttl* is omitted the default_ttl must be applied."""
    cache = TTLCache(capacity=5, default_ttl=0.5)   # 500 ms TTL

    cache.put("x", 10)            # no explicit ttl → uses default
    assert cache.get("x") == 10

    # Wait longer than the default TTL
    with patch("time.monotonic", return_value=1.0):
        with pytest.raises(KeyError):
            cache.get("x")

    assert cache.size() == 0


def test_delete_removes_item():
    """Explicit deletion should remove the key and adjust size."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.size() == 2
    cache.delete("a")

    assert cache.size() == 1
    with pytest.raises(KeyError):
        cache.get("a")


def test_capacity_zero():
    """A cache with capacity 0 never stores any item."""
    cache = TTLCache(capacity=0, default_ttl=5.0)

    cache.put("x", 1)          # should be ignored
    with pytest.raises(KeyError):
        cache.get("x")

    assert cache.size() == 0