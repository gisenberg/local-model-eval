"""TTLCache – LRU cache with per‑item TTL (time‑to‑live).

The cache stores ``key → value`` pairs, evicts the *least recently used*
entry when the capacity is exceeded and removes entries whose TTL has
expired lazily on every access.  All operations run in O(1) average time.

Typical usage::

    >>> from cache import TTLCache
    >>> c = TTLCache(3, default_ttl=60)
    >>> c.put('a', 42)
    >>> c.get('a')          # → 42
    >>> c.put('b', 'hi')
    >>> c.delete('a')
"""

from __future__ import annotations

import time
from typing import Any, Optional


class _Node:
    """Internal doubly‑linked list node used by TTLCache."""
    __slots__ = ("key", "value", "ttl_start", "prev", "next")

    def __init__(self,
                 key: Optional[Any] = None,
                 value: Any = None,
                 ttl_start: float = 0.0) -> None:
        self.key = key
        self.value = value
        self.ttl_start = ttl_start          # monotonic time when the entry was created / last accessed
        self.prev: _Node | None = None
        self.next: _Node | None = None


class TTLCache:
    """LRU cache with per‑item TTL.

    Parameters
    ----------
    capacity : int
        Maximum number of entries that can be stored simultaneously.
    default_ttl : float, optional
        Default TTL (seconds) for entries whose ``ttp`` argument is omitted.
        Must be non‑negative; a value of 0 means the entry never expires.

    Methods
    -------
    get(key) -> Optional[Any]
        Return the stored value if it exists and has not expired,
        otherwise delete it and return ``None``.  Accesses are O(1).
    put(key, value, ttl=None) -> None
        Insert or update an entry with optional TTL.  Evicts LRU when full.
    delete(key) -> bool
        Remove *key* from the cache; returns ``True`` if it existed.
    size() -> int
        Current number of stored entries (ignores expired ones).
    """

    def __init__(self, capacity: int, default_ttl: float = 60.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = max(default_ttl, 0.0)

        # doubly‑linked list with sentinel head/tail nodes
        self._head: _Node = _Node()
        self._tail: _Node = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

        # hash map key → node (only valid, non‑expired entries)
        self._map: dict[Any, _Node] = {}

    # ------------------------------------------------------------------ #
    # internal helpers – all O(1) -------------------------------------- #
    # ------------------------------------------------------------------ #
    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* right after the head sentinel (most‑recent)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the list and clear its map entry."""
        prev_n = node.prev
        nxt_n = node.next
        prev_n.next = nxt_n
        nxt_n.prev = prev_n
        if node.key is not None:
            del self._map[node.key]

    def _move_to_head(self, node: _Node) -> None:
        """Move *node* to the front of the list (most‑recent)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove the least‑recently used entry when capacity is exceeded."""
        lru = self._head.next
        if lru is not self._tail:          # should never happen for a valid cache
            self._remove_node(lru)

    # ------------------------------------------------------------------ #
    # public API -------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def get(self, key: Any) -> Optional[Any]:
        """Return the value associated with *key* if it exists and is not expired.

        The function lazily removes expired entries using ``time.monotonic()``.
        """
        node = self._map.get(key)
        if node is None:
            return None

        now = time.monotonic()
        if now - node.ttl_start > self.default_ttl:
            # TTL expired – clean up lazily
            self._remove_node(node)
            return None

        # Access makes this entry most‑recent → move to head
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert *key* with *value*.  If the key already exists its value is updated.

        ``ttp`` may be omitted; in that case ``default_ttl`` is used.
        When the cache exceeds its capacity the LRU entry is evicted."""
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        now = time.monotonic()
        ttl_val = ttl if ttl is not None else self.default_ttl
        node = _Node(key=key, value=value, ttl_start=now)

        if key in self._map:
            # existing entry – update value and reset its TTL start time
            node.value = value
            node.ttl_start = now
        else:
            self._add_to_head(node)
            self._map[key] = node

        if len(self._map) > self.capacity:
            self._evict_lru()

    def delete(self, key: Any) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if a mapping existed and was removed."""
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Current number of stored entries (ignores expired ones)."""
        return len(self._map)


# -------------------------------------------------------------------------- #
# pytest tests – using ``unittest.mock.patch`` on ``time.monotonic`` -------- #
# -------------------------------------------------------------------------- #
import unittest
from unittest import mock

