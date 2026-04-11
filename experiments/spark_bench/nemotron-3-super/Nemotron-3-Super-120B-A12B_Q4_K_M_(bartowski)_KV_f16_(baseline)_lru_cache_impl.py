"""
TTLCache – LRU cache with per‑item time‑to‑live.

The cache keeps a doubly‑linked list (most‑recently‑used at the head,
least‑recently‑used at the tail) together with a hash map for O(1)
look‑ups.  Expired entries are removed lazily – they are deleted the
next time they are accessed (get, put, delete or when making room for a
new item).  The public API follows the specification in the prompt.
"""

from __future__ import annotations

import time
from typing import Any, Optional


class TTLCache:
    """LRU cache with optional TTL per entry.

    Parameters
    ----------
    capacity:
        Maximum number of items the cache may hold.
    default_ttl:
        Default time‑to‑live in seconds for entries that do not supply an
        explicit ``ttl`` argument to :meth:`put`.
    """

    class _Node:
        __slots__ = ("key", "value", "expiry", "prev", "next")

        def __init__(
            self,
            key: str,
            value: Any,
            expiry: float,
            prev: Optional["TTLCache._Node"] = None,
            next: Optional["TTLCache._Node"] = None,
        ) -> None:
            self.key: str = key
            self.value: Any = value
            self.expiry: float = expiry
            self.prev: Optional[TTLCache._Node] = prev
            self.next: Optional[TTLCache._Node] = next

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if default_ttl < 0:
            raise ValueError("default_ttl must be >= 0")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict[str, TTLCache._Node] = {}
        self._head: Optional[TTLCache._Node] = None  # most‑recently‑used
        self._tail: Optional[TTLCache._Node] = None  # least‑recently‑used

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _now(self) -> float:
        """Current monotonic time."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Return True if *node* has expired."""
        return node.expiry <= self._now()

    def _unlink(self, node: _Node) -> None:
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

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* as the new head (most‑recently‑used)."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if self._tail is None:  # list was empty
            self._tail = node

    def _move_to_head(self, node: _Node) -> None:
        """Assume *node* is already in the list; move it to the head."""
        if node is self._head:
            return
        self._unlink(node)
        self._add_to_head(node)

    def _evict_lru_non_expired(self) -> None:
        """
        Evict the least‑recently‑used **non‑expired** item.
        If all items are expired, they are removed first.
        """
        # Remove expired nodes from the tail first (they are LRU among expired)
        while self._tail and self._is_expired(self._tail):
            expired = self._tail
            self._unlink(expired)
            del self._map[expired.key]

        # If we still need to make room, evict the LRU non‑expired node
        if self._tail:
            lru = self._tail
            self._unlink(lru)
            del self._map[lru.key]

    def _make_room(self) -> None:
        """Ensure there is space for at least one more entry."""
        while len(self._map) >= self._capacity:
            self._evict_lru_non_expired()

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """
        Return the value for *key* if present and not expired.
        Accessing the key makes it most‑recently‑used.

        Returns
        -------
        The cached value or ``None`` if missing/expired.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            # expired – remove it
            self._unlink(node)
            del self._map[node.key]
            return None
        self._move_to_head(node)
        return node.value

    def put(
        self, key: str, value: Any, ttl: Optional[float] = None
    ) -> None:
        """
        Insert or update *key* with *value*.
        If the cache is at capacity, evict the LRU non‑expired item
        (clearing all expired items first if necessary).

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to store.
        ttl:
            Time‑to‑live in seconds; if ``None`` the ``default_ttl`` is used.
        """
        expiry = self._now() + (ttl if ttl is not None else self._default_ttl)

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        # new entry
        self._make_room()
        node = self._Node(key, value, expiry)
        self._map[key] = node
        self._add_to_head(node)

    def delete(self, key: str) -> bool:
        """
        Remove *key* from the cache.

        Returns
        -------
        True if the key existed, False otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._unlink(node)
        return True

    def size(self) -> int:
        """
        Return the number of **non‑expired** items.
        Expired items are removed lazily during this call.

        Returns
        -------
        Count of valid entries.
        """
        now = self._now()
        # iterate over a copy of keys because we may delete while iterating
        for k in list(self._map.keys()):
            node = self._map[k]
            if node.expiry <= now:
                self._unlink(node)
                del self._map[k]
        return len(self._map)
