# ttl_lru_cache.py
from __future__ import annotations

import time
from typing import Any, Optional, Dict


class _Node:
    """Doubly‑linked list node used by TTLCache."""
    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(
        self,
        key: Any,
        value: Any,
        expires_at: float,
        prev: Optional[_Node] = None,
        next: Optional[_Node] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at          # absolute monotonic time
        self.prev = prev
        self.next = next


class TTLCache:
    """
    LRU cache with optional per‑item TTL.

    Parameters
    ----------
    capacity: int
        Maximum number of items the cache may hold.
    default_ttl: float | None
        Time‑to‑live in seconds applied when ``put`` does not receive an explicit
        ``ttl`` argument. ``None`` means items never expire (except when
        manually removed).

    The cache provides O(1) average‑time ``get``, ``put`` and ``delete`` by
    combining a hash map (key → node) with a doubly‑linked list that tracks
    usage order. Expired entries are removed lazily – they are only cleaned
    up when they are encountered during an operation.
    """

    def __init__(self, capacity: int, default_ttl: Optional[float] = None) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self._capacity: int = capacity
        self._default_ttl: Optional[float] = default_ttl

        self._map: Dict[Any, _Node] = {}          # key → node
        self._head: Optional[_Node] = None        # most‑recently used
        self._tail: Optional[_Node] = None        # least‑recently used
        self._size: int = 0                       # number of *live* entries

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _now(self) -> float:
        """Current monotonic time – allows easy mocking in tests."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Return True if ``node`` has passed its TTL."""
        return node.expires_at is not None and self._now() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        """Unlink ``node`` from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:                     # node is head
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:                     # node is tail
            self._tail = node.prev
        node.prev = node.next = None
        self._size -= 1

    def _add_to_head(self, node: _Node) -> None:
        """Insert ``node`` right after the head (MRU position)."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if self._tail is None:          # list was empty
            self._tail = node
        self._size += 1

    def _move_to_head(self, node: _Node) -> None:
        """Assume ``node`` is already in the list – move it to MRU position."""
        if node is self._head:
            return
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the LRU node (tail)."""
        if self._tail is None:
            return None
        node = self._tail
        self._remove_node(node)
        return node

    def _cleanup_expired(self) -> None:
        """
        Lazily remove expired nodes from the tail side.
        Because the list is ordered by usage, expired items are guaranteed to
        be near the tail – we keep popping while the tail is expired.
        """
        while self._tail and self._is_expired(self._tail):
            expired = self._pop_tail()
            del self._map[expired.key]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: Any) -> Any:
        """
        Return the value associated with ``key`` if present and not expired.
        On a hit the node is moved to the MRU position.

        Returns
        -------
        The stored value, or ``None`` if the key is missing or expired.
        """
        self._cleanup_expired()
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            # Expired but still in map – purge it and report miss.
            self._remove_node(node)
            del self._map[key]
            return None
        self._move_to_head(node)
        return node.value

    def put(
        self,
        key: Any,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Insert or update ``key`` with ``value``.

        Parameters
        ----------
        ttl: float | None
            Time‑to‑live in seconds for this entry. If ``None`` the cache’s
            ``default_ttl`` is used. ``None`` also means “no expiration”.
        """
        self._cleanup_expired()

        # Determine absolute expiration time.
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at: Optional[float] = None
        if effective_ttl is not None:
            expires_at = self._now() + effective_ttl

        if key in self._map:
            node = self._map[key]
            # Update value and possibly TTL.
            node.value = value
            node.expires_at = expires_at if expires_at is not None else node.expires_at
            self._move_to_head(node)
            return

        # New entry.
        node = _Node(key, value, expires_at if expires_at is not None else float("inf"))
        self._map[key] = node
        self._add_to_head(node)

        # Enforce capacity.
        if self._size > self._capacity:
            lru = self._pop_tail()
            if lru is not None:
                del self._map[lru.key]

    def delete(self, key: Any) -> None:
        """Remove ``key`` from the cache if it exists."""
        node = self._map.pop(key, None)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """
        Approximate number of *live* (non‑expired) items in the cache.
        The method performs a lazy cleanup before returning the count.
        """
        self._cleanup_expired()
        return self._size


# --------------------------------------------------------------------- #
# Example usage (not part of the test suite)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    cache = TTLCache(capacity=2, default_ttl=0.5)
    cache.put("a", 1)               # uses default_ttl → expires in 0.5 s
    cache.put("b", 2, ttl=1.0)      # explicit TTL → expires in 1.0 s
    print(cache.get("a"))           # → 1
    time.sleep(0.6)
    print(cache.get("a"))           # → None (expired)
    print(cache.get("b"))           # → 2 (still valid)

# test_ttl_lru_cache.py
import time
from unittest.mock import patch

import pytest



def test_get_miss_returns_none():
    cache = TTLCache(capacity=3)
    assert cache.get("missing") is None


def test_get_hit_updates_lru_order():
    """After a get, the accessed key becomes MRU."""
    cache = TTLCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0   # baseline

        # Access 'a' → should move it to head
        assert cache.get("a") == 1
        # Insert a new item; cache is full, so LRU ('b') should be evicted
        cache.put("c", 3)

        # 'b' is gone, 'a' and 'c' remain
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3


def test_put_with_ttl_expires():
    """Items with a TTL disappear after the elapsed time."""
    cache = TTLCache(capacity=10, default_ttl=None)

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache.put("x", 42, ttl=0.1)   # expires at 0.1

        # Immediately after put → hit
        assert cache.get("x") == 42

        # Advance time just before expiration → still present
        mock_time.return_value = 0.09
        assert cache.get("x") == 42

        # Advance past expiration → miss
        mock_time.return_value = 0.11
        assert cache.get("x") is None


def test_delete_removes_key():
    cache = TTLCache(capacity=5)
    cache.put("k", "v")
    assert cache.get("k") == "v"

    cache.delete("k")
    assert cache.get("k") is None
    assert cache.size() == 0


def test_capacity_enforcement_evicts_lru():
    """When over capacity, the least‑recently used (non‑expired) item is removed."""
    cache = TTLCache(capacity=3)

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache.put("a", 1)   # LRU
        cache.put("b", 2)
        cache.put("c", 3)   # MRU

        # Cache is full; insert a new item → should evict 'a'
        cache.put("d", 4)

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert cache.size() == 3


def test_lazy_cleanup_does_not_remove_non_expired():
    """
    Expired items are removed only when they are encountered;
    non‑expired items stay in the map until they become LRU or are accessed.
    """
    cache = TTLCache(capacity=3, default_ttl=0.2)

    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        cache.put("alive", "yes", ttl=1.0)   # lives long
        cache.put("dead", "no", ttl=0.1)     # will expire at 0.1
        cache.put("another", 1)              # uses default ttl (0.2) → expires at 0.2

        # Fast‑forward to just after 'dead' expired but before 'another'
        mock_time.return_value = 0.15
        assert cache.get("dead") is None          # expired → removed on get
        assert cache.get("alive") == "yes"
        assert cache.get("another") == "1"

        # Now advance past 'another' expiration
        mock_time.return_value = 0.25
        # Accessing 'another' triggers its removal
        assert cache.get("another") is None
        assert cache.get("alive") == "yes"
        assert cache.size() == 1