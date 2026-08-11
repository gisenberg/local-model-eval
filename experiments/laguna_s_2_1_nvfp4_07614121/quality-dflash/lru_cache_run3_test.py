from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked list node for LRU tracking."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key: Any = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    An LRU cache with per-entry TTL (time-to-live).

    - O(1) average time complexity for get, put, delete.
    - Lazy cleanup: expired entries are removed only when accessed.
    - Uses a doubly-linked list + hash map (no OrderedDict).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of entries in the cache.
            default_ttl: Default time-to-live in seconds for entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        if default_ttl < 0:
            raise ValueError("Default TTL cannot be negative.")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._cache: Dict[Any, _Node] = {}

        # Sentinel head and tail nodes for the doubly-linked list
        self._head: _Node = _Node(None, None, 0)
        self._tail: _Node = _Node(None, None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the sentinel head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front (most recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the least recently used node (before sentinel tail)."""
        lru_node = self._tail.prev
        self._remove(lru_node)
        return lru_node

    def _is_expired(self, node: _Node, current_time: float) -> bool:
        """Check if a node has expired."""
        return current_time >= node.expires_at

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns None if not found or expired.

        Args:
            key: The key to look up.

        Returns:
            The value if found and not expired, else None.
        """
        current_time = time.monotonic()

        if key not in self._cache:
            return None

        node = self._cache[key]

        if self._is_expired(node, current_time):
            # Lazy cleanup
            del self._cache[key]
            self._remove(node)
            return None

        # Move to front as it was recently used
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional TTL.

        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default if None.
        """
        current_time = time.monotonic()
        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = current_time + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
        else:
            if len(self._cache) >= self.capacity:
                # Evict LRU
                lru_node = self._pop_tail()
                del self._cache[lru_node.key]

            new_node = _Node(key, value, expires_at)
            self._cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.
        """
        if key in self._cache:
            node = self._cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """
        Return the number of active (non-expired) entries in the cache.

        Note: This method performs lazy cleanup of expired entries.
        """
        current_time = time.monotonic()
        expired_keys = [
            key for key, node in self._cache.items()
            if self._is_expired(node, current_time)
        ]
        for key in expired_keys:
            node = self._cache.pop(key)
            self._remove(node)
        return len(self._cache)

import pytest
from unittest.mock import patch
import time



def test_get_returns_none_for_missing_key():
    cache = TTLCache(capacity=2, default_ttl=10)
    assert cache.get("missing") is None


def test_put_and_get_basic():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"


def test_lru_eviction():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_ttl_expiration():
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("key", "value")
    assert cache.get("key") == "value"

    with patch("time.monotonic", return_value=time.monotonic() + 6):
        assert cache.get("key") is None


def test_custom_ttl_overrides_default():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("short", "data", ttl=1)

    with patch("time.monotonic", return_value=time.monotonic() + 2):
        assert cache.get("short") is None


def test_delete_removes_entry():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("key", "value")
    cache.delete("key")
    assert cache.get("key") is None
    assert cache.size() == 0