from __future__ import annotations
import time
from typing import Any, Optional

class _Node:
    """Doubly-linked list node for cache entries."""

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU Cache with Time-To-Live (TTL) support.

    Implements an O(1) average-time cache using a doubly-linked list and hash map.
    Entries expire based on per-item TTL or a default TTL. Lazy cleanup of expired
    entries occurs on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize cache with given capacity and default TTL.

        Args:
            capacity (int): Maximum number of entries.
            default_ttl (float): Default lifetime in seconds for each entry.

        Raises:
            ValueError: If capacity ≤ 0 or default_ttl < 0.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        if default_ttl < 0:
            raise ValueError("TTL cannot be negative.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._head: Optional[_Node] = None  # Least recently used
        self._tail: Optional[_Node] = None  # Most recently used
        self._current_time: float = time.monotonic()

    def _update_time(self) -> None:
        """Refresh current time reference."""
        self._current_time = time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node's value has expired."""
        return self._current_time > node.expires_at

    def _move_to_front(self, node: _Node) -> None:
        """Promote node to most-recently-used position."""
        if node is self._tail:
            return
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._head:
            self._head = node.next
        node.prev = self._tail
        node.next = None
        if self._tail:
            self._tail.next = node
        self._tail = node

    def _remove_node(self, node: _Node) -> None:
        """Detach a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = None
        node.next = None

    def _add_node(self, node: _Node) -> None:
        """Insert node as most-recently-used at tail."""
        node.prev = self._tail
        node.next = None
        if self._tail:
            self._tail.next = node
        self._tail = node
        if not self._head:
            self._head = node

    def _evict_lru(self) -> None:
        """Remove the least recently used node if at capacity."""
        if self._head and self._cache:
            lru = self._head
            del self._cache[lru.key]
            self._remove_node(lru)

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value associated with key.

        If key is missing or expired, returns None. Access refreshes position.

        Args:
            key (Any): Key to look up.

        Returns:
            Value if available and not expired; otherwise None.
        """
        if key not in self._cache:
            return None
        node = self._cache[key]
        if self._is_expired(node):
            del self._cache[key]
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.

        Args:
            key (Any): Unique identifier.
            value (Any): Associated value.
            ttl (Optional[float]): Override default TTL in seconds.
        """
        self._update_time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = self._current_time + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return

        if len(self._cache) >= self._capacity:
            self._evict_lru()

        node = _Node(key, value, expires_at)
        self._cache[key] = node
        self._add_node(node)

    def delete(self, key: Any) -> bool:
        """Delete a key from cache.

        Args:
            key (Any): Key to remove.

        Returns:
            True if key was present and removed; False otherwise.
        """
        if key in self._cache:
            node = self._cache.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return current number of valid (non-expired) entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._head = self._tail = None
        self._current_time = time.monotonic()

# test_ttl_cache.py
import pytest
from unittest.mock import patch
import time

def test_put_and_get(monkeypatch):
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10)
        cache.put('a', 1)
        assert cache.get('a') == 1

def test_default_ttl_expiration():
    with patch('time.monotonic', side_effect=[0.0, 10.1]) as mock_time:
        cache = TTLCache(2, 10)
        cache.put('a', 1)
        assert cache.get('a') == 1
        mock_time.return_value = 10.1
        assert cache.get('a') is None

def test_custom_ttl_override():
    with patch('time.monotonic', side_effect=[0.0, 3.1]) as mock_time:
        cache = TTLCache(2, 10)
        cache.put('b', 2, ttl=3)
        assert cache.get('b') == 2
        mock_time.return_value = 3.1
        assert cache.get('b') is None

def test_capacity_eviction():
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10)
        cache.put('x', 1)
        cache.put('y', 2)
        cache.put('z', 3)
        assert cache.size() == 2
        assert cache.get('x') is None
        assert cache.get('y') == 2

def test_size_remains_on_expired_access():
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 1)
        cache.put('exp', 'value')
        assert cache.size() == 1
        with patch('time.monotonic', return_value=1.5):
            assert cache.get('exp') is None
            assert cache.size() == 0

def test_delete_key():
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10)
        cache.put('k', 'v')
        assert cache.delete('k') is True
        assert cache.get('k') is None
        assert cache.delete('nonexistent') is False