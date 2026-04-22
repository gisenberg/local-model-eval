import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU Cache with TTL support using a hash map and doubly-linked list.

    Provides O(1) average time complexity for get, put, and delete operations.
    Implements lazy expiration: entries are checked for TTL only when accessed.
    Not thread-safe.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes simplify boundary handling in the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node immediately after the head sentinel (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list in O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the node before the tail sentinel (LRU position)."""
        if self._tail.prev == self._head:
            return None
        node = self._tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Any:
        """Retrieve a value by key, or None if missing/expired.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if the key is absent or expired.
        """
        node = self._cache.get(key)
        if node is None:
            return None

        # Lazy cleanup: check if TTL has expired
        if time.monotonic() >= node.expiry:
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        # Update access order to MRU
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
        else:
            if self._size >= self.capacity:
                evicted = self._pop_tail()
                if evicted:
                    del self._cache[evicted.key]
                    self._size -= 1

            expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            node = _Node(key, value, expiry)
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists.

        Args:
            key: The cache key to delete.
        """
        node = self._cache.pop(key, None)
        if node is not None:
            self._remove_node(node)
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache.

        Returns:
            The number of cached entries.
        """
        return self._size

import pytest
from unittest.mock import patch

# Assuming TTLCache is in the same module or imported
# 
@patch('time.monotonic', side_effect=[100.0, 100.0])
def test_basic_put_and_get(mock_time: Any) -> None:
    """Test standard insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1


@patch('time.monotonic', side_effect=[100.0, 115.0])
def test_ttl_expiration(mock_time: Any) -> None:
    """Test lazy cleanup when TTL expires on access."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    # Time advances past default TTL (10s)
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[100.0] * 5)
def test_lru_eviction(mock_time: Any) -> None:
    """Test LRU eviction when capacity is exceeded."""
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    
    assert cache.get('a') is None  # Evicted
    assert cache.get('b') == 2     # Still valid
    assert cache.get('c') == 3     # Most recently used


@patch('time.monotonic', side_effect=[100.0, 105.0])
def test_custom_ttl_override(mock_time: Any) -> None:
    """Test that custom TTL overrides default_ttl."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=5.0)
    # At 105.0, custom TTL (5s) has expired, but default (10s) would not
    assert cache.get('a') is None


@patch('time.monotonic', side_effect=[100.0, 100.0])
def test_delete_operation(mock_time: Any) -> None:
    """Test explicit key deletion."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[100.0] * 3)
def test_size_tracking(mock_time: Any) -> None:
    """Test that size() accurately reflects cache state."""
    cache = TTLCache(3, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    
    cache.delete('b')
    assert cache.size() == 2