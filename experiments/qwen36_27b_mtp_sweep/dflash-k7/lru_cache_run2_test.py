import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.

    Uses a hash map and a doubly-linked list to achieve O(1) average time
    complexity for get, put, and delete operations. TTL expiration is
    handled lazily upon access (no background threads or timers).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes to simplify edge cases in linked list operations
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head to mark it as recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the tail sentinel (least recently used)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL using monotonic time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key. Returns None if key is missing or expired.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Uses default_ttl if None.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
        else:
            if self._size >= self.capacity:
                # Evict LRU item
                evicted = self._pop_tail()
                del self._cache[evicted.key]
                self._size -= 1

            new_node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
            self._cache[key] = new_node
            self._add_to_head(new_node)
            self._size += 1

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The cache key to remove.
        """
        if key in self._cache:
            node = self._cache.pop(key)
            self._remove_node(node)
            self._size -= 1

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            Integer count of cached items.
        """
        return self._size

import pytest
from unittest.mock import patch

@patch('ttl_cache.time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired items are removed lazily on access."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    # Advance time past TTL
    mock_time.return_value = 6.0
    assert cache.get('a') is None
    assert cache.size() == 0  # Lazy cleanup removed it

@patch('ttl_cache.time.monotonic')
def test_lru_eviction_on_capacity(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that custom TTL overrides default TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion of a key."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_update_existing_key_resets_ttl(mock_time):
    """Test that updating an existing key refreshes its TTL and moves it to head."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 2.0
    cache.put('a', 10)  # Update value and reset TTL
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 10  # Should still be valid
    assert cache.size() == 1