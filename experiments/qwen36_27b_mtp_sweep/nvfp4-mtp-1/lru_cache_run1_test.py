import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU cache with TTL support.
    Uses a doubly-linked list for O(1) LRU ordering and a hash map for O(1) lookups.
    Implements lazy cleanup: expired entries are removed upon access rather than via background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize cache with maximum capacity and default TTL in seconds."""
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        # Sentinel nodes eliminate boundary checks in linked list operations
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after head (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _pop_tail(self) -> _Node:
        """Remove and return the LRU node (right before tail)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key.
        Returns None if key is missing or expired.
        Updates LRU order on successful access.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        now = time.monotonic()
        
        # Lazy cleanup: check expiration on access
        if now >= node.expires_at:
            self._remove_node(node)
            del self._cache[key]
            return None
            
        # Move to MRU position
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        If key exists, updates value and TTL, moves to MRU.
        If capacity exceeded, evicts LRU entry.
        """
        if key in self._cache:
            self._remove_node(self._cache[key])
            del self._cache[key]

        now = time.monotonic()
        node = _Node(key, value, now + (ttl if ttl is not None else self._default_ttl))
        self._add_to_head(node)
        self._cache[key] = node

        if len(self._cache) > self._capacity:
            evict_node = self._pop_tail()
            del self._cache[evict_node.key]

    def delete(self, key: Any) -> None:
        """Remove key from cache if it exists."""
        if key in self._cache:
            self._remove_node(self._cache[key])
            del self._cache[key]

    def size(self) -> int:
        """Return current number of entries in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch
import time

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0  # Exceeds default TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=5.0)
    mock_time.return_value = 6.0
    assert cache.get('a') is None  # Custom TTL expired
    
    cache.put('b', 2)  # Uses default TTL
    mock_time.return_value = 11.0
    assert cache.get('b') is None  # Default TTL expired

@patch('time.monotonic')
def test_delete_operation(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('a', 10)  # Update 'a', moves to MRU
    cache.put('c', 3)   # Should evict 'b' (now LRU)
    assert cache.get('a') == 10
    assert cache.get('b') is None
    assert cache.get('c') == 3
    assert cache.size() == 2