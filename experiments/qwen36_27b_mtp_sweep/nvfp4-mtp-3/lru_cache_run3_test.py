import time
from typing import Any, Optional, Dict

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with TTL support using a doubly-linked list and hash map.
    
    Provides O(1) average time complexity for get, put, and delete operations.
    Uses lazy cleanup: expired entries are removed upon access rather than 
    via background threads or periodic scans.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: Dict[Any, _Node] = {}
        self._size = 0
        
        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _evict(self) -> None:
        """Remove the least recently used item (tail)."""
        if self._size == 0:
            return
        lru_node = self._tail.prev
        self._remove_node(lru_node)
        del self._cache[lru_node.key]
        self._size -= 1

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return node.expires_at <= time.monotonic()

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value by key. Returns None if missing or expired."""
        if key not in self._cache:
            return None

        node = self._cache[key]
        # Lazy cleanup: remove expired entry on access
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key-value pair with optional TTL."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.monotonic() + effective_ttl

        if key in self._cache:
            # Update existing entry
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Evict if at capacity
            if self._size >= self.capacity:
                self._evict()

            # Insert new entry
            new_node = _Node(key, value, expires_at)
            self._add_to_head(new_node)
            self._cache[key] = new_node
            self._size += 1

    def delete(self, key: Any) -> None:
        """Remove key from cache if present."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return current number of items in cache."""
        return self._size

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
    mock_time.return_value = 6.0  # Advance time past TTL
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
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    mock_time.return_value = 1.0
    cache.put('a', 2)  # Updates value and resets TTL
    assert cache.get('a') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)  # Custom TTL shorter than default
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0