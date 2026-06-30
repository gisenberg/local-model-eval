import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) list operations."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    
    Uses a custom doubly-linked list and hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed on access or during eviction.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        # Sentinel nodes simplify edge-case handling in the linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head of the list."""
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key.
        
        Returns None if the key is missing or has expired.
        Performs lazy cleanup on access.
        """
        node = self._cache.get(key)
        if node is None:
            return None
            
        # Lazy cleanup: remove if expired
        if time.monotonic() >= node.expiry:
            self._remove_node(node)
            del self._cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If key exists, updates value and refreshes TTL.
        If cache is full, evicts the least recently used item.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        # Evict LRU if at capacity
        if len(self._cache) >= self.capacity:
            lru = self._tail.prev
            if lru is not self._head:
                self._remove_node(lru)
                del self._cache[lru.key]

        # Insert new node
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        node = _Node(key, value, expiry)
        self._cache[key] = node
        self._add_to_head(node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """Return the number of items currently in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that entries are lazily cleaned up when expired."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used items are evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_refreshes_ttl(mock_time):
    """Test that updating an existing key refreshes its TTL and moves it to head."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 3.0
    cache.put('a', 2)  # Updates value, refreshes TTL to 3.0 + 5.0 = 8.0
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 2  # Still valid (6.0 < 8.0)
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion removes key from map and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that per-key TTL correctly overrides the cache default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)  # Default TTL is 100s
    cache.put('a', 1, ttl=2.0)  # Custom TTL is 2s
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0