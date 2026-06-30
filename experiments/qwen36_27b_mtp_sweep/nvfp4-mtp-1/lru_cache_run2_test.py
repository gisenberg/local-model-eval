import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'ttl', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, ttl: float, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Expired items are lazily cleaned up upon access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        
        # Sentinel nodes simplify edge-case list manipulation
        self._head = _Node(None, None, 0, 0)
        self._tail = _Node(None, None, 0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> Optional[Any]:
        """Remove and return the key of the least recently used node."""
        if len(self._cache) == 0:
            return None
        node = self._tail.prev
        self._remove_node(node)
        return node.key

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return node.expires_at < time.monotonic()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key. Returns None if key is missing or expired.
        Moves accessed item to the head (LRU update). O(1) avg time.
        """
        node = self._cache.get(key)
        if node is None:
            return None
            
        # Lazy cleanup: check expiration on access
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. O(1) avg time.
        If key exists and is not expired, updates value and refreshes TTL.
        If capacity is exceeded, evicts the LRU item.
        """
        node = self._cache.get(key)
        if node is not None:
            if self._is_expired(node):
                # Treat expired key as a miss; remove before insertion
                self._remove_node(node)
                del self._cache[key]
            else:
                # Update existing valid entry
                node.value = value
                node.ttl = ttl if ttl is not None else self._default_ttl
                node.expires_at = time.monotonic() + node.ttl
                self._move_to_head(node)
                return

        # Handle new key or replacement of expired key
        if len(self._cache) >= self._capacity:
            evicted_key = self._remove_tail()
            if evicted_key is not None:
                del self._cache[evicted_key]

        actual_ttl = ttl if ttl is not None else self._default_ttl
        node = _Node(key, value, actual_ttl, time.monotonic() + actual_ttl)
        self._add_to_head(node)
        self._cache[key] = node

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) avg time."""
        node = self._cache.get(key)
        if node is not None:
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """Return the current number of valid items in the cache. O(1) time."""
        return len(self._cache)

import pytest
from unittest.mock import patch
import time

# Import TTLCache from the module above
# 
@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup when accessing an expired key."""
    mock_time.side_effect = [0.0, 15.0]  # put at 0.0, get at 15.0 (>10s TTL)
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test capacity enforcement evicts least recently used item."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_update_refreshes_ttl_and_maintains_lru(mock_time):
    """Test that updating a key refreshes TTL and moves it to head."""
    mock_time.side_effect = [0.0, 0.0, 0.0, 15.0, 15.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)  # expires at 10.0
    cache.put('b', 2)  # expires at 10.0
    cache.put('a', 99, ttl=20)  # updates 'a', expires at 20.0, moves to head
    
    # At time 15.0, 'b' should be expired, 'a' should be valid
    assert cache.get('b') is None
    assert cache.get('a') == 99

def test_delete():
    """Test explicit removal of a key."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_size_and_capacity_enforcement(mock_time):
    """Test size tracking and capacity limits with mixed operations."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    
    cache.put('d', 4)  # Evicts 'a'
    assert cache.size() == 3
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.get('d') == 4