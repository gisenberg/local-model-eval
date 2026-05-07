import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'exp_time', 'prev', 'next')
    
    def __init__(self, key: Any, value: Any, exp_time: float) -> None:
        self.key = key
        self.value = value
        self.exp_time = exp_time
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU Cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy expiration cleanup (only checks TTL on access).
    """
    
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for easier list manipulation
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_tail(self, node: _Node) -> None:
        """Add node right before tail (MRU position)."""
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a specific node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_tail(self, node: _Node) -> None:
        """Move existing node to MRU position."""
        self._remove_node(node)
        self._add_to_tail(node)

    def _remove_lru(self) -> _Node:
        """Remove and return the LRU node (right after head)."""
        lru_node = self.head.next
        self._remove_node(lru_node)
        return lru_node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.exp_time

    def get(self, key: Any) -> Any:
        """Retrieve value by key. Returns None if key is missing or expired."""
        node = self.cache.get(key)
        if node is None:
            return None

        # Lazy cleanup: check expiration only on access
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair with optional TTL."""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.exp_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_tail(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU
                lru = self._remove_lru()
                del self.cache[lru.key]

            exp_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            new_node = _Node(key, value, exp_time)
            self.cache[key] = new_node
            self._add_to_tail(new_node)

    def delete(self, key: Any) -> bool:
        """Remove key from cache. Returns True if key existed, False otherwise."""
        node = self.cache.pop(key, None)
        if node is not None:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return current number of items in cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired items are cleaned up lazily on get()."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    # Advance time past TTL
    mock_time.return_value = 6.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    mock_time.return_value = 1.0
    cache.put('b', 2)
    mock_time.return_value = 2.0
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    """Test updating value and refreshing TTL for existing key."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 3.0
    cache.put('a', 10, ttl=10.0)  # Update value & extend TTL
    
    mock_time.return_value = 6.0  # Past original TTL, but within new TTL
    assert cache.get('a') == 10
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion and return values."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    assert cache.delete('a') is True
    assert cache.delete('nonexistent') is False
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_size_and_capacity_limits(mock_time):
    """Test size tracking and hard capacity enforcement."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    assert cache.size() == 0
    
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    
    cache.put('d', 4)  # Evicts 'a'
    assert cache.size() == 3
    assert cache.get('a') is None