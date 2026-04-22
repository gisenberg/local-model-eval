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
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map + custom doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are evicted on access or insertion,
    rather than via background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for easier list manipulation
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._current_size = 0

    def _remove(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move existing node to MRU position."""
        self._remove(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove the least recently used node (right before tail)."""
        lru = self.tail.prev
        if lru is not self.head:
            self._remove(lru)
            del self.cache[lru.key]
            self._current_size -= 1

    def _is_expired(self, node: _Node) -> bool:
        """Check if node's TTL has elapsed."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key missing or expired.
        Moves accessed key to MRU position. O(1) avg time.
        """
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup on access
            self._remove(node)
            del self.cache[key]
            self._current_size -= 1
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair. If ttl is None, uses default_ttl.
        Evicts LRU if capacity exceeded. O(1) avg time.
        """
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Treat expired existing key as a new insertion
                self._remove(node)
                del self.cache[key]
                self._current_size -= 1
            else:
                node.value = value
                node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_head(node)
                return

        # Evict if at capacity
        if self._current_size >= self.capacity:
            self._evict_lru()

        # Insert new node
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = _Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_head(new_node)
        self._current_size += 1

    def delete(self, key: Any) -> None:
        """Remove key from cache. No-op if key doesn't exist. O(1) avg time."""
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]
            self._current_size -= 1

    def size(self) -> int:
        """Return current number of valid entries in cache. O(1) time."""
        return self._current_size

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that expired entries return None and are lazily cleaned up."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # TTL exceeded
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
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
def test_custom_ttl_override(mock_time):
    """Test that custom ttl overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    
    cache.put('a', 1, ttl=2.0)
    mock_time.return_value = 3.0
    assert cache.get('a') is None  # Custom TTL expired
    
    mock_time.return_value = 0.0
    cache.put('b', 2)  # Uses default 10.0
    mock_time.return_value = 5.0
    assert cache.get('b') == 2  # Default TTL still valid

@patch('time.monotonic')
def test_delete_functionality(mock_time):
    """Test explicit deletion and idempotent delete."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    
    cache.delete('nonexistent')  # Should not raise
    assert cache.size() == 1

@patch('time.monotonic')
def test_size_tracking_with_expiration_and_eviction(mock_time):
    """Test that size() accurately reflects valid entries after mixed operations."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    
    mock_time.return_value = 6.0
    cache.get('a')  # Expires and removes 'a'
    assert cache.size() == 1
    
    cache.put('c', 3)  # Evicts 'b' (LRU)
    assert cache.size() == 2
    assert cache.get('b') is None
    assert cache.get('c') == 3