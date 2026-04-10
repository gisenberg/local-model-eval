from typing import Any, Optional, Dict
import time


class _Node:
    """Internal doubly-linked list node for LRU cache."""
    __slots__ = ('key', 'value', 'ttl', 'expires_at', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, ttl: float, expires_at: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU cache with time-based expiration (TTL).
    
    Supports O(1) average time complexity for get, put, and delete operations.
    Uses a doubly-linked list + hash map internally.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: Dict[str, _Node] = {}
        
        # Dummy head and tail for doubly-linked list
        self._head = _Node('', None, 0, 0)
        self._tail = _Node('', None, 0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _add_to_front(self, node: _Node) -> None:
        """Add node to front of list (most recently used)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
    
    def _remove_node(self, node: _Node) -> None:
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to front of list."""
        self._remove_node(node)
        self._add_to_front(node)
    
    def _evict_lru(self) -> None:
        """Evict least-recently-used non-expired item.
        
        If all items are expired, clears all of them first.
        """
        current_time = time.monotonic()
        node = self._tail.prev
        
        while node != self._head:
            if node.expires_at > current_time:
                # Found non-expired LRU item, evict it
                self._remove_node(node)
                del self._cache[node.key]
                return
            else:
                # Expired item, remove and continue to next LRU
                self._remove_node(node)
                del self._cache[node.key]
                node = self._tail.prev
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key.
        
        Returns value if key exists and not expired, otherwise None.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        current_time = time.monotonic()
        
        if node.expires_at <= current_time:
            # Expired, remove from cache
            self._remove_node(node)
            del self._cache[key]
            return None
            
        # Move to front (MRU)
        self._move_to_front(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key-value pair.
        
        If key exists and is expired, removes it first.
        If at capacity, evicts LRU non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Defaults to default_ttl.
        """
        current_time = time.monotonic()
        actual_ttl = ttl if ttl is not None else self._default_ttl
        
        if key in self._cache:
            node = self._cache[key]
            if node.expires_at <= current_time:
                # Expired, remove and treat as new insertion
                self._remove_node(node)
                del self._cache[key]
            else:
                # Update existing non-expired item
                node.value = value
                node.ttl = actual_ttl
                node.expires_at = current_time + actual_ttl
                self._move_to_front(node)
                return
        
        # Check capacity before inserting new item
        if len(self._cache) >= self._capacity:
            self._evict_lru()
        
        # Insert new node
        node = _Node(key, value, actual_ttl, current_time + actual_ttl)
        self._add_to_front(node)
        self._cache[key] = node
    
    def delete(self, key: str) -> bool:
        """Remove key from cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if key existed and was removed, False otherwise.
        """
        if key not in self._cache:
            return False
            
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        return True
    
    def size(self) -> int:
        """Return count of non-expired items.
        
        Performs lazy cleanup check on all items.
        
        Returns:
            Number of non-expired items in cache.
        """
        current_time = time.monotonic()
        count = 0
        for node in self._cache.values():
            if node.expires_at > current_time:
                count += 1
        return count


# ==================== TESTS ====================

import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_get_put(mock_time):
    """Test basic put and get operations."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.get('b') is None
    
    cache.put('b', 2)
    assert cache.get('b') == 2
    assert cache.get('a') == 1  # Still accessible


@patch('time.monotonic')
def test_capacity_eviction(mock_time):
    """Test LRU eviction when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None  # Evicted
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    
    # Access 'b' to make it MRU
    cache.get('b')
    cache.put('d', 4)  # Should evict 'c' (now LRU)
    
    assert cache.get('c') is None
    assert cache.get('b') == 2
    assert cache.get('d') == 4


@patch('time.monotonic')
def test_ttl_expiry(mock_time):
    """Test that items expire after TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 1.0)
    
    cache.put('a', 1)
    assert cache.get('a') == 1
    
    mock_time.return_value = 2.0  # Expired
    assert cache.get('a') is None
    
    # Verify it's actually removed
    assert cache.size() == 0


@patch('time.monotonic')
def test_custom_per_key_ttl(mock_time):
    """Test custom TTL per key overrides default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    
    cache.put('a', 1, ttl=1.0)  # Custom short TTL
    cache.put('b', 2)           # Default long TTL
    
    mock_time.return_value = 2.0
    
    assert cache.get('a') is None  # Expired (custom TTL)
    assert cache.get('b') == 2     # Not expired (default TTL)
    
    # Verify size counts only non-expired
    assert cache.size() == 1


@patch('time.monotonic')
def test_delete(mock_time):
    """Test delete operation."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    
    cache.put('a', 1)
    assert cache.delete('a') is True
    assert cache.get('a') is None
    assert cache.delete('a') is False  # Already deleted
    
    # Delete expired item
    cache.put('b', 2, ttl=1.0)
    mock_time.return_value = 2.0
    assert cache.delete('b') is True  # Should still return True even if expired


@patch('time.monotonic')
def test_size_mixed_expired_valid(mock_time):
    """Test size() returns count of non-expired items only."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    
    cache.put('a', 1, ttl=1.0)  # Will expire
    cache.put('b', 2, ttl=10.0) # Valid
    cache.put('c', 3, ttl=1.0)  # Will expire
    
    mock_time.return_value = 2.0
    
    # 'a' and 'c' are expired, 'b' is valid
    assert cache.size() == 1
    
    # Access expired 'a' to trigger cleanup
    assert cache.get('a') is None
    
    # Size should still be 1 (only 'b' is valid, 'c' still expired but not accessed)
    assert cache.size() == 1
    
    # Access 'c' to clean it up
    assert cache.get('c') is None
    assert cache.size() == 1
    
    # Now add new item
    cache.put('d', 4)
    assert cache.size() == 2