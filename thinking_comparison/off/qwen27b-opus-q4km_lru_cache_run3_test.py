from __future__ import annotations
from typing import Any, Optional
import time


class _Node:
    """Doubly-linked list node for cache entries."""
    
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Supports O(1) get, put, and delete operations with automatic
    eviction of least-recently-used items when at capacity.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cached items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")
        
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map: key -> _Node
        self._cache: dict[str, _Node] = {}
        
        # Doubly-linked list sentinels
        self._head: _Node  # Points to LRU item (will be initialized)
        self._tail: _Node  # Points to MRU item (will be initialized)
        self._init_sentinels()
        
        # Current count of valid (non-expired) items
        self._size = 0
    
    def _init_sentinels(self) -> None:
        """Initialize head and tail sentinel nodes."""
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expires_at
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def _add_to_tail(self, node: _Node) -> None:
        """Add a node to the tail (most recently used position)."""
        node.prev = self._tail.prev
        node.next = self._tail
        if self._tail.prev:
            self._tail.prev.next = node
        self._tail.prev = node
    
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        current = self._head.next
        while current != self._tail:
            if self._is_expired(current):
                # Remove expired item
                self._remove_node(current)
                del self._cache[current.key]
                current = current.next
            else:
                # Found non-expired LRU item
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                return
        # All items expired - clear everything
        self._clear_all()
    
    def _clear_all(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        self._init_sentinels()
        self._size = 0
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items (called during eviction)."""
        current = self._head.next
        while current != self._tail:
            if self._is_expired(current):
                next_node = current.next
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                current = next_node
            else:
                current = current.next
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for a key if it exists and is not expired.
        
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, None otherwise.
        """
        node = self._cache.get(key)
        if node is None:
            return None
        
        if self._is_expired(node):
            # Remove expired item
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        
        # Move to tail (most recently used)
        self._remove_node(node)
        self._add_to_tail(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds (overrides default).
        """
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            # Move to tail (most recently used)
            self._remove_node(node)
            self._add_to_tail(node)
            return
        
        # New key - check capacity
        if self._size >= self._capacity:
            self._evict_lru()
        
        # Insert new node
        node = _Node(key, value, expires_at)
        self._cache[key] = node
        self._add_to_tail(node)
        self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        node = self._cache.get(key)
        if node is None:
            return False
        
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Uses lazy cleanup - expired items are removed on access.
        
        Returns:
            Number of non-expired items.
        """
        return self._size

import pytest
from unittest.mock import patch
from typing import Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




class MockTime:
    """Mock time provider for deterministic testing."""
    def __init__(self, start: float = 0.0):
        self._current = start
    
    def monotonic(self) -> float:
        return self._current
    
    def advance(self, seconds: float) -> None:
        self._current += seconds


def test_basic_get_put():
    """Test basic get and put operations."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # Put items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") is None  # Non-existent key
        
        # Update existing key
        cache.put("key1", "updated_value")
        assert cache.get("key1") == "updated_value"


def test_capacity_eviction_lru_order():
    """Test that LRU items are evicted when capacity is reached."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        
        # Fill cache
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add new item - should evict 'b' (LRU)
        cache.put("d", 4)
        
        assert cache.get("a") == 1  # Still there (was accessed)
        assert cache.get("b") is None  # Evicted (LRU)
        assert cache.get("c") == 3  # Still there
        assert cache.get("d") == 4  # Newly added


def test_ttl_expiry():
    """Test that items expire after their TTL."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=3, default_ttl=5.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.advance(6.0)
        
        # Item should be expired
        assert cache.get("key1") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    """Test custom TTL per key overrides default."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # Item with custom TTL of 2 seconds
        cache.put("short_ttl", "value1", ttl=2.0)
        cache.put("default_ttl", "value2")
        
        # Advance 3 seconds - short TTL expired, default still valid
        mock_time.advance(3.0)
        
        assert cache.get("short_ttl") is None  # Expired
        assert cache.get("default_ttl") == "value2"  # Still valid


def test_delete_operation():
    """Test delete operation."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Delete existing key
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Delete non-existent key
        assert cache.delete("key1") is False
        assert cache.delete("nonexistent") is False
        
        # Verify remaining item
        assert cache.get("key2") == "value2"
        assert cache.size() == 1


def test_size_with_mixed_expired_valid():
    """Test size() returns count of non-expired items only."""
    mock_time = MockTime(0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1", ttl=2.0)  # Expires in 2s
        cache.put("key2", "value2", ttl=5.0)  # Expires in 5s
        cache.put("key3", "value3")            # Expires in 10s (default)
        
        assert cache.size() == 3
        
        # Advance 3 seconds - key1 expired
        mock_time.advance(3.0)
        
        # Access key1 - should be cleaned up
        assert cache.get("key1") is None
        assert cache.size() == 2  # key1 removed
        
        # Advance to 6 seconds - key2 also expired
        mock_time.advance(3.0)
        
        assert cache.get("key2") is None
        assert cache.size() == 1  # Only key3 remains
        
        # Advance to 11 seconds - all expired
        mock_time.advance(5.0)
        
        assert cache.get("key3") is None
        assert cache.size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])