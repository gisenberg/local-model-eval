import time
from typing import Any, Optional, Dict, List
from collections import deque


class _Node:
    """Doubly linked list node for LRU tracking."""
    __slots__ = ('key', 'value', 'ttl', 'expire_time', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, ttl: float, expire_time: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expire_time = expire_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU cache with time-based expiration."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for items
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # Most recently used
        self._tail: Optional[_Node] = None  # Least recently used
        self._size = 0
    
    def _add_to_head(self, node: _Node) -> None:
        """Add node to the head (most recently used position)."""
        if self._head is None:
            self._head = self._tail = node
        else:
            node.next = self._head
            self._head.prev = node
            self._head = node
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
            
        node.prev = node.next = None
    
    def _evict_expired_items(self) -> None:
        """Remove all expired items from the cache."""
        current = self._head
        while current:
            if current.expire_time <= time.monotonic():
                next_node = current.next
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                current = next_node
            else:
                break
    
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # First remove any expired items
        self._evict_expired_items()
        
        if self._size >= self.capacity:
            # Find the first non-expired item from the tail
            current = self._tail
            while current:
                if current.expire_time > time.monotonic():
                    self._remove_node(current)
                    del self._cache[current.key]
                    self._size -= 1
                    return
                current = current.prev
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        # Check if expired
        if node.expire_time <= time.monotonic():
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        if ttl is None:
            ttl = self.default_ttl
            
        expire_time = time.monotonic() + ttl
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.ttl = ttl
            node.expire_time = expire_time
            
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key
            if self._size >= self.capacity:
                self._evict_lru()
            
            node = _Node(key, value, ttl, expire_time)
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key existed and was removed, False otherwise
        """
        if key not in self._cache:
            return False
            
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items
        """
        self._evict_expired_items()
        return self._size

import pytest
from unittest.mock import patch
from typing import Any, Optional


class TestTTLCache:
    """Test suite for TTLCache implementation."""
    
    def test_basic_get_put(self):
        """Test basic get and put operations."""
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        assert cache.size() == 2
    
    def test_capacity_eviction_lru_order(self):
        """Test that LRU eviction works correctly when capacity is reached."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add third item - should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.size() == 2
    
    def test_ttl_expiry(self):
        """Test that items expire after their TTL."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=3, default_ttl=5.0)
            cache.put("key1", "value1")
            
            # Before expiry
            mock_time.return_value = 4.0
            assert cache.get("key1") == "value1"
            
            # After expiry
            mock_time.return_value = 6.0
            assert cache.get("key1") is None
            assert cache.size() == 0
    
    def test_custom_per_key_ttl(self):
        """Test that custom TTL per key works correctly."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=3, default_ttl=10.0)
            cache.put("key1", "value1", ttl=3.0)  # Custom TTL
            cache.put("key2", "value2")  # Default TTL
            
            # key1 should expire first
            mock_time.return_value = 4.0
            assert cache.get("key1") is None
            assert cache.get("key2") == "value2"
            
            # key2 should still be valid
            mock_time.return_value = 9.0
            assert cache.get("key2") == "value2"
            
            # key2 should expire
            mock_time.return_value = 11.0
            assert cache.get("key2") is None
    
    def test_delete_operation(self):
        """Test delete operation."""
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False  # Already deleted
        assert cache.get("key2") == "value2"
        assert cache.size() == 1
    
    def test_size_with_mixed_expired_valid_items(self):
        """Test size() with mixed expired and valid items."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=5, default_ttl=5.0)
            cache.put("key1", "value1", ttl=3.0)  # Will expire
            cache.put("key2", "value2", ttl=10.0)  # Will stay valid
            cache.put("key3", "value3", ttl=2.0)   # Will expire
            
            # Before any expiry
            assert cache.size() == 3
            
            # After some items expire
            mock_time.return_value = 4.0
            assert cache.size() == 1  # Only key2 should remain
            
            # After all items expire
            mock_time.return_value = 11.0
            assert cache.size() == 0