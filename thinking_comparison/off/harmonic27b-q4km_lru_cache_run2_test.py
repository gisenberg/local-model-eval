import time
from typing import Any, Optional
from collections import deque


class _Node:
    """Doubly linked list node for LRU tracking."""
    def __init__(self, key: str, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Supports O(1) average time complexity for get, put, and delete operations.
    Uses a doubly-linked list for LRU ordering and a hash map for O(1) access.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for cached items
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        self._head = _Node("", None, 0)  # Dummy head
        self._tail = _Node("", None, 0)  # Dummy tail
        self._head.next = self._tail
        self._tail.prev = self._head
        
    def _add_to_head(self, node: _Node) -> None:
        """Add node to the head of the doubly linked list (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        
    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        
    def _evict_expired_items(self) -> None:
        """Remove all expired items from the cache."""
        current = self._head.next
        while current != self._tail:
            if current.expiry_time <= time.monotonic():
                next_node = current.next
                self._remove_node(current)
                del self._cache[current.key]
                current = next_node
            else:
                current = current.next
                
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # Find the first non-expired item from the tail
        current = self._tail.prev
        while current != self._head:
            if current.expiry_time > time.monotonic():
                self._remove_node(current)
                del self._cache[current.key]
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
        if node.expiry_time <= time.monotonic():
            self._remove_node(node)
            del self._cache[key]
            return None
            
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl
        
        # First, evict all expired items
        self._evict_expired_items()
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.expiry_time = expiry_time
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key
            if len(self._cache) >= self.capacity:
                self._evict_lru()
            
            node = _Node(key, value, expiry_time)
            self._cache[key] = node
            self._add_to_head(node)
    
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
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items
        """
        self._evict_expired_items()
        return len(self._cache)


# Test suite
import pytest
from unittest.mock import patch


def test_basic_get_put():
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None
    
    cache.put("key2", "value2")
    assert cache.get("key2") == "value2"
    assert cache.size() == 2


def test_capacity_eviction_lru_order():
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    # Add 3 items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Access key1 to make it most recently used
    cache.get("key1")
    
    # Add 4th item - should evict key2 (LRU)
    cache.put("key4", "value4")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None  # Should be evicted
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"
    assert cache.size() == 3


def test_ttl_expiry():
    """Test that items expire after their TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("key1", "value1")
        
        # Should exist at time 0
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        assert cache.get("key1") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    """Test that custom TTL per key works correctly."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # Add item with custom TTL of 3 seconds
        cache.put("key1", "value1", ttl=3.0)
        cache.put("key2", "value2")  # Uses default TTL of 10
        
        # Advance time to 4 seconds
        mock_time.return_value = 4.0
        
        # key1 should be expired, key2 should still be valid
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.size() == 1


def test_delete_operation():
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("key1") is False  # Already deleted
    
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    assert cache.delete("key2") is True
    assert cache.size() == 1


def test_size_with_mixed_expired_valid():
    """Test size() with mixed expired and valid items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        # Add items with different TTLs
        cache.put("key1", "value1", ttl=2.0)  # Expires at t=2
        cache.put("key2", "value2", ttl=10.0)  # Expires at t=10
        cache.put("key3", "value3", ttl=3.0)   # Expires at t=3
        
        # At t=0, all should be valid
        assert cache.size() == 3
        
        # Advance to t=2.5 - key1 should be expired
        mock_time.return_value = 2.5
        assert cache.size() == 2  # Only key2 and key3 valid
        
        # Advance to t=3.5 - key3 should also be expired
        mock_time.return_value = 3.5
        assert cache.size() == 1  # Only key2 valid
        
        # Advance to t=11 - all should be expired
        mock_time.return_value = 11.0
        assert cache.size() == 0