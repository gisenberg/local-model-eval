from typing import Any, Optional
import time

class Node:
    """Node in doubly-linked list for LRU cache."""
    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """LRU cache with time-based expiration."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, Node] = {}
        
        # Doubly-linked list sentinels
        self.head = Node("", None, 0)  # Most recently used
        self.tail = Node("", None, 0)  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _is_expired(self, node: Node) -> bool:
        """Check if a node is expired."""
        return time.monotonic() >= node.expires_at
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node: Node) -> None:
        """Add a node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from cache."""
        expired_keys = [key for key, node in self.cache.items() if self._is_expired(node)]
        for key in expired_keys:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        # Check if expired
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to most recently used position
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The key
            value: The value to store
            ttl: Optional custom TTL in seconds (uses default_ttl if None)
        """
        # First, clean up expired items
        self._cleanup_expired()
        
        # If key exists, update it
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._remove_node(node)
            self._add_to_head(node)
            return
        
        # If at capacity, evict LRU item
        if len(self.cache) >= self.capacity:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache[lru_node.key]
        
        # Add new item
        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expires_at)
        self.cache[key] = new_node
        self._add_to_head(new_node)
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if key existed and was removed, False otherwise
        """
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True
    
    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.
        
        Returns:
            Count of non-expired items
        """
        # Lazy cleanup - remove expired items
        self._cleanup_expired()
        return len(self.cache)

import pytest
from unittest.mock import patch

# Import TTLCache from the implementation above

@patch('time.monotonic')
def test_basic_get_put(mock_monotonic):
    """Test basic get and put operations."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None

@patch('time.monotonic')
def test_capacity_eviction(mock_monotonic):
    """Test LRU eviction when capacity is reached."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

@patch('time.monotonic')
def test_ttl_expiry(mock_monotonic):
    """Test time-based expiration."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=5.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Advance time past TTL
    mock_monotonic.return_value = 10.0
    assert cache.get("key1") is None  # Expired

@patch('time.monotonic')
def test_custom_ttl(mock_monotonic):
    """Test custom per-key TTL."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=2.0)  # Custom TTL of 2 seconds
    cache.put("key2", "value2")  # Uses default TTL of 10 seconds
    
    # Advance time to 3 seconds
    mock_monotonic.return_value = 3.0
    assert cache.get("key1") is None  # Expired (2s TTL)
    assert cache.get("key2") == "value2"  # Still valid (10s TTL)

@patch('time.monotonic')
def test_delete_operation(mock_monotonic):
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("nonexistent") is False

@patch('time.monotonic')
def test_size_with_mixed_items(mock_monotonic):
    """Test size() with mixed expired and valid items."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=5, default_ttl=5.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Advance time to expire some items
    mock_monotonic.return_value = 10.0
    
    # size() should return count of non-expired items
    assert cache.size() == 0  # All items expired