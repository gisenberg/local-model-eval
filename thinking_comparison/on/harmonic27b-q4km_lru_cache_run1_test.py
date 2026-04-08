from typing import Any, Optional
import time

class _Node:
    """Doubly-linked list node for cache entries."""
    def __init__(self, key: str, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev = None
        self.next = None

class TTLCache:
    """LRU cache with time-based expiration using doubly-linked list + hash map."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items
            default_ttl: Default time-to-live in seconds
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache = {}  # key -> _Node
        # Dummy head and tail for doubly-linked list
        self._head = _Node("", None, 0)
        self._tail = _Node("", None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expiry_time
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node: _Node) -> None:
        """Add node right after head (most recently used)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from cache."""
        expired_keys = [key for key, node in self._cache.items() 
                       if self._is_expired(node)]
        for key in expired_keys:
            node = self._cache.pop(key)
            self._remove_node(node)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.
        Accessing makes it most-recently-used.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None
        
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        Evicts least-recently-used non-expired item if at capacity.
        """
        # Clean up expired items first
        self._cleanup_expired()
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.expiry_time = time.monotonic() + (ttl or self.default_ttl)
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key
            if len(self._cache) >= self.capacity:
                # Evict least recently used (node before tail)
                lru_node = self._tail.prev
                if lru_node != self._head:  # Make sure it's not the dummy head
                    self._remove_node(lru_node)
                    del self._cache[lru_node.key]
            
            new_node = _Node(key, value, time.monotonic() + (ttl or self.default_ttl))
            self._cache[key] = new_node
            self._add_to_head(new_node)
    
    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self._cache:
            return False
        
        node = self._cache.pop(key)
        self._remove_node(node)
        return True
    
    def size(self) -> int:
        """Return count of non-expired items."""
        self._cleanup_expired()
        return len(self._cache)

import pytest
from unittest.mock import patch

def test_basic_get_put():
    """Test basic get/put operations."""
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1
    assert cache.get("nonexistent") is None

def test_capacity_eviction():
    """Test LRU eviction when capacity is reached."""
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" (least recently used)
    
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test time-based expiration."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(3, 1.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        with patch('time.monotonic', return_value=2.0):
            assert cache.get("a") is None  # Expired
            assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test custom TTL per key."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(3, 10.0)
        cache.put("a", 1, ttl=1.0)
        cache.put("b", 2, ttl=5.0)
        
        with patch('time.monotonic', return_value=2.0):
            assert cache.get("a") is None  # Expired (1s TTL)
            assert cache.get("b") == 2  # Still valid (5s TTL)
            assert cache.size() == 1

def test_delete():
    """Test delete operation."""
    cache = TTLCache(3, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") == True
    assert cache.delete("a") == False  # Already deleted
    assert cache.get("a") is None

def test_size_with_mixed_expired_valid():
    """Test size with mixed expired and valid items."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(3, 1.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3, ttl=5.0)
        
        with patch('time.monotonic', return_value=2.0):
            assert cache.size() == 1  # Only "c" is valid
            assert cache.get("c") == 3
            assert cache.get("a") is None
            assert cache.get("b") is None