import time
from typing import Any, Optional
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
    """
    LRU cache with time-based expiration.
    
    Supports O(1) average time complexity for get, put, and delete operations
    using a doubly-linked list and hash map internally.
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
        
        # Hash map for O(1) access
        self._cache: dict[str, _Node] = {}
        
        # Doubly linked list for LRU ordering
        self._head: Optional[_Node] = None  # Most recently used
        self._tail: Optional[_Node] = None  # Least recently used
        
        self._current_size = 0
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expire_time
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
            
        node.prev = None
        node.next = None
    
    def _add_to_head(self, node: _Node) -> None:
        """Add a node to the head of the list (most recently used)."""
        node.prev = None
        node.next = self._head
        
        if self._head:
            self._head.prev = node
            
        self._head = node
        
        if not self._tail:
            self._tail = node
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        keys_to_remove = []
        
        for key, node in self._cache.items():
            if self._is_expired(node):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._current_size -= 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for a key if it exists and is not expired.
        
        Accessing a key makes it the most-recently-used item.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        if self._is_expired(node):
            # Remove expired item
            self._remove_node(node)
            del self._cache[key]
            self._current_size -= 1
            return None
        
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If the cache is at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default_ttl)
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_time = time.monotonic() + effective_ttl
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.ttl = effective_ttl
            node.expire_time = expire_time
            
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Create new node
            node = _Node(key, value, effective_ttl, expire_time)
            
            # Check if we need to evict
            if self._current_size >= self.capacity:
                # First, clean up any expired items
                self._cleanup_expired()
                
                # If still at capacity, evict LRU
                if self._current_size >= self.capacity and self._tail:
                    lru_node = self._tail
                    self._remove_node(lru_node)
                    del self._cache[lru_node.key]
                    self._current_size -= 1
            
            # Add new node
            self._cache[key] = node
            self._add_to_head(node)
            self._current_size += 1
    
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
        self._current_size -= 1
        
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items
        """
        self._cleanup_expired()
        return self._current_size

import pytest
from unittest.mock import patch
from typing import Any

# Import the TTLCache implementation
  # Replace with actual import path

@pytest.fixture
def mock_time():
    """Fixture to provide mock time.monotonic for deterministic testing."""
    with patch('your_module.time.monotonic') as mock_monotonic:  # Replace with actual module name
        mock_monotonic.return_value = 0.0
        yield mock_monotonic

def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    cache.put("key2", "value2")
    assert cache.get("key2") == "value2"
    
    assert cache.get("nonexistent") is None

def test_capacity_eviction_lru_order(mock_time):
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = TTLCache(capacity=3, default_ttl=100.0)
    
    # Add 3 items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Access key1 to make it most recently used
    cache.get("key1")
    
    # Add 4th item - should evict key2 (least recently used)
    cache.put("key4", "value4")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None  # Should be evicted
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"

def test_ttl_expiry(mock_time):
    """Test that items expire after their TTL."""
    cache = TTLCache(capacity=3, default_ttl=5.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Advance time past TTL
    mock_time.return_value = 6.0
    
    assert cache.get("key1") is None  # Should be expired
    assert cache.size() == 0

def test_custom_per_key_ttl(mock_time):
    """Test custom TTL per key overrides default."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=2.0)  # Custom TTL of 2 seconds
    cache.put("key2", "value2")  # Uses default TTL of 10 seconds
    
    # Advance time to 3 seconds
    mock_time.return_value = 3.0
    
    assert cache.get("key1") is None  # Expired (2s TTL)
    assert cache.get("key2") == "value2"  # Still valid (10s TTL)

def test_delete_operation(mock_time):
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.delete("key1") == True
    assert cache.get("key1") is None
    
    assert cache.delete("nonexistent") == False

def test_size_with_mixed_expired_valid_items(mock_time):
    """Test size() returns count of non-expired items only."""
    cache = TTLCache(capacity=5, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=2.0)  # Expires in 2s
    cache.put("key2", "value2", ttl=5.0)  # Expires in 5s
    cache.put("key3", "value3")  # Expires in 10s
    
    assert cache.size() == 3
    
    # Advance time to 3 seconds
    mock_time.return_value = 3.0
    
    # key1 should be expired, key2 and key3 should be valid
    assert cache.size() == 2
    
    # Advance time to 6 seconds
    mock_time.return_value = 6.0
    
    # Only key3 should be valid now
    assert cache.size() == 1