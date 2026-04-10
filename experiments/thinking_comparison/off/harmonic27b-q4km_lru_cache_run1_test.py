from __future__ import annotations
from typing import Any, Optional
import time
from collections import deque


class _Node:
    """Doubly linked list node for LRU tracking."""
    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with time-based expiration."""

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for items
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map for O(1) access
        self._cache: dict[str, _Node] = {}
        
        # Doubly linked list for LRU ordering
        # head is dummy, tail is dummy
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        
        self._current_time = 0.0

    def _get_current_time(self) -> float:
        """Get current time using monotonic clock."""
        return time.monotonic()

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        current_time = self._get_current_time()
        expired_keys = []
        
        for key, node in self._cache.items():
            if node.expires_at <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            node = self._cache.pop(key)
            self._remove_node(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        
        Accessing a key makes it most-recently-used.
        Returns None if key doesn't exist or is expired.
        """
        self._cleanup_expired()
        
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        # Double-check expiration after cleanup
        if node.expires_at <= self._get_current_time():
            del self._cache[key]
            self._remove_node(node)
            return None
            
        # Move to most recently used position
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If at capacity, evicts the least-recently-used non-expired item.
        Custom TTL overrides default TTL.
        """
        self._cleanup_expired()
        
        current_time = self._get_current_time()
        expires_at = current_time + (ttl if ttl is not None else self._default_ttl)
        
        if key in self._cache:
            # Update existing entry
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new entry
            if len(self._cache) >= self._capacity:
                # Evict least recently used (node before tail)
                lru_node = self._tail.prev
                if lru_node != self._head:
                    del self._cache[lru_node.key]
                    self._remove_node(lru_node)
            
            new_node = _Node(key, value, expires_at)
            self._cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed, False otherwise.
        """
        self._cleanup_expired()
        
        if key not in self._cache:
            return False
            
        node = self._cache.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        """
        self._cleanup_expired()
        return len(self._cache)

# test_ttl_cache.py
import pytest
from unittest.mock import patch
from typing import Any
import time

# Import the TTLCache class
  # Replace with actual import


@pytest.fixture
def mock_time():
    """Fixture to provide mocked time.monotonic for deterministic testing."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 0.0
        yield mock_monotonic


def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None
    
    cache.put("key2", "value2")
    assert cache.get("key2") == "value2"
    assert cache.size() == 2


def test_capacity_eviction_lru_order(mock_time):
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    # Fill cache to capacity
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Access key1 to make it most recently used
    cache.get("key1")
    
    # Add new key - should evict key2 (least recently used)
    cache.put("key4", "value4")
    
    assert cache.get("key1") == "value1"  # Still exists
    assert cache.get("key2") is None       # Evicted
    assert cache.get("key3") == "value3"   # Still exists
    assert cache.get("key4") == "value4"   # New entry
    assert cache.size() == 3


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
    """Test that custom TTL per key overrides default TTL."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=2.0)  # Custom TTL of 2 seconds
    cache.put("key2", "value2")           # Default TTL of 10 seconds
    
    # Advance time to 3 seconds
    mock_time.return_value = 3.0
    
    assert cache.get("key1") is None  # Expired (2 < 3)
    assert cache.get("key2") == "value2"  # Still valid (10 > 3)
    assert cache.size() == 1


def test_delete_operation(mock_time):
    """Test delete operation and return values."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    assert cache.delete("nonexistent") is False
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.size() == 0


def test_size_with_mixed_expired_valid(mock_time):
    """Test size() returns count of only non-expired items."""
    cache = TTLCache(capacity=5, default_ttl=5.0)
    
    # Add items with different TTLs
    cache.put("key1", "value1", ttl=2.0)
    cache.put("key2", "value2", ttl=10.0)
    cache.put("key3", "value3", ttl=3.0)
    
    assert cache.size() == 3
    
    # Advance time to 4 seconds
    mock_time.return_value = 4.0
    
    # key1 and key3 should be expired, only key2 should remain
    assert cache.size() == 1
    assert cache.get("key2") == "value2"
    assert cache.get("key1") is None
    assert cache.get("key3") is None