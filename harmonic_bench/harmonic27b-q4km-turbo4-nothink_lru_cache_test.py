from __future__ import annotations
from typing import Any, Optional
import time


class _Node:
    """Doubly linked list node for LRU tracking."""
    __slots__ = ("key", "value", "expiry", "prev", "next")
    
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with time-based expiration using doubly-linked list + hash map."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live in seconds for items
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: dict[str, _Node] = {}
        self._head: _Node = _Node("", None, 0)  # Dummy head
        self._tail: _Node = _Node("", None, 0)  # Dummy tail
        self._head.next = self._tail
        self._tail.prev = self._head
        
    def _add_to_head(self, node: _Node) -> None:
        """Add node right after head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        
    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        
    def _evict_expired(self) -> None:
        """Remove all expired items from the cache."""
        current = self._head.next
        while current != self._tail:
            if current.expiry <= time.monotonic():
                next_node = current.next
                self._remove_node(current)
                del self._map[current.key]
                current = next_node
            else:
                current = current.next
                
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # First remove all expired items
        self._evict_expired()
        
        # If still at capacity, evict LRU
        if len(self._map) >= self._capacity:
            lru_node = self._tail.prev
            if lru_node != self._head:
                self._remove_node(lru_node)
                del self._map[lru_node.key]
                
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self._map:
            return None
            
        node = self._map[key]
        current_time = time.monotonic()
        
        if node.expiry <= current_time:
            # Item is expired, remove it
            self._remove_node(node)
            del self._map[key]
            return None
            
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value
        
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional custom TTL.
        If at capacity, evicts the least-recently-used non-expired item.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        current_time = time.monotonic()
        actual_ttl = ttl if ttl is not None else self._default_ttl
        expiry = current_time + actual_ttl
        
        if key in self._map:
            # Update existing key
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key
            if len(self._map) >= self._capacity:
                self._evict_lru()
                
            node = _Node(key, value, expiry)
            self._map[key] = node
            self._add_to_head(node)
            
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key existed and was removed, False otherwise
        """
        if key not in self._map:
            return False
            
        node = self._map[key]
        self._remove_node(node)
        del self._map[key]
        return True
        
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items
        """
        self._evict_expired()
        return len(self._map)

import pytest
from unittest.mock import patch
from typing import Any
import time

# Import the TTLCache class
  # Replace with actual module name


def test_basic_get_put():
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("nonexistent") is None


def test_capacity_eviction_lru_order():
    """Test that LRU eviction works correctly when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"


def test_ttl_expiry():
    """Test that items expire after their TTL."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=3, default_ttl=5.0)
        
        # Set initial time to 0
        mock_time.return_value = 0.0
        cache.put("key1", "value1")
        
        # Time advances to 4 seconds (still valid)
        mock_time.return_value = 4.0
        assert cache.get("key1") == "value1"
        
        # Time advances to 6 seconds (expired)
        mock_time.return_value = 6.0
        assert cache.get("key1") is None


def test_custom_per_key_ttl():
    """Test that custom TTL per key overrides default TTL."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        mock_time.return_value = 0.0
        cache.put("key1", "value1", ttl=2.0)  # Custom TTL of 2 seconds
        cache.put("key2", "value2")  # Uses default TTL of 10 seconds
        
        # At time 3, key1 should be expired but key2 should still be valid
        mock_time.return_value = 3.0
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"


def test_delete_operation():
    """Test delete operation returns correct boolean values."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    assert cache.delete("nonexistent") is False
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.delete("key1") is False  # Already deleted


def test_size_with_mixed_expired_valid():
    """Test size() returns count of only non-expired items."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        mock_time.return_value = 0.0
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        assert cache.size() == 3
        
        # Advance time to expire key1 and key2
        mock_time.return_value = 6.0
        
        # Size should only count non-expired items
        assert cache.size() == 1  # Only key3 remains valid