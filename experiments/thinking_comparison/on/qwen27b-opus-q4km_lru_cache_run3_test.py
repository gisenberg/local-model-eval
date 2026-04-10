import time
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a cache entry in the doubly-linked list."""
    key: str
    value: Any
    expiration: float  # time.monotonic() value when item expires
    prev: Optional['Node'] = None
    next: Optional['Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Supports O(1) get/put/delete operations with configurable capacity
    and per-item or default TTL (time-to-live).
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
            raise ValueError("Default TTL must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map: key -> Node
        self._cache: dict[str, Node] = {}
        
        # Doubly-linked list sentinels
        self._head: Node  # Points to MRU (most recently used)
        self._tail: Node  # Points to LRU (least recently used)
        self._init_sentinels()
        
        # Track actual number of items (may include expired)
        self._size = 0
    
    def _init_sentinels(self) -> None:
        """Initialize head and tail sentinel nodes."""
        self._head = Node(key="", value=None, expiration=0.0)
        self._tail = Node(key="", value=None, expiration=0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() > node.expiration
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        node.prev = None
        node.next = None
    
    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
    
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # Find LRU non-expired node (traverse from tail)
        current = self._tail.prev
        while current != self._head:
            if not self._is_expired(current):
                # Found LRU non-expired item
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                return
            current = current.prev
        
        # All items are expired - clear them all
        self._clear_all()
    
    def _clear_all(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        self._init_sentinels()
        self._size = 0
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items (lazy cleanup)."""
        expired_keys = [key for key, node in self._cache.items() 
                       if self._is_expired(node)]
        for key in expired_keys:
            node = self._cache.pop(key)
            self._remove_node(node)
            self._size -= 1
    
    def _get_or_create_node(self, key: str, value: Any, ttl: float) -> Node:
        """Get existing node or create a new one."""
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration = time.monotonic() + ttl
            return node
        return Node(key=key, value=value, expiration=time.monotonic() + ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for a key if it exists and is not expired.
        
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        # Check if expired
        if self._is_expired(node):
            # Remove expired item
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Overrides default_ttl if provided.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        current_time = time.monotonic()
        
        # Check if key exists
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration = current_time + effective_ttl
            # Move to head (MRU)
            self._remove_node(node)
            self._add_to_head(node)
            return
        
        # Check if at capacity
        if self._size >= self._capacity:
            self._evict_lru()
        
        # Create new node and add to cache
        node = Node(key=key, value=value, expiration=current_time + effective_ttl)
        self._cache[key] = node
        self._add_to_head(node)
        self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key not in self._cache:
            return False
            
        node = self._cache.pop(key)
        self._remove_node(node)
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items.
        """
        self._cleanup_expired()
        return self._size

import pytest
from unittest.mock import patch
from typing import Any
import sys
sys.path.insert(0, '.')




@patch('ttl_cache.time.monotonic')
def test_basic_get_put(mock_monotonic: Any) -> None:
    """Test basic get and put operations."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    # Put items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Get items
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    
    # Non-existent key
    assert cache.get("key4") is None
    
    # Update existing key
    cache.put("key1", "updated_value1")
    assert cache.get("key1") == "updated_value1"


@patch('ttl_cache.time.monotonic')
def test_capacity_eviction_lru_order(mock_monotonic: Any) -> None:
    """Test that LRU item is evicted when at capacity."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    # Fill cache
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Access key1 to make it MRU
    cache.get("key1")
    
    # Add key3 - should evict key2 (LRU)
    cache.put("key3", "value3")
    
    assert cache.get("key1") == "value1"  # Still there (was accessed)
    assert cache.get("key2") is None      # Evicted (LRU)
    assert cache.get("key3") == "value3"  # Just added


@patch('ttl_cache.time.monotonic')
def test_ttl_expiry(mock_monotonic: Any) -> None:
    """Test that expired items return None."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=5.0)
    
    cache.put("key1", "value1")
    
    # Item should be valid
    assert cache.get("key1") == "value1"
    
    # Advance time past TTL
    mock_monotonic.return_value = 6.0
    
    # Item should be expired
    assert cache.get("key1") is None


@patch('ttl_cache.time.monotonic')
def test_custom_per_key_ttl(mock_monotonic: Any) -> None:
    """Test custom TTL per key overrides default."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    # Put with custom TTL of 2 seconds
    cache.put("key1", "value1", ttl=2.0)
    
    # Advance time to 3 seconds - custom TTL expired
    mock_monotonic.return_value = 3.0
    assert cache.get("key1") is None
    
    # Put another item with default TTL
    cache.put("key2", "value2")
    
    # Advance time to 5 seconds - default TTL still valid
    mock_monotonic.return_value = 5.0
    assert cache.get("key2") == "value2"
    
    # Advance time to 11 seconds - default TTL expired
    mock_monotonic.return_value = 11.0
    assert cache.get("key2") is None


@patch('ttl_cache.time.monotonic')
def test_delete(mock_monotonic: Any) -> None:
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Delete existing key
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    # Delete non-existent key
    assert cache.delete("key3") is False
    
    # Remaining item still accessible
    assert cache.get("key2") == "value2"


@patch('ttl_cache.time.monotonic')
def test_size_with_mixed_expired_valid(mock_monotonic: Any) -> None:
    """Test size() returns count of non-expired items only."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=5, default_ttl=5.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    assert cache.size() == 3
    
    # Advance time - key1 and key2 expire (TTL=5, time=6)
    mock_monotonic.return_value = 6.0
    
    # size() should perform lazy cleanup and return 1
    assert cache.size() == 1
    
    # Verify key3 is still valid
    assert cache.get("key3") == "value3"
    
    # Verify key1 and key2 are gone
    assert cache.get("key1") is None
    assert cache.get("key2") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])