from __future__ import annotations
from typing import Any, Optional
import time


class _Node:
    """Internal doubly-linked list node for cache entries."""
    
    __slots__ = ('key', 'value', 'next', 'prev', 'expiration_time')
    
    def __init__(
        self, 
        key: str, 
        value: Any, 
        expiration_time: float
    ):
        self.key = key
        self.value = value
        self.next: Optional[_Node] = None
        self.prev: Optional[_Node] = None
        self.expiration_time = expiration_time


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a doubly-linked list + hash map for O(1) operations.
    Expired items are lazily cleaned up on access.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cached items.
        """
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map: key -> Node
        self._cache: dict[str, _Node] = {}
        
        # Doubly-linked list sentinels
        self._head: _Node  # Most recently used (after sentinel)
        self._tail: _Node  # Least recently used (before sentinel)
        self._init_sentinels()
    
    def _init_sentinels(self) -> None:
        """Initialize head and tail sentinel nodes."""
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expiration_time
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list (O(1))."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.next = None
        node.prev = None
    
    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head sentinel (O(1))."""
        node.next = self._head.next
        node.prev = self._head
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node
    
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item (O(1) amortized)."""
        # Start from tail (LRU end) and find first non-expired node
        node = self._tail.prev
        while node and node != self._head:
            if self._is_expired(node):
                # Remove expired node
                self._remove_node(node)
                del self._cache[node.key]
                node = node.prev
            else:
                # Found LRU non-expired node, evict it
                self._remove_node(node)
                del self._cache[node.key]
                return
        # All items expired - clear everything
        self._clear_all()
    
    def _clear_all(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._init_sentinels()
    
    def _cleanup_expired(self) -> None:
        """Lazy cleanup of expired items (called on access)."""
        keys_to_remove = [
            key for key, node in self._cache.items() 
            if self._is_expired(node)
        ]
        for key in keys_to_remove:
            node = self._cache.pop(key)
            self._remove_node(node)
    
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
            self._remove_node(node)
            del self._cache[key]
            return None
        
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If the cache is at capacity, evicts the least-recently-used 
        non-expired item. If all items are expired, clears them first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds (overrides default).
        """
        if self._capacity == 0:
            return
            
        actual_ttl = ttl if ttl is not None else self._default_ttl
        expiration_time = time.monotonic() + actual_ttl
        
        # Check if key already exists
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration_time = expiration_time
            # Move to head (most recently used)
            self._remove_node(node)
            self._add_to_head(node)
            return
        
        # Check capacity - evict if needed
        if len(self._cache) >= self._capacity:
            self._evict_lru()
        
        # Insert new node
        node = _Node(key, value, expiration_time)
        self._cache[key] = node
        self._add_to_head(node)
    
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
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items.
        """
        self._cleanup_expired()
        return len(self._cache)

import pytest
from unittest.mock import patch
from typing import Any


@patch('time.monotonic')
def test_basic_get_put(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test basic get/put operations."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=3, default_ttl=10.0)
    
    # Put items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Get items
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") is None  # Non-existent key
    
    # Update existing key
    cache.put("key1", "updated_value1")
    assert cache.get("key1") == "updated_value1"


@patch('time.monotonic')
def test_capacity_eviction_lru_order(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test that LRU eviction works correctly when capacity is reached."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=3, default_ttl=100.0)
    
    # Fill cache to capacity
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Access key1 to make it MRU
    cache.get("key1")
    
    # Add new key - should evict key2 (LRU)
    cache.put("key4", "value4")
    
    assert cache.get("key1") == "value1"  # Still there (was accessed)
    assert cache.get("key2") is None      # Evicted (LRU)
    assert cache.get("key3") == "value3"  # Still there
    assert cache.get("key4") == "value4"  # Newly added


@patch('time.monotonic')
def test_ttl_expiry(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test that items expire after their TTL."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    
    # Item should be accessible before expiry
    assert cache.get("key1") == "value1"
    
    # Advance time past TTL
    mock_monotonic.return_value = 15.0
    
    # Item should be expired
    assert cache.get("key1") is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_custom_per_key_ttl(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test custom TTL per key overrides default."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=3, default_ttl=100.0)
    
    # Put with custom TTL of 5 seconds
    cache.put("key1", "value1", ttl=5.0)
    cache.put("key2", "value2")  # Uses default TTL of 100
    
    # Advance time to 10 seconds
    mock_monotonic.return_value = 10.0
    
    # key1 should be expired (TTL=5), key2 should still be valid (TTL=100)
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"


@patch('time.monotonic')
def test_delete_operation(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=3, default_ttl=100.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Delete existing key
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    # Delete non-existent key
    assert cache.delete("key1") is False
    assert cache.delete("nonexistent") is False
    
    # Remaining item should still be accessible
    assert cache.get("key2") == "value2"
    assert cache.size() == 1


@patch('time.monotonic')
def test_size_with_mixed_expired_valid(mock_monotonic: Any, ttl_cache: Any) -> None:
    """Test size() returns count of non-expired items only."""
    mock_monotonic.return_value = 0.0
    
    cache = ttl_cache(capacity=5, default_ttl=10.0)
    
    # Add items with different TTLs
    cache.put("key1", "value1", ttl=5.0)   # Expires at t=5
    cache.put("key2", "value2", ttl=15.0)  # Expires at t=15
    cache.put("key3", "value3", ttl=5.0)   # Expires at t=5
    
    # Initially all 3 are valid
    assert cache.size() == 3
    
    # Advance time to t=7 (key1 and key3 expired)
    mock_monotonic.return_value = 7.0
    
    # size() should return 1 (only key2 is valid)
    assert cache.size() == 1
    
    # Verify key2 is still accessible
    assert cache.get("key2") == "value2"
    
    # Advance time past key2's TTL
    mock_monotonic.return_value = 20.0
    
    # All expired now
    assert cache.size() == 0

import pytest
from typing import Any


@pytest.fixture
def ttl_cache(ttl_cache_module: Any) -> Any:
    """Fixture that returns the TTLCache class."""
    return ttl_cache_module.TTLCache