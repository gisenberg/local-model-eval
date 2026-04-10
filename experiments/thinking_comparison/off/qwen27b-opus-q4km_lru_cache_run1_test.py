from __future__ import annotations
import time
from typing import Any, Optional


class _Node:
    """Internal doubly-linked list node for cache entries."""
    
    __slots__ = ('key', 'value', 'exp_time', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, exp_time: float):
        self.key = key
        self.value = value
        self.exp_time = exp_time
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Supports O(1) get, put, and delete operations using a 
    doubly-linked list and hash map.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items to store.
            default_ttl: Default time-to-live in seconds for cached items.
        """
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map: key -> Node
        self._cache: dict[str, _Node] = {}
        
        # Doubly-linked list sentinels
        self._head: _Node = _Node("", None, 0.0)  # Most recently used
        self._tail: _Node = _Node("", None, 0.0)  # Least recently used
        self._head.next = self._tail
        self._tail.prev = self._head
        
        self._size = 0  # Number of non-expired items
    
    def _current_time(self) -> float:
        """Get current monotonic time."""
        return time.monotonic()
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return self._current_time() >= node.exp_time
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
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
    
    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # Start from tail (LRU end) and find first non-expired
        node = self._tail.prev
        while node != self._head:
            if self._is_expired(node):
                # Remove expired node
                self._remove_node(node)
                del self._cache[node.key]
                self._size -= 1
                node = node.prev
            else:
                # Found LRU non-expired, evict it
                self._remove_node(node)
                del self._cache[node.key]
                self._size -= 1
                return
    
    def _clear_expired(self) -> None:
        """Remove all expired items from the cache."""
        node = self._tail.prev
        while node != self._head:
            if self._is_expired(node):
                next_node = node.prev
                self._remove_node(node)
                del self._cache[node.key]
                self._size -= 1
                node = next_node
            else:
                node = node.prev
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the given key.
        
        Returns the value if the key exists and is not expired.
        Accessing a key makes it most-recently-used.
        Returns None if key doesn't exist or has expired.
        
        Args:
            key: The key to look up.
            
        Returns:
            The cached value, or None if not found/expired.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        
        # Check if expired
        if self._is_expired(node):
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
        Insert or update a key-value pair in the cache.
        
        If the key already exists, updates the value and refreshes the TTL.
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Uses default_ttl if not provided.
        """
        if self._capacity <= 0:
            return
        
        exp_time = self._current_time() + (ttl if ttl is not None else self._default_ttl)
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.exp_time = exp_time
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key
            if self._size >= self._capacity:
                # Check if all items are expired - clear them all
                all_expired = True
                node = self._tail.prev
                while node != self._head:
                    if not self._is_expired(node):
                        all_expired = False
                        break
                    node = node.prev
                
                if all_expired:
                    self._clear_expired()
                else:
                    self._evict_lru()
            
            new_node = _Node(key, value, exp_time)
            self._cache[key] = new_node
            self._add_to_head(new_node)
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
        
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Count of non-expired items.
        """
        # Lazy cleanup: remove expired items
        node = self._tail.prev
        while node != self._head:
            if self._is_expired(node):
                next_node = node.prev
                self._remove_node(node)
                del self._cache[node.key]
                self._size -= 1
                node = next_node
            else:
                node = node.prev
        
        return self._size

# tests.py
import pytest
from unittest.mock import patch



@patch('ttl_cache.time.monotonic')
def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("nonexistent") is None


@patch('ttl_cache.time.monotonic')
def test_capacity_eviction_lru_order(mock_time):
    """Test that LRU eviction works correctly when at capacity."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Access key1 to make it MRU
    cache.get("key1")
    
    # Add key3 - should evict key2 (LRU)
    cache.put("key3", "value3")
    
    assert cache.get("key1") == "value1"  # Still there (was accessed)
    assert cache.get("key2") is None      # Evicted (LRU)
    assert cache.get("key3") == "value3"  # Still there


@patch('ttl_cache.time.monotonic')
def test_ttl_expiry(mock_time):
    """Test that items expire after their TTL."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    
    # Before expiry
    assert cache.get("key1") == "value1"
    
    # After expiry
    mock_time.return_value = 15.0
    assert cache.get("key1") is None


@patch('ttl_cache.time.monotonic')
def test_custom_per_key_ttl(mock_time):
    """Test custom TTL per key overrides default."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("key1", "value1", ttl=5.0)  # Custom short TTL
    cache.put("key2", "value2")           # Default TTL
    
    # Advance time past key1's TTL but before key2's
    mock_time.return_value = 10.0
    
    assert cache.get("key1") is None  # Expired (5s TTL)
    assert cache.get("key2") == "value2"  # Still valid (100s TTL)


@patch('ttl_cache.time.monotonic')
def test_delete(mock_time):
    """Test delete operation."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("key1", "value1")
    
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("key1") is False  # Already deleted


@patch('ttl_cache.time.monotonic')
def test_size_with_mixed_expired_valid(mock_time):
    """Test size() returns count of non-expired items only."""
    mock_time.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=100.0)
    
    cache.put("key1", "value1", ttl=5.0)   # Expires at t=5
    cache.put("key2", "value2")            # Expires at t=100
    cache.put("key3", "value3", ttl=10.0)  # Expires at t=10
    
    # At t=0, all valid
    assert cache.size() == 3
    
    # At t=6, key1 expired
    mock_time.return_value = 6.0
    assert cache.size() == 2
    
    # At t=15, key1 and key3 expired
    mock_time.return_value = 15.0
    assert cache.size() == 1
    
    # At t=150, all expired
    mock_time.return_value = 150.0
    assert cache.size() == 0