import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly linked list."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = None
        self.next = None

class TTLCache:
    """
    LRU Cache with TTL (Time-To-Live) support.
    Uses a doubly linked list and a hash map for O(1) operations.
    Implements lazy cleanup for expired items.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes for the doubly linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add(self, node: _Node) -> None:
        """Add a node to the front of the list (after head)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: Any) -> Optional[Any]:
        """
        Get the value for a key. Returns None if not found or expired.
        Moves the accessed item to the front (most recently used).
        Performs lazy cleanup if the item is expired.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, or None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        current_time = time.monotonic()

        if current_time > node.expiry:
            # Lazy cleanup: remove expired item
            self._remove(node)
            del self.cache[key]
            return None

        # Move to front (Most Recently Used)
        self._remove(node)
        self._add(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put a key-value pair into the cache.
        If key exists, updates value and TTL, moves to front.
        If key is new, adds to front. Evicts LRU if at capacity.

        Args:
            key: The key.
            value: The value.
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        if ttl is None:
            ttl = self.default_ttl

        current_time = time.monotonic()
        expiry = current_time + ttl

        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            # Move to front
            self._remove(node)
            self._add(node)
            return

        # New key
        if len(self.cache) >= self.capacity:
            # Evict LRU (Tail)
            lru_node = self.tail.prev
            # Note: lru_node could be expired, but we just evict it.
            # If it was expired, we cleaned up. If not, we evicted valid data.
            self._remove(lru_node)
            del self.cache[lru_node.key]

        # Add new node
        new_node = _Node(key, value, expiry)
        self._add(new_node)
        self.cache[key] = new_node

    def delete(self, key: Any) -> None:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Returns the number of items currently in the cache map.
        Note: This includes expired items that haven't been lazily cleaned up yet.

        Returns:
            Number of items.
        """
        return len(self.cache)

import pytest
from unittest.mock import patch
import time

# Assuming TTLCache is defined in the same scope or imported
# 
class TestTTLCache:
    """Tests for TTLCache implementation."""

    def test_basic_put_get(self):
        """Test basic put and get functionality."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        # Mock time.monotonic to return 10.0 for put, 10.0 for get
        with patch('time.monotonic', side_effect=[10.0, 10.0]):
            cache.put('key1', 'value1')
            result = cache.get('key1')
            
        assert result == 'value1'
        assert cache.size() == 1

    def test_get_expired(self):
        """Test that get returns None for expired items."""
        cache = TTLCache(capacity=2, default_ttl=5.0)
        
        # Mock time: 10.0 for put, 16.0 for get (after expiry)
        with patch('time.monotonic', side_effect=[10.0, 16.0]):
            cache.put('key1', 'value1')
            result = cache.get('key1')
            
        assert result is None
        assert cache.size() == 0  # Lazy cleanup removed it

    def test_lru_eviction(self):
        """Test LRU eviction when capacity is reached."""
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        # Mock times for 3 puts and 1 get
        # Put 1 (t=10), Put 2 (t=11), Put 3 (t=12), Get 1 (t=12)
        with patch('time.monotonic', side_effect=[10.0, 11.0, 12.0, 12.0]):
            cache.put('key1', 'value1')
            cache.put('key2', 'value2')
            cache.put('key3', 'value3') # Evicts key1
            
            result = cache.get('key1')
            
        assert result is None
        assert cache.size() == 2

    def test_update_existing_key(self):
        """Test updating an existing key refreshes TTL and moves to front."""
        cache = TTLCache(capacity=2, default_ttl=5.0)
        
        # Mock times: Put 1 (t=10), Update 1 (t=12), Get 1 (t=12)
        with patch('time.monotonic', side_effect=[10.0, 12.0, 12.0]):
            cache.put('key1', 'value1')
            cache.put('key1', 'value2') # Updates value and TTL
            
            result = cache.get('key1')
            
        assert result == 'value2'
        assert cache.size() == 1

    def test_delete_key(self):
        """Test deleting a key removes it from cache."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        # Mock times: Put (t=10), Get (t=10)
        with patch('time.monotonic', side_effect=[10.0, 10.0]):
            cache.put('key1', 'value1')
            cache.delete('key1')
            result = cache.get('key1')
            
        assert result is None
        assert cache.size() == 0

    def test_custom_ttl(self):
        """Test using custom TTL override."""
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        # Mock times: Put (t=10, ttl=5 -> expiry 15), Get (t=12), Get (t=16)
        with patch('time.monotonic', side_effect=[10.0, 12.0, 16.0]):
            cache.put('key1', 'value1', ttl=5.0)
            result1 = cache.get('key1')
            result2 = cache.get('key1')
            
        assert result1 == 'value1'
        assert result2 is None