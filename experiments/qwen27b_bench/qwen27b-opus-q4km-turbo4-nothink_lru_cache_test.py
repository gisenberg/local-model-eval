import time
from typing import Any, Optional

class _Node:
    """Doubly linked list node for O(1) removal and insertion."""
    def __init__(self, key: str, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a hash map for O(1) access and a doubly-linked list for O(1) 
    reordering and eviction.
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
        
        # Hash map for O(1) access to nodes
        self.cache_map: dict[str, _Node] = {}
        
        # Doubly linked list sentinels
        # head points to the most recently used item
        # tail points to the least recently used item
        self.head = _Node("", None, 0.0)
        self.tail = _Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Current count of items in the map (including expired ones)
        self._current_size = 0

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expiry_time

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
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _evict_lru(self) -> None:
        """Evict the least recently used item (node before tail)."""
        if self.tail.prev == self.head:
            return
            
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        del self.cache_map[lru_node.key]
        self._current_size -= 1

    def _cleanup_expired(self) -> None:
        """
        Lazy cleanup: Remove all expired items from the cache.
        This is called when the cache is full or during size checks.
        """
        # We iterate through the map. Since we might modify the map,
        # we collect keys to delete first to avoid runtime error.
        keys_to_delete = []
        for key, node in self.cache_map.items():
            if self._is_expired(node):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            node = self.cache_map.pop(key)
            self._remove_node(node)
            self._current_size -= 1

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the given key.
        
        Returns the value if the key exists and is not expired.
        Accessing a key makes it the most recently used.
        Returns None if the key does not exist or has expired.
        """
        if key not in self.cache_map:
            return None
            
        node = self.cache_map[key]
        
        if self._is_expired(node):
            # Remove expired item
            self._remove_node(node)
            del self.cache_map[key]
            self._current_size -= 1
            return None
            
        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the cache is at capacity, evicts the least recently used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)
        
        # If key exists, update it
        if key in self.cache_map:
            node = self.cache_map[key]
            node.value = value
            node.expiry_time = expiry_time
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
            return

        # If at capacity, we need to make room
        if self._current_size >= self.capacity:
            # First, try to clean up expired items to free space
            self._cleanup_expired()
            
            # If still full, evict LRU
            if self._current_size >= self.capacity:
                self._evict_lru()
        
        # Insert new node
        new_node = _Node(key, value, expiry_time)
        self.cache_map[key] = new_node
        self._add_to_head(new_node)
        self._current_size += 1

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed and was removed, False otherwise.
        """
        if key not in self.cache_map:
            return False
            
        node = self.cache_map[key]
        self._remove_node(node)
        del self.cache_map[key]
        self._current_size -= 1
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup of expired items before returning the count.
        """
        self._cleanup_expired()
        return self._current_size

import pytest
from unittest.mock import patch
import time
from typing import Any

# Import the class we just implemented
  # Adjust import path based on your file structure

class TestTTLCache:
    
    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.put("key2", "value2")
        assert cache.get("key2") == "value2"
        
        # Non-existent key
        assert cache.get("key3") is None

    def test_capacity_eviction_lru_order(self):
        """Test that LRU item is evicted when capacity is reached."""
        # Mock time to prevent actual waiting
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=2, default_ttl=100.0)
            
            cache.put("a", 1)
            cache.put("b", 2)
            
            # Access 'a' to make it MRU
            cache.get("a")
            
            # Add 'c', should evict 'b' (LRU)
            cache.put("c", 3)
            
            assert cache.get("a") == 1  # 'a' exists
            assert cache.get("b") is None  # 'b' evicted
            assert cache.get("c") == 3  # 'c' exists

    def test_ttl_expiry(self):
        """Test that items expire after TTL."""
        # Start time at 0
        with patch('time.monotonic', side_effect=[0.0, 0.0, 5.1]):
            cache = TTLCache(capacity=1, default_ttl=5.0)
            
            cache.put("key", "val")
            assert cache.get("key") == "val"
            
            # Time moves to 5.1 (expired)
            assert cache.get("key") is None

    def test_custom_per_key_ttl(self):
        """Test custom TTL overrides default."""
        with patch('time.monotonic', side_effect=[0.0, 0.0, 2.0, 2.0, 10.0]):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            
            # Short TTL key
            cache.put("short", "s", ttl=1.0)
            # Long TTL key
            cache.put("long", "l", ttl=20.0)
            
            # At t=2.0, short should be expired, long valid
            assert cache.get("short") is None
            assert cache.get("long") == "l"
            
            # At t=10.0, long should still be valid
            assert cache.get("long") == "l"

    def test_delete(self):
        """Test deletion of keys."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            
            cache.put("k", "v")
            assert cache.delete("k") is True
            assert cache.get("k") is None
            
            # Delete non-existent
            assert cache.delete("nonexistent") is False

    def test_size_with_mixed_expired_valid(self):
        """Test size() returns count of non-expired items only."""
        with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 6.0]):
            cache = TTLCache(capacity=3, default_ttl=5.0)
            
            cache.put("a", 1)
            cache.put("b", 2)
            cache.put("c", 3)
            
            # All valid
            assert cache.size() == 3
            
            # Time passes, all expire
            # Note: size() triggers cleanup
            assert cache.size() == 0
            
            # Add new item
            cache.put("d", 4)
            assert cache.size() == 1