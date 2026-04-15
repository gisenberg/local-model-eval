import time
import time as time_module
from typing import Any, Dict, List, Optional, Tuple

class Node:
    """Doubly linked list node to store key-value pairs and their timestamps."""
    def __init__(self, key: str, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.next: Optional['Node'] = None
        self.prev: Optional['Node'] = None
        self.expiry_time: float = time_module.monotonic() + ttl

class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a combination of a doubly-linked list and a hash map for O(1) operations.
    Expiration is checked lazily upon access or insertion.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of non-expired items the cache can hold.
            default_ttl: Default time-to-live in seconds for new items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be greater than 0.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._current_time = time_module.monotonic()
        
        # Hash map: key -> Node
        self._cache: Dict[str, Node] = {}
        
        # Doubly linked list head and tail
        # Sentinel nodes to simplify edge cases
        self._head: Node = Node("", None, 0)
        self._tail: Node = Node("", None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _update_time(self) -> None:
        """Update the current monotonic time to ensure accuracy."""
        self._current_time = time_module.monotonic()

    def _is_expired(self, node: Node) -> bool:
        """Check if a specific node has expired based on current time."""
        return self._current_time >= node.expiry_time

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add a node to the front of the list (Most Recently Used)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """Remove the least recently used non-expired item if capacity is exceeded."""
        # Traverse from tail (LRU) to find the first non-expired node
        current = self._tail.prev
        while current != self._head:
            if not self._is_expired(current):
                # Found the LRU valid item
                self._remove_node(current)
                if current.key in self._cache:
                    del self._cache[current.key]
                return
            # If expired, we will clear it anyway, but we stop at the first valid one
            # to ensure we don't evict a valid item if we only need to remove expired ones.
            # However, the requirement says: "If at capacity, evict the least-recently-used non-expired item."
            # If all are expired, we clear them all.
            current = current.prev
        
        # If we reach here, all items in the list are expired. 
        # We need to clear the whole list to make room.
        # Since the loop above didn't return, it means current == self._head (all expired).
        # We just need to clear the map and reset the list.
        self._cache.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _cleanup_expired(self) -> None:
        """Remove all expired nodes from the list and map."""
        current = self._head.next
        while current != self._tail:
            if self._is_expired(current):
                self._remove_node(current)
                if current.key in self._cache:
                    del self._cache[current.key]
            else:
                break
            current = current.next

    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if the key exists and is not expired, otherwise None.
            Accessing a valid key moves it to the front of the list (MRU).
        """
        self._update_time()
        
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        
        if self._is_expired(node):
            # Expired, remove it and return None
            self._remove_node(node)
            if key in self._cache:
                del self._cache[key]
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional custom TTL. If None, uses default_ttl.
        """
        self._update_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self._cache:
            # Update existing
            node = self._cache[key]
            node.value = value
            node.ttl = effective_ttl
            node.expiry_time = self._current_time + effective_ttl
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # New item
            new_node = Node(key, value, effective_ttl)
            new_node.expiry_time = self._current_time + effective_ttl
            
            self._cache[key] = new_node
            self._add_to_head(new_node)
            
            # Check capacity
            if len(self._cache) > self.capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        if key not in self._cache:
            return False
        
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Note: This performs a lazy cleanup. It does not remove expired items 
        from the internal structures immediately but counts only valid ones.
        """
        self._update_time()
        count = 0
        current = self._head.next
        while current != self._tail:
            if not self._is_expired(current):
                count += 1
            current = current.next
        return count

# Alias for convenience
LRUCache = TTLCache

import pytest
import unittest.mock as mock
from typing import Any, Optional

class TestTTLCache:
    def setup_method(self):
        """Set up a fresh cache instance for each test."""
        self.cache = TTLCache(capacity=2, default_ttl=10.0)

    def test_basic_put_and_get(self):
        """Test basic insertion and retrieval."""
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("nonexistent") is None

    def test_capacity_eviction_lru_order(self):
        """Test that the LRU item is evicted when capacity is reached."""
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)  # 'a' should be evicted
        
        assert self.cache.get("a") is None
        assert self.cache.get("b") == 2
        assert self.cache.get("c") == 3
        
        # Access 'b' to make it MRU, then add 'd'
        self.cache.get("b")
        self.cache.put("d", 4)  # 'c' should be evicted now
        
        assert self.cache.get("c") is None
        assert self.cache.get("d") == 4

    def test_ttl_expiry(self):
        """Test that items expire after their TTL."""
        self.cache.put("key1", "value1", ttl=0.1)
        
        # Should exist immediately
        assert self.cache.get("key1") == "value1"
        
        # Mock time to pass 0.2 seconds (past TTL)
        with mock.patch.object(self.cache, '_update_time') as mock_update:
            mock_update.return_value = 100.2  # Current time is 100.2
            
            # Expiry check happens inside get
            result = self.cache.get("key1")
            assert result is None

    def test_custom_per_key_ttl(self):
        """Test overriding default TTL with a custom value."""
        # Default is 10s, set custom to 0.1s
        self.cache.put("key1", "value1", ttl=0.1)
        
        # Verify it works immediately
        assert self.cache.get("key1") == "value1"
        
        # Mock time to pass 0.2s
        with mock.patch.object(self.cache, '_update_time') as mock_update:
            mock_update.return_value = 100.2
            assert self.cache.get("key1") is None
            
        # Verify default TTL still works for new items
        self.cache.put("key2", "value2")
        with mock.patch.object(self.cache, '_update_time') as mock_update:
            mock_update.return_value = 100.2
            assert self.cache.get("key2") == "value2"

    def test_delete_operation(self):
        """Test the delete method."""
        self.cache.put("key1", "value1")
        assert self.cache.delete("key1") is True
        assert self.cache.get("key1") is None
        assert self.cache.delete("key1") is False  # Already deleted

    def test_size_with_mixed_expired_valid_items(self):
        """Test size() returns count of non-expired items only."""
        self.cache.put("valid", "data")
        self.cache.put("expiring", "data", ttl=0.1)
        
        # Initially both are valid
        assert self.cache.size() == 2
        
        # Mock time to expire 'expiring'
        with mock.patch.object(self.cache, '_update_time') as mock_update:
            mock_update.return_value = 100.2
            # size() should reflect only the valid item
            assert self.cache.size() == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])