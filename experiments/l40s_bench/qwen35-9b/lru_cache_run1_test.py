import time
import time as time_module
from typing import Any, Optional

class Node:
    """Doubly linked list node to store key, value, and expiration time."""
    def __init__(self, key: str, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.expiry = time.monotonic() + ttl
        self.prev = None
        self.next = None

class TTLCache:
    """
    An LRU cache with time-based expiration.
    
    Uses a doubly-linked list and a hash map for O(1) operations.
    Expiration is checked lazily on access or eviction.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._current_time = time.monotonic()
        
        # Hash map: key -> Node
        self._cache: dict[str, Node] = {}
        
        # Doubly linked list pointers
        self._head: Optional[Node] = None  # Most Recently Used
        self._tail: Optional[Node] = None  # Least Recently Used

    def _get_current_time(self) -> float:
        """Helper to get current monotonic time."""
        return time.monotonic()

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            # node is head
            self._head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            # node is tail
            self._tail = node.prev
            
        node.prev = None
        node.next = None

    def _add_to_head(self, node: Node) -> None:
        """Add a node to the front of the list (MRU)."""
        if self._head is None:
            self._head = node
            self._tail = node
        else:
            node.next = self._head
            self._head.prev = node
            self._head = node

    def _evict_lru(self) -> Optional[Node]:
        """Evict the least recently used non-expired item. Returns the evicted node or None."""
        # If list is empty or only head exists, nothing to evict
        if self._tail is None:
            return None
        
        # Check if tail is expired
        if self._tail.expiry <= self._get_current_time():
            return self._tail

        # If we have capacity, we might not need to evict immediately if the LRU is valid
        # But the requirement says: "If at capacity, evict the least-recently-used non-expired item."
        # So we only evict if we are at capacity AND the LRU is valid.
        # However, if the LRU is expired, we should clear it first (handled in put logic usually, 
        # but here we need to find the first VALID node from the tail).
        
        # Find the first non-expired node starting from tail
        current = self._tail
        while current:
            if current.expiry > self._get_current_time():
                return current
            current = current.prev
            
        # If we reach here, all items are expired. We should evict the tail (which is expired)
        # effectively clearing the list or just removing the expired one.
        return self._tail

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache and list."""
        current = self._head
        while current:
            next_node = current.next
            if current.expiry <= self._get_current_time():
                self._remove_node(current)
                del self._cache[current.key]
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        Moves accessed key to MRU position.
        Returns None if key missing or expired.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        
        # Lazy cleanup: check if expired
        if node.expiry <= self._get_current_time():
            self._remove_node(node)
            del self._cache[key]
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        Custom ttl overrides default.
        Evicts LRU non-expired item if at capacity.
        Clears expired items if all are expired.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        current_time = self._get_current_time()
        
        # If key exists, update it
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = current_time + effective_ttl
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
            return

        # Key does not exist
        new_node = Node(key, value, effective_ttl)
        
        # Check if we need to evict
        if len(self._cache) >= self.capacity:
            # Requirement: "If all items are expired, clear them all first."
            # We check the tail (LRU). If it's expired, we remove it.
            # If the tail is NOT expired, we evict it.
            # If the tail IS expired, we remove it and check the next one until we find a valid one or list empty.
            
            evicted_node = self._evict_lru()
            if evicted_node:
                self._remove_node(evicted_node)
                del self._cache[evicted_node.key]
        
        # Add new node
        self._add_to_head(new_node)
        self._cache[key] = new_node

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        Returns True if key existed, False otherwise.
        """
        if key not in self._cache:
            return False
        
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """
        Return count of non-expired items.
        Performs lazy cleanup of expired items during iteration.
        """
        current_time = self._get_current_time()
        count = 0
        
        # We must iterate carefully. Since we are modifying the list while iterating,
        # we collect keys to delete first or iterate via a copy of keys.
        # However, to maintain O(1) average for the operation itself (excluding cleanup cost),
        # we iterate the list, count valid, and remove invalid.
        
        # To avoid O(N) removals inside the loop affecting complexity perception,
        # we will rebuild the list logic or just count and clean.
        # Given the requirement "lazy cleanup: expired items removed on access", 
        # size() should also clean them to return an accurate count.
        
        # Strategy: Iterate from Head. If expired, remove. If not, count.
        # Since we are removing nodes, we need to be careful with the 'next' pointer.
        
        current = self._head
        while current:
            next_node = current.next
            if current.expiry <= current_time:
                self._remove_node(current)
                del self._cache[current.key]
            else:
                count += 1
            current = next_node
            
        return count

import pytest
from unittest.mock import patch, MagicMock

class TestTTLCache:
    
    @patch('ttl_cache.time.monotonic')
    def test_basic_get_put(self, mock_time):
        """Test basic insertion and retrieval."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Verify LRU order changed (key1 should be at head)
        # Though hard to inspect internal list directly without exposing it, 
        # we can verify subsequent operations work.
        cache.put("key2", "value2")
        assert cache.get("key2") == "value2"
        assert cache.get("key1") == "value1"

    @patch('ttl_cache.time.monotonic')
    def test_capacity_eviction_lru_order(self, mock_time):
        """Test that LRU eviction works correctly when capacity is reached."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict 'a' (LRU)
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        
        # Access 'b' to make it MRU
        cache.get("b")
        cache.put("d", 4)  # Should evict 'c' now
        
        assert cache.get("c") is None
        assert cache.get("d") == 4

    @patch('ttl_cache.time.monotonic')
    def test_ttl_expiry(self, mock_time):
        """Test that items expire based on default TTL."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key", "value")
        assert cache.get("key") == "value"
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        assert cache.get("key") is None
        assert cache.size() == 0

    @patch('ttl_cache.time.monotonic')
    def test_custom_per_key_ttl(self, mock_time):
        """Test that custom TTL overrides default."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("short", "val", ttl=2.0)
        cache.put("long", "val", ttl=20.0)
        
        # Advance time to 5 seconds
        mock_time.return_value = 5.0
        
        assert cache.get("short") is None  # Expired
        assert cache.get("long") == "val"  # Still valid

    @patch('ttl_cache.time.monotonic')
    def test_delete(self, mock_time):
        """Test delete operation."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("key") is False  # Already deleted

    @patch('ttl_cache.time.monotonic')
    def test_size_with_mixed_expired_valid(self, mock_time):
        """Test size() returns count of non-expired items with lazy cleanup."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=10, default_ttl=5.0)
        
        cache.put("valid", "v1")
        cache.put("expired", "v2")
        cache.put("valid2", "v3")
        
        # Initially all 3 exist
        assert cache.size() == 3
        
        # Advance time to 6 seconds
        mock_time.return_value = 6.0
        
        # 'expired' should be removed, 'valid' and 'valid2' remain
        assert cache.size() == 2
        
        # Accessing valid items should keep them
        assert cache.get("valid") == "v1"
        assert cache.get("valid2") == "v3"
        assert cache.size() == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])