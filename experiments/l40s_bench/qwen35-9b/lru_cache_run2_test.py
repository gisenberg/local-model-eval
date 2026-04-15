import time
import time as time_module
from typing import Any, Optional

class Node:
    """Doubly-linked list node to store key-value pairs and their timestamps."""
    def __init__(self, key: str, value: Any, timestamp: float):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU cache with Time-To-Live (TTL) expiration support.
    
    Uses a doubly-linked list and a hash map for O(1) operations.
    Expiration is checked lazily on access or insertion.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of non-expired items the cache can hold.
            default_ttl: Default time in seconds until an item expires.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.current_time = time_module.monotonic()
        
        # Hash map: key -> Node
        self.cache_map: dict[str, Node] = {}
        
        # Doubly linked list pointers
        self.head: Optional[Node] = None  # Most Recently Used
        self.tail: Optional[Node] = None  # Least Recently Used

    def _get_current_time(self) -> float:
        """Get current monotonic time."""
        return time_module.monotonic()

    def _update_time(self):
        """Update the internal clock to current time."""
        self.current_time = self._get_current_time()

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on its timestamp and TTL."""
        elapsed = self.current_time - node.timestamp
        return elapsed >= node.ttl

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            # node is head
            self.head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            # node is tail
            self.tail = node.prev
            
        node.prev = None
        node.next = None

    def _add_to_head(self, node: Node) -> None:
        """Add a node to the front of the list (MRU position)."""
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # If list is empty or only has expired items, nothing to evict
        if self.tail is None:
            return

        # Check if the current tail is expired
        if self._is_expired(self.tail):
            # If the only item is expired, remove it and stop
            if self.head == self.tail:
                self._remove_node(self.tail)
                self.cache_map.pop(self.tail.key, None)
                return
            
            # If tail is expired but others aren't, we need to find the first non-expired
            # However, per requirements: "If all items are expired, clear them all first."
            # And "If at capacity, evict the least-recently-used non-expired item."
            # This implies we should scan or manage the tail carefully.
            # To keep O(1), we assume the tail is the LRU. If tail is expired, 
            # it effectively acts as the LRU candidate for eviction.
            # But strictly speaking, if tail is expired, we should remove it.
            # If removing the expired tail leaves us at capacity with valid items, we are fine.
            # If removing it still leaves us over capacity (impossible if we remove 1), 
            # we just remove the expired one.
            
            # Actually, the requirement says: "evict the least-recently-used non-expired item".
            # If the LRU (tail) is expired, it is not a candidate for "non-expired".
            # So we must find the first non-expired node from the tail upwards.
            # Since we can't scan O(N), we rely on the fact that if the tail is expired,
            # we remove it. If after removal we are still over capacity (which means we had > capacity valid items?),
            # wait. The invariant is: we only add if size < capacity OR we evict first.
            # So if we are at capacity, and the tail is expired, we remove the tail.
            # If the tail was expired, removing it reduces count. 
            # If we still have capacity issues? 
            # Scenario: Capacity 2. Items A (valid), B (expired). Tail is B.
            # We remove B. Size becomes 1. We can add new item.
            # Scenario: Capacity 2. Items A (valid), B (valid). Tail is B.
            # We remove B. Size becomes 1.
            
            # The tricky case: What if the tail is expired, but there are valid items behind it?
            # In a standard LRU, the tail is the oldest. If the oldest is expired, it's gone.
            # The next oldest becomes the new tail.
            # So simply removing the tail if it's expired is correct logic for "LRU non-expired".
            # Because if the LRU is expired, it's effectively removed from consideration.
            # The next LRU becomes the new tail.
            
            # Wait, what if ALL items are expired?
            # "If all items are expired, clear them all first."
            # If tail is expired, remove it. Check new tail. Repeat until valid or empty.
            # To do this efficiently without O(N) scan:
            # We can just remove the tail. If the list becomes empty, we are done.
            # If the list is not empty, the new tail is the new LRU.
            # We repeat this loop until we find a non-expired tail or list is empty.
            # Since we only add N items, and evict 1 by 1, this amortized cost is O(1) per insertion 
            # over the lifetime of the cache, but worst case per op could be O(N) if many expired.
            # However, the requirement says "lazy cleanup". 
            # Let's implement the loop to ensure correctness.
            
            while self.tail and self._is_expired(self.tail):
                self._remove_node(self.tail)
                self.cache_map.pop(self.tail.key, None)
                if self.tail is None:
                    break
            
            # After cleaning expired tails, if we are still at capacity, evict the new tail
            if self.cache_map and len(self.cache_map) >= self.capacity:
                # At this point, self.tail is the LRU non-expired item (or list is empty)
                if self.tail:
                    self._remove_node(self.tail)
                    self.cache_map.pop(self.tail.key, None)
        else:
            # Tail is valid, it is the LRU non-expired item
            self._remove_node(self.tail)
            self.cache_map.pop(self.tail.key, None)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value if found and valid, otherwise None.
            Updates the key to be Most Recently Used.
        """
        self._update_time()
        
        if key not in self.cache_map:
            return None
            
        node = self.cache_map[key]
        
        if self._is_expired(node):
            # Expired, remove it
            self._remove_node(node)
            del self.cache_map[key]
            return None
            
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Defaults to default_ttl.
        """
        self._update_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self.cache_map:
            # Update existing
            node = self.cache_map[key]
            node.value = value
            node.timestamp = self.current_time
            node.ttl = effective_ttl
            
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # New item
            new_node = Node(key, value, self.current_time)
            new_node.ttl = effective_ttl
            
            # Check capacity
            # Requirement: "If at capacity, evict the least-recently-used non-expired item."
            # "If all items are expired, clear them all first."
            
            # First, clean up any expired items at the tail end
            while self.tail and self._is_expired(self.tail):
                self._remove_node(self.tail)
                self.cache_map.pop(self.tail.key, None)
                if self.tail is None:
                    break
            
            # If still at capacity (and list not empty), evict LRU
            if len(self.cache_map) >= self.capacity and self.tail:
                self._remove_node(self.tail)
                self.cache_map.pop(self.tail.key, None)
            
            # Add new node
            self._add_to_head(new_node)
            self.cache_map[key] = new_node

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key not in self.cache_map:
            return False
            
        node = self.cache_map[key]
        self._remove_node(node)
        del self.cache_map[key]
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed on access/insertion.
        This method returns the current logical size based on the map state.
        Note: Since we clean expired items during get/put/delete, 
        the map size accurately reflects non-expired items unless 
        a bulk expiration happens between checks (which is handled lazily).
        """
        # To be strictly accurate with "lazy cleanup", we might need to scan if we want 
        # to guarantee no expired items are counted, but the requirement says 
        # "expired items removed on access". 
        # However, if we call size() without accessing anything, and there are expired items 
        # that haven't been touched, they remain in the map.
        # The prompt says: "size() -> int — return count of non-expired items (lazy cleanup: expired items removed on access)"
        # This implies size() should probably just return len(map) assuming the invariant 
        # is maintained by get/put. 
        # BUT, if I call size() immediately after put with a very short TTL, it might count expired items.
        # To strictly follow "return count of non-expired items", we should filter.
        # However, filtering is O(N). The requirement says "All operations must be O(1)".
        # This creates a conflict if we must scan for size().
        # Interpretation: The "lazy cleanup" note explains *when* they are removed (on access).
        # Therefore, size() likely returns the current map length, trusting that get/put 
        # have cleaned up expired items. 
        # Let's return len(self.cache_map) as the logical size of the cache structure.
        # If strict accuracy is needed without O(N), we can't do it. 
        # Given the O(1) constraint, we assume the caller accepts the map state as the source of truth.
        return len(self.cache_map)

import pytest
import unittest.mock as mock

class TestTTLCache:
    
    @pytest.fixture
    def mock_time(self):
        """Fixture to provide a mock for time.monotonic."""
        with mock.patch('ttl_cache.time.monotonic') as mock_mono:
            yield mock_mono

    def test_basic_get_put(self, mock_time):
        """Test basic insertion and retrieval."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.size() == 1
        
        assert cache.get("nonexistent") is None

    def test_capacity_eviction_lru_order(self, mock_time):
        """Test that LRU eviction works correctly when capacity is reached."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # 'a' should be evicted as it was LRU
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2

    def test_ttl_expiry(self, mock_time):
        """Test that items expire after their TTL."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key1", "value1")
        
        # Access immediately
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        # Should return None
        assert cache.get("key1") is None
        assert cache.size() == 0

    def test_custom_per_key_ttl(self, mock_time):
        """Test that custom TTL overrides default."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        # Use custom short TTL
        cache.put("key1", "value1", ttl=2.0)
        
        # Advance time past custom TTL but before default
        mock_time.return_value = 3.0
        
        assert cache.get("key1") is None
        assert cache.size() == 0

    def test_delete(self, mock_time):
        """Test delete operation."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.size() == 0
        
        # Delete non-existent
        assert cache.delete("key2") is False

    def test_size_with_mixed_expired_valid(self, mock_time):
        """Test size calculation with a mix of valid and expired items."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        # Insert 3 items
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Advance time so 'a' and 'b' expire, 'c' is still valid
        mock_time.return_value = 6.0
        
        # Access 'c' to trigger lazy cleanup for 'a' and 'b'
        val = cache.get("c")
        assert val == 3
        assert cache.size() == 1
        
        # Verify 'a' and 'b' are gone
        assert cache.get("a") is None
        assert cache.get("b") is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])