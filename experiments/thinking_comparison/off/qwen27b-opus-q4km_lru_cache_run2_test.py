import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    def __init__(self, key: str, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    Uses a doubly-linked list and a hash map for O(1) operations.
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
        
        # Hash map for O(1) lookup: key -> Node
        self.cache_map: dict[str, _Node] = {}
        
        # Doubly-linked list sentinels
        # head.next is the MRU item, tail.prev is the LRU item
        self.head = _Node("", None, 0.0)
        self.tail = _Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expire_at

    def _add_to_head(self, node: _Node) -> None:
        """Move node to the head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _evict_lru(self) -> None:
        """Evict the least recently used item (tail.prev)."""
        if self.tail.prev != self.head:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache_map[lru_node.key]

    def _cleanup_expired(self) -> None:
        """
        Lazy cleanup: Remove all expired items from the cache.
        This is called before operations that need accurate state.
        """
        # We iterate through the list. Since we might modify the list,
        # we collect keys to delete first or carefully traverse.
        # To keep it O(N) worst case but O(1) amortized/average, 
        # we iterate from tail (LRU) to head.
        
        current = self.tail.prev
        while current != self.head:
            if self._is_expired(current):
                # Mark for removal
                node_to_remove = current
                current = current.prev
                self._remove_node(node_to_remove)
                del self.cache_map[node_to_remove.key]
            else:
                current = current.prev

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the key.
        Returns None if key doesn't exist or has expired.
        Updates the key to be most-recently-used.
        """
        if key not in self.cache_map:
            return None
            
        node = self.cache_map[key]
        
        if self._is_expired(node):
            # Lazy deletion on access
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
        If at capacity, evicts the LRU non-expired item.
        """
        # Determine TTL
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = time.monotonic() + effective_ttl

        if key in self.cache_map:
            # Update existing
            node = self.cache_map[key]
            node.value = value
            node.expire_at = expire_at
            # Move to head
            self._remove_node(node)
            self._add_to_head(node)
            return

        # If at capacity, we need to make room.
        # Requirement: "If all items are expired, clear them all first."
        # We check if we are at capacity. If so, we try to evict.
        # However, if the cache is full of expired items, we should clear them.
        
        if len(self.cache_map) >= self.capacity:
            # Check if we can just evict the LRU
            # But first, let's ensure we aren't holding onto expired garbage 
            # that prevents us from inserting new items efficiently if capacity is tight.
            # The prompt says "If all items are expired, clear them all first."
            # This implies a check before eviction logic.
            
            # Optimization: If the cache is full, check if the LRU is expired.
            # If the LRU is expired, it's likely others are too. 
            # We perform a cleanup pass if we are at capacity to free up space.
            self._cleanup_expired()
            
            # After cleanup, if we are still at capacity, evict the LRU
            if len(self.cache_map) >= self.capacity:
                self._evict_lru()

        # Insert new node
        new_node = _Node(key, value, expire_at)
        self.cache_map[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove the key from the cache.
        Returns True if the key existed, False otherwise.
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
        Performs lazy cleanup to ensure accuracy.
        """
        self._cleanup_expired()
        return len(self.cache_map)

import pytest
import time
from unittest.mock import patch
from typing import Any, Optional

# Import the implementation
# Assuming the implementation is in a file named 'solution.py' or similar.
# For this snippet, we assume it's available in the namespace or imported.
# If running locally, ensure 'TTLCache' is imported.
  # Adjust import based on your file structure

class MockTime:
    def __init__(self, start: float = 0.0):
        self.current = start

    def monotonic(self) -> float:
        return self.current

    def advance(self, seconds: float):
        self.current += seconds

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    
    cache.put("key2", "value2")
    assert cache.get("key2") == "value2"
    assert cache.get("key1") == "value1" # Still valid

def test_capacity_eviction():
    """Test LRU eviction order when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a") 
    
    # Add 'c', should evict 'b' (LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None # Evicted

def test_ttl_expiry():
    """Test automatic expiration based on default TTL."""
    mock_time = MockTime(start=0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=10, default_ttl=5.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.advance(6.0)
        
        assert cache.get("key1") is None
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test overriding default TTL with custom TTL."""
    mock_time = MockTime(start=0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=10, default_ttl=10.0)
        
        # Default TTL item
        cache.put("long_live", "val1")
        # Short TTL item
        cache.put("short_live", "val2", ttl=2.0)
        
        assert cache.get("long_live") == "val1"
        assert cache.get("short_live") == "val2"
        
        # Advance 3 seconds
        mock_time.advance(3.0)
        
        # Short live should be gone, long live should remain
        assert cache.get("short_live") is None
        assert cache.get("long_live") == "val1"

def test_delete_operation():
    """Test explicit deletion."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    assert cache.delete("non_existent") is False

def test_size_with_mixed_expired_valid():
    """Test size() returns count of non-expired items only."""
    mock_time = MockTime(start=0.0)
    
    with patch('time.monotonic', side_effect=mock_time.monotonic):
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        assert cache.size() == 3
        
        # Expire 'a' and 'b'
        mock_time.advance(6.0)
        
        # 'c' was inserted last, but wait, they all have same TTL start time?
        # Yes, inserted sequentially at t=0. All expire at t=5.
        # Let's adjust logic: Insert 'c' later.
        
        # Reset for clarity
        cache = TTLCache(capacity=5, default_ttl=5.0)
        cache.put("a", 1) # Expires t=5
        cache.put("b", 2) # Expires t=5
        mock_time.advance(3.0)
        cache.put("c", 3) # Expires t=8
        
        # At t=3: a(valid), b(valid), c(valid) -> size 3
        assert cache.size() == 3
        
        mock_time.advance(3.0) # t=6
        # At t=6: a(expired), b(expired), c(valid) -> size 1
        assert cache.size() == 1
        
        # Verify 'a' and 'b' are actually gone
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") == 3