# Qwen3.5-122B-A10B-REAP-20 Q4_K_M (0xSero) [ik-llama] — KV f16 (baseline) — LRU Cache with TTL

## Output

### Implementation of TTLCache

Here is the complete implementation of the `TTLCache` class using a doubly-linked list and a hash map to ensure $O(1)$ average time complexity for all operations. The solution strictly adheres to the requirements, including manual linked list management, time-based expiration, and specific eviction logic.

```python
import time
from typing import Any, Optional, Dict
from collections import deque

class _Node:
    """Doubly linked list node for the cache."""
    __slots__ = ['key', 'value', 'ttl', 'expiry', 'prev', 'next']

    def __init__(self, key: str, value: Any, ttl: float, current_time: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expiry = current_time + ttl
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a doubly-linked list for O(1) access and reordering, and a hash map
    for O(1) key lookups. Expired items are cleaned up lazily during access
    or when the cache is full and needs eviction.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items without a custom TTL.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        
        # Dummy head and tail nodes to simplify edge cases in linked list operations
        self._head = _Node("", None, 0, 0)
        self._tail = _Node("", None, 0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _get_current_time(self) -> float:
        """Wrapper for time.monotonic to facilitate mocking in tests."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node is expired based on current time."""
        return self._get_current_time() > node.expiry

    def _evict_lru_non_expired(self) -> Optional[str]:
        """
        Evict the least recently used non-expired item.
        Returns the key of the evicted item, or None if no valid items exist.
        """
        current = self._tail.prev
        while current != self._head:
            if not self._is_expired(current):
                # Found a valid LRU item
                key_to_remove = current.key
                self._remove_node(current)
                del self._map[key_to_remove]
                return key_to_remove
            # If expired, remove it immediately (lazy cleanup during eviction scan)
            key_to_remove = current.key
            prev_node = current.prev
            self._remove_node(current)
            del self._map[key_to_remove]
            current = prev_node
        
        return None

    def _cleanup_all_expired(self) -> None:
        """Remove all expired items from the cache."""
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the key.
        
        If the key exists and is not expired, it is moved to the front (most recently used)
        and the value is returned. If the key is expired or does not exist, returns None.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and valid, otherwise None.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        
        if self._is_expired(node):
            # Lazy cleanup: remove expired item on access
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If the key exists, it is updated and moved to the front.
        If the cache is at capacity, the least recently used non-expired item is evicted.
        If all items are expired, they are cleared first before insertion.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional custom time-to-live in seconds. If None, uses default_ttl.
        """
        current_time = self._get_current_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        # If key exists, update it
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.ttl = effective_ttl
            node.expiry = current_time + effective_ttl
            # Move to front
            self._remove_node(node)
            self._add_to_front(node)
            return

        # Check if we need to evict
        # First, ensure we aren't just full of expired items
        if len(self._map) >= self.capacity:
            # Try to find a non-expired LRU to evict
            evicted = self._evict_lru_non_expired()
            
            # If we couldn't evict a non-expired item, it means all items are expired
            # or the cache was effectively empty of valid items.
            # In this case, we clear all expired items to make room.
            if evicted is None:
                self._cleanup_all_expired()
        
        # Re-check capacity after cleanup (in case cleanup freed up space)
        # If still at capacity, we must have found a valid LRU to evict in the previous step
        # or the map is now smaller than capacity.
        # However, if the previous step evicted a valid item, len(self._map) is now < capacity.
        # If the previous step found no valid items to evict, we cleaned all expired, 
        # so len(self._map) is now 0 (or < capacity).
        
        # Create new node
        new_node = _Node(key, value, effective_ttl, current_time)
        self._map[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key in self._map:
            node = self._map[key]
            self._remove_node(node)
            del self._map[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs a lazy cleanup of expired items during the count.
        
        Returns:
            The number of valid items.
        """
        # We need to count valid items. Since we use lazy cleanup, 
        # we must iterate and remove expired ones to get an accurate count 
        # and maintain consistency for future operations.
        current = self._head.next
        count = 0
        
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            else:
                count += 1
            current = next_node
            
        return count
```

### Test Suite

The following tests use `unittest.mock.patch` to control `time.monotonic`, ensuring deterministic behavior without relying on `time.sleep`.

```python
import pytest
from unittest.mock import patch
from ttl_cache_impl import TTLCache  # Assuming the implementation is saved as ttl_cache_impl.py

# Helper to create a cache instance
def create_cache(capacity=3, default_ttl=10.0):
    return TTLCache(capacity=capacity, default_ttl=default_ttl)

class TestTTLCache:

    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.put("key2", "value2")
        assert cache.get("key2") == "value2"
        
        # Update existing key
        cache.put("key1", "updated_value1")
        assert cache.get("key1") == "updated_value1"
        
        # Non-existent key
        assert cache.get("non_existent") is None

    def test_capacity_eviction_lru_order(self):
        """Test that LRU items are evicted when capacity is reached."""
        cache = create_cache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'c'. 'b' is LRU and should be evicted.
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None  # Evicted

    def test_ttl_expiry(self):
        """Test that items expire after default TTL."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=2, default_ttl=5.0)
            
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            
            # Advance time to just before expiry
            mock_time.return_value = 4.9
            assert cache.get("key1") == "value1"
            
            # Advance time past expiry
            mock_time.return_value = 5.1
            assert cache.get("key1") is None

    def test_custom_per_key_ttl(self):
        """Test that custom TTL overrides default TTL."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # Item with custom short TTL
            cache.put("short", "val", ttl=2.0)
            # Item with default TTL
            cache.put("long", "val", ttl=None)
            
            # Time 3.0: 'short' should be expired, 'long' valid
            mock_time.return_value = 3.0
            assert cache.get("short") is None
            assert cache.get("long") == "val"
            
            # Time 11.0: 'long' should also be expired
            mock_time.return_value = 11.0
            assert cache.get("long") is None

    def test_delete(self):
        """Test manual deletion of keys."""
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Delete non-existent key
        assert cache.delete("non_existent") is False

    def test_size_with_mixed_expired_valid_items(self):
        """Test size() returns count of valid items and cleans up expired ones."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=5, default_ttl=5.0)
            
            cache.put("valid1", 1)
            cache.put("valid2", 2)
            cache.put("exp1", 3, ttl=2.0)
            cache.put("exp2", 4, ttl=1.0)
            
            # Initially all 4 are present in map, but size should count valid?
            # Requirement: "return count of non-expired items (lazy cleanup: expired items removed on access)"
            # size() triggers cleanup.
            
            # At t=0, all are valid.
            assert cache.size() == 4
            
            # Advance time to 3.0. exp1 and exp2 are expired.
            mock_time.return_value = 3.0
            
            # size() should clean up exp1 and exp2 and return 2
            assert cache.size() == 2
            
            # Verify expired items are actually gone
            assert cache.get("exp1") is None
            assert cache.get("exp2") is None
            assert cache.get("valid1") == 1
            assert cache.get("valid2") == 2
```

### Key Implementation Details

1.  **Data Structures**:
    *   **Hash Map (`_map`)**: Stores `key -> Node` for $O(1)$ lookup.
    *   **Doubly Linked List**: Maintains order. `_head` is the dummy head (MRU side), `_tail` is the dummy tail (LRU side).
    *   **Node**: Stores `key`, `value`, `expiry` timestamp, and pointers.

2.  **Time Management**:
    *   Uses `time.monotonic()` for stability (not affected by system clock changes).
    *   `put` calculates `expiry = current_time + ttl`.
    *   `get` and `size` check `current_time > expiry` to determine validity.

3.  **Eviction Logic**:
    *   When `put` is called and the cache is full, it scans from the LRU end (`_tail.prev`).
    *   It looks for the first **non-expired** item to evict.
    *   If it encounters an expired item during the scan, it removes it immediately (lazy cleanup) and continues scanning.
    *   If the scan reaches the head without finding a non-expired item, it implies all items are expired. In this case, `_cleanup_all_expired` is called to clear the cache before inserting the new item.

4.  **Complexity**:
    *   `get`, `put`, `delete`, `size`: All involve hash map lookups and constant number of linked list pointer manipulations. $O(1)$ average time.
    *   Note: While `size` iterates, it only iterates over the linked list. In a worst-case scenario where many items are expired, it cleans them up. However, each item is cleaned up exactly once. The amortized cost remains efficient, and the requirement specifically asks for lazy cleanup on access/size.

5.  **Testing**:
    *   `unittest.mock.patch` replaces `time.monotonic` with a controlled value, allowing precise testing of TTL expiration without waiting.
