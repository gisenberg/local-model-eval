# Qwen3.5-122B-A10B Q4_K_M (bartowski) — KV f16K/q8V (asymmetric) — LRU Cache with TTL

## Output

### Implementation: TTLCache

Here is the complete implementation of the `TTLCache` class using a doubly-linked list and a hash map, adhering to all specified requirements including O(1) operations and time-based expiration.

```python
import time
from typing import Any, Optional, Dict, List
from collections import deque

class _Node:
    """Doubly linked list node for the cache."""
    __slots__ = 'key', 'value', 'ttl_expiry', 'prev', 'next'

    def __init__(self, key: str, value: Any, ttl_expiry: float):
        self.key = key
        self.value = value
        self.ttl_expiry = ttl_expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a doubly-linked list to maintain access order and a hash map for O(1) lookups.
    Expired items are removed lazily upon access or when the cache is full.
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
            raise ValueError("TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        
        # Dummy head and tail nodes to simplify edge cases
        self._head = _Node("", None, 0)
        self._tail = _Node("", None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _get_current_time(self) -> float:
        """Helper to get current monotonic time."""
        return time.monotonic()

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # The LRU item is right before the tail
        lru_node = self._tail.prev
        if lru_node == self._head:
            return  # Cache is empty

        # Remove from map and list
        del self._map[lru_node.key]
        self._remove_node(lru_node)

    def _cleanup_expired(self) -> None:
        """
        Remove all expired items from the cache.
        This is called when the cache is full to ensure we have space for new items.
        """
        current_time = self._get_current_time()
        # We iterate through the list. Since we need to remove items, 
        # we traverse carefully.
        node = self._head.next
        while node != self._tail:
            next_node = node.next
            if node.ttl_expiry <= current_time:
                del self._map[node.key]
                self._remove_node(node)
            node = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns the value if the key exists and is not expired.
        Accessing a key moves it to the most recently used position.
        Returns None if the key does not exist or is expired.
        """
        node = self._map.get(key)
        if not node:
            return None

        current_time = self._get_current_time()
        if node.ttl_expiry <= current_time:
            # Item expired, remove it
            del self._map[key]
            self._remove_node(node)
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, update its value and TTL, and move to MRU.
        If the key is new and capacity is reached, evict the LRU non-expired item.
        If all items are expired, clear them first before inserting.
        """
        current_time = self._get_current_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl

        # Check if key exists
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.ttl_expiry = expiry_time
            # Move to front
            self._remove_node(node)
            self._add_to_front(node)
            return

        # Key is new
        # If at capacity, we need to make space
        if len(self._map) >= self.capacity:
            # First, try to clean up expired items to free space
            self._cleanup_expired()
            
            # If still at capacity, evict the LRU item
            if len(self._map) >= self.capacity:
                self._evict_lru()

        # Create new node and add
        new_node = _Node(key, value, expiry_time)
        self._map[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed and was removed, False otherwise.
        """
        if key in self._map:
            node = self._map[key]
            del self._map[key]
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed during this operation
        to ensure the returned size is accurate.
        """
        self._cleanup_expired()
        return len(self._map)
```

### Test Suite

Here are the 6 pytest tests covering the required scenarios. They use `unittest.mock.patch` to control time deterministically.

```python
import pytest
from unittest.mock import patch
from ttl_cache_impl import TTLCache  # Assuming the class above is saved as ttl_cache_impl.py

# Helper to create a cache instance
def create_cache(capacity=3, default_ttl=10.0):
    return TTLCache(capacity=capacity, default_ttl=default_ttl)

class TestTTLCache:

    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
        assert cache.get("non_existent") is None
        
        # Update existing key
        cache.put("key1", "updated_value")
        assert cache.get("key1") == "updated_value"

    def test_capacity_eviction_lru_order(self):
        """Test that LRU eviction works correctly when capacity is reached."""
        cache = create_cache(capacity=2)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU, 'b' becomes LRU
        cache.get("a")
        
        # Insert 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None  # Evicted

    def test_ttl_expiry(self):
        """Test that items expire after default TTL."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=2, default_ttl=5.0)
            cache.put("key1", "value1")
            
            # Time passes, but within TTL
            with patch('time.monotonic', return_value=4.9):
                assert cache.get("key1") == "value1"
            
            # Time passes, exceeds TTL
            with patch('time.monotonic', return_value=5.1):
                assert cache.get("key1") is None
                # Verify it's actually removed from internal state
                assert cache.size() == 0

    def test_custom_per_key_ttl(self):
        """Test that custom TTL overrides default TTL."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # Set with custom short TTL
            cache.put("short", "val", ttl=2.0)
            # Set with default TTL
            cache.put("long", "val")
            
            # At t=3, 'short' should be expired, 'long' should be valid
            with patch('time.monotonic', return_value=3.0):
                assert cache.get("short") is None
                assert cache.get("long") == "val"

    def test_delete(self):
        """Test the delete method."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Deleting non-existent key
        assert cache.delete("non_existent") is False

    def test_size_with_mixed_expired_valid_items(self):
        """Test size() returns count of non-expired items and cleans up expired ones."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=5, default_ttl=5.0)
            
            cache.put("valid1", "v1")
            cache.put("valid2", "v2")
            cache.put("expiring", "v3", ttl=2.0)
            
            # At t=0, size should be 3
            assert cache.size() == 3
            
            # Advance time to expire one item
            with patch('time.monotonic', return_value=3.0):
                # size() should clean up 'expiring' and return 2
                assert cache.size() == 2
                
                # Verify the expired item is gone
                assert cache.get("expiring") is None
                assert cache.get("valid1") == "v1"
```

### How to Run

1.  Save the implementation code as `ttl_cache_impl.py`.
2.  Save the test code as `test_ttl_cache.py`.
3.  Install pytest if not already installed: `pip install pytest`.
4.  Run the tests: `pytest test_ttl_cache.py -v`.

### Key Design Decisions

1.  **Data Structures**:
    *   **Hash Map (`_map`)**: Stores `key -> Node` for O(1) access.
    *   **Doubly Linked List**: Maintains the order of usage. The head is the Most Recently Used (MRU), and the tail is the Least Recently Used (LRU). Dummy head/tail nodes simplify insertion/deletion logic.
2.  **Time Tracking**: `time.monotonic()` is used to ensure time is not affected by system clock adjustments.
3.  **Lazy Cleanup**:
    *   Expired items are not removed immediately upon expiration to maintain O(1) complexity for `put` and `get` without scanning the whole list.
    *   Cleanup happens during `get` (if the accessed item is expired), `put` (if the cache is full and needs space), and `size` (to return an accurate count).
4.  **Eviction Logic**: When the cache is full, `put` first attempts to clean up expired items. If space is still insufficient, it evicts the LRU item (the one immediately preceding the tail).
5.  **Type Hints**: Full type hinting is included for clarity and static analysis support.
