# Qwen3.5-9B (L40S vLLM) — LRU Cache with TTL — Run 3

## Output

Here is the implementation of the `TTLCache` class followed by the comprehensive test suite.

### Implementation: `ttl_cache.py`

```python
import time
import time as time_module
from typing import Any, Optional

class Node:
    """Doubly linked list node to store key-value pairs and their timestamps."""
    def __init__(self, key: str, value: Any, timestamp: float):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a doubly-linked list and a hash map for O(1) average time complexity.
    Expiration is checked lazily on access or eviction.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
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
        self.tail: Optional[Node] = None  # Least Recently Used (potentially expired)

    def _get_current_time(self) -> float:
        """Returns the current monotonic time."""
        return time_module.monotonic()

    def _update_time(self):
        """Updates the internal clock to the current monotonic time."""
        self.current_time = self._get_current_time()

    def _evict_lru(self) -> Optional[Node]:
        """
        Removes the least recently used node from the list and map.
        Returns the evicted node, or None if the list is empty.
        """
        if self.tail is None:
            return None
        
        evicted_node = self.tail
        self._remove_node(self.tail)
        self.cache_map.pop(evicted_node.key, None)
        return evicted_node

    def _remove_node(self, node: Node):
        """Removes a node from the doubly linked list."""
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

    def _add_to_head(self, node: Node):
        """Adds a node to the front of the list (MRU position)."""
        if self.head is None:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            self.tail = node if self.tail is None else self.tail

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has expired based on its timestamp and TTL."""
        if node.timestamp is None:
            return False
        return (self.current_time - node.timestamp) >= node.ttl

    def _cleanup_expired(self, node: Optional[Node] = None) -> bool:
        """
        Removes expired nodes starting from the tail (LRU end).
        Returns True if any node was removed.
        """
        if node is None:
            node = self.tail
        
        while node and self._is_expired(node):
            self._remove_node(node)
            self.cache_map.pop(node.key, None)
            node = self.tail
            # If we removed the head, update head/tail pointers correctly
            # The loop condition handles the logic, but we need to ensure 
            # we don't get stuck if head/tail logic gets messy.
            # Since we always remove from tail, head only changes if list becomes empty.
            
        return True

    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value if the key exists and is not expired, otherwise None.
            If retrieved, the item is moved to the MRU position.
        """
        self._update_time()
        
        # Lazy cleanup: check if the tail is expired before accessing map
        if self.tail and self._is_expired(self.tail):
            self._cleanup_expired()
        
        if key not in self.cache_map:
            return None
        
        node = self.cache_map[key]
        
        # Check expiration
        if self._is_expired(node):
            self._remove_node(node)
            self.cache_map.pop(key)
            return None
        
        # Move to MRU
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Insert or update a key-value pair in the cache.
        
        Args:
            key: The key to store.
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
            # Move to MRU
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new
            new_node = Node(key, value, self.current_time)
            new_node.ttl = effective_ttl
            
            # Check capacity
            while len(self.cache_map) >= self.capacity:
                # If all items are expired, clear them all first
                if self.tail and self._is_expired(self.tail):
                    self._cleanup_expired()
                    continue
                
                # Evict LRU
                self._evict_lru()
            
            self._add_to_head(new_node)
            self.cache_map[key] = new_node

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        if key not in self.cache_map:
            return False
        
        node = self.cache_map[key]
        self._remove_node(node)
        self.cache_map.pop(key)
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.
        
        Performs lazy cleanup of expired items.
        """
        self._update_time()
        
        # Perform cleanup if the tail is expired
        if self.tail and self._is_expired(self.tail):
            self._cleanup_expired()
            
        return len(self.cache_map)
```

### Tests: `test_ttl_cache.py`

```python
import pytest
import unittest.mock as mock
from typing import Any, Optional

from ttl_cache import TTLCache, Node

class TestTTLCache:
    
    @pytest.fixture
    def mock_time(self):
        """Fixture to provide a mock for time.monotonic."""
        with mock.patch('ttl_cache.time.monotonic') as mock_monotonic:
            mock_monotonic.return_value = 100.0
            yield mock_monotonic

    def test_basic_get_put(self, mock_time):
        """Test basic insertion and retrieval."""
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Verify LRU order update
        cache.put("key2", "value2")
        cache.get("key1") # Access key1 again
        cache.put("key3", "value3")
        
        # key2 should be evicted before key1 because key1 was accessed recently
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"

    def test_capacity_eviction_lru_order(self, mock_time):
        """Test that LRU eviction works correctly when capacity is reached."""
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        # Fill cache
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 2
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_ttl_expiry(self, mock_time):
        """Test that items expire based on default TTL."""
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key1", "value1")
        
        # Initially valid
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.return_value = 105.0
        
        # Should be expired
        assert cache.get("key1") is None
        
        # Size should reflect expiration
        assert cache.size() == 0

    def test_custom_per_key_ttl(self, mock_time):
        """Test custom TTL overrides default TTL."""
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("short", "val", ttl=2.0)
        cache.put("long", "val", ttl=20.0)
        
        # Advance time to 12 seconds
        mock_time.return_value = 110.0
        
        # 'short' should be expired (100 + 2 = 102 < 110)
        assert cache.get("short") is None
        # 'long' should be valid (100 + 20 = 120 > 110)
        assert cache.get("long") == "val"

    def test_delete(self, mock_time):
        """Test delete operation."""
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Deleting non-existent key
        assert cache.delete("nonexistent") is False

    def test_size_with_mixed_expired_valid_items(self, mock_time):
        """Test size() returns count of non-expired items with lazy cleanup."""
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Initially size is 3
        assert cache.size() == 3
        
        # Advance time so 'a' and 'b' expire, 'c' is still valid
        mock_time.return_value = 105.0
        
        # Size should be 1 without accessing anything
        assert cache.size() == 1
        
        # Accessing 'c' should still return it
        assert cache.get("c") == 3
        
        # Accessing 'a' should return None and not affect size negatively
        assert cache.get("a") is None
        assert cache.size() == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### How to Run

1.  Save the first code block as `ttl_cache.py`.
2.  Save the second code block as `test_ttl_cache.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_ttl_cache.py -v
    ```

### Key Features Explained

*   **Doubly Linked List + Hash Map**: Ensures $O(1)$ insertion, deletion, and lookup. The `cache_map` provides direct access to nodes, while the linked list maintains LRU order.
*   **Lazy Cleanup**: The `size()`, `get()`, and `put()` methods check for expiration starting from the `tail` (LRU end). This avoids iterating through the entire list to find expired items, maintaining efficiency.
*   **Time-Based Expiry**: Uses `time.monotonic()` to prevent system clock adjustments from affecting cache behavior. The `current_time` is updated on every operation to ensure consistency.
*   **Capacity Handling**: When the cache is full, it checks if the tail is expired. If the tail (and potentially preceding nodes) are expired, they are cleaned up before evicting a new non-expired item, ensuring we don't evict valid data unnecessarily.
*   **Custom TTL**: The `put` method accepts an optional `ttl` argument, allowing specific keys to have different lifespans than the cache default.
