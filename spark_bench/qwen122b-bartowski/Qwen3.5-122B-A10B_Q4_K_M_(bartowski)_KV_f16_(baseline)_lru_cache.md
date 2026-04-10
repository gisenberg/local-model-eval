# Qwen3.5-122B-A10B Q4_K_M (bartowski) — KV f16 (baseline) — LRU Cache with TTL

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
        """Helper to get current time using monotonic clock."""
        return time.monotonic()

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

    def _evict_lru(self, current_time: float) -> None:
        """
        Evict the least recently used non-expired item.
        If all items are expired, clear them all.
        """
        # First, try to find a non-expired LRU item
        candidate = self._tail.prev
        expired_nodes = []

        while candidate != self._head:
            if candidate.ttl_expiry <= current_time:
                expired_nodes.append(candidate)
                candidate = candidate.prev
            else:
                # Found a valid LRU item, evict it
                self._remove_node(candidate)
                del self._map[candidate.key]
                return

        # If we reached here, all items in the list are expired (or list was empty)
        # Clear all expired items
        for node in expired_nodes:
            self._remove_node(node)
            del self._map[node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns the value if the key exists and is not expired.
        Accessing a key moves it to the most recently used position.
        Returns None if the key does not exist or is expired.
        """
        current_time = self._get_current_time()
        node = self._map.get(key)

        if not node:
            return None

        # Check expiration
        if node.ttl_expiry <= current_time:
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, update its value and TTL, and move to MRU.
        If the cache is at capacity, evict the LRU non-expired item first.
        If all items are expired, clear them before inserting.
        Custom TTL overrides the default TTL.
        """
        current_time = self._get_current_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl

        # Check if key already exists
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.ttl_expiry = expiry_time
            # Move to front
            self._remove_node(node)
            self._add_to_front(node)
            return

        # If key is new, check capacity
        if len(self._map) >= self.capacity:
            self._evict_lru(current_time)

        # Create new node
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
            self._remove_node(node)
            del self._map[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed during this traversal
        if they haven't been cleaned up by other operations.
        """
        current_time = self._get_current_time()
        valid_count = 0
        nodes_to_remove = []

        # Iterate through the map to check validity
        # Note: Iterating over a dict while modifying it is unsafe, 
        # so we collect keys to remove first.
        for key, node in list(self._map.items()):
            if node.ttl_expiry <= current_time:
                nodes_to_remove.append(key)
            else:
                valid_count += 1

        # Remove expired items
        for key in nodes_to_remove:
            node = self._map[key]
            self._remove_node(node)
            del self._map[key]

        return valid_count
```

### Test Suite

Here are the 6 pytest tests covering the required scenarios. They use `unittest.mock.patch` to control time deterministically.

```python
import pytest
from unittest.mock import patch
from typing import Any
from your_module import TTLCache  # Replace 'your_module' with the actual filename

# Helper to create a cache instance
def create_cache(capacity: int = 3, default_ttl: float = 10.0):
    return TTLCache(capacity, default_ttl)

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("a", 1)
        cache.put("b", 2)
        
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

def test_capacity_eviction_lru_order():
    """Test that LRU items are evicted when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    cache = create_cache(capacity=5, default_ttl=5.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("x", 100)
        assert cache.get("x") == 100
        
    # Simulate time passing (6 seconds later)
    with patch('your_module.time.monotonic', return_value=6.0):
        assert cache.get("x") is None  # Expired
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    cache = create_cache(capacity=5, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
    # At t=5: 'short' should be expired, 'long' should be valid
    with patch('your_module.time.monotonic', return_value=5.0):
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete():
    """Test manual deletion of keys."""
    cache = create_cache(capacity=3, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("key1", "val1")
        cache.put("key2", "val2")
        
        assert cache.delete("key1") is True
        assert cache.delete("key1") is False  # Already deleted
        assert cache.get("key1") is None
        assert cache.get("key2") == "val2"

def test_size_with_mixed_expired_valid():
    """Test size() returns count of valid items and cleans up expired ones."""
    cache = create_cache(capacity=5, default_ttl=5.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("valid1", 1)
        cache.put("valid2", 2)
        cache.put("exp1", 3, ttl=2.0)
        cache.put("exp2", 4, ttl=3.0)
        
    # At t=4: exp1 and exp2 are expired, valid1 and valid2 are valid
    with patch('your_module.time.monotonic', return_value=4.0):
        # size() should clean up expired items and return 2
        assert cache.size() == 2
        
        # Verify expired items are gone
        assert cache.get("exp1") is None
        assert cache.get("exp2") is None
        
        # Verify valid items remain
        assert cache.get("valid1") == 1
        assert cache.get("valid2") == 2
```

### Key Implementation Details

1.  **Data Structures**:
    *   **Hash Map (`_map`)**: Stores `key -> Node` for O(1) access.
    *   **Doubly Linked List**: Maintains the order of usage. The `head` points to the Most Recently Used (MRU) item, and the `tail` points to the Least Recently Used (LRU) item. Dummy head/tail nodes simplify insertion and deletion logic.
2.  **Time Tracking**: Uses `time.monotonic()` to ensure time is non-decreasing and relative to the system start, preventing issues with system clock adjustments.
3.  **Lazy Cleanup**: Expired items are not removed immediately upon expiration. They are removed when:
    *   `get()` is called on them.
    *   `put()` triggers eviction (if the cache is full).
    *   `size()` is called (which iterates and cleans up).
4.  **Eviction Logic**: When the cache is full, `_evict_lru` scans from the tail (LRU) upwards. If it finds a non-expired item, it evicts it. If it encounters only expired items, it removes them all before inserting the new item.
5.  **Type Hints**: Full type hinting is applied to arguments, return values, and internal class attributes.
6.  **Testing**: The tests mock `time.monotonic` to simulate time passing without using `time.sleep`, ensuring deterministic and fast test execution.
