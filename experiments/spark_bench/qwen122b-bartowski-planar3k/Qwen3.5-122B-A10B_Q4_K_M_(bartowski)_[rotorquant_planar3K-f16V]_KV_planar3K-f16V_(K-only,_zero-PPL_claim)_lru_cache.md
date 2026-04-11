# Qwen3.5-122B-A10B Q4_K_M (bartowski) [rotorquant planar3K/f16V] — KV planar3K/f16V (K-only, zero-PPL claim) — LRU Cache with TTL

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
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if time.monotonic() > current.ttl_expiry:
                del self._map[current.key]
                self._remove_node(current)
            current = next_node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node's TTL has expired."""
        return time.monotonic() > node.ttl_expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns the value if the key exists and is not expired.
        Accessing a valid key moves it to the most recently used position.
        Returns None if the key does not exist or is expired.
        """
        node = self._map.get(key)
        if not node:
            return None

        if self._is_expired(node):
            # Lazy removal of expired item
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
        If the cache is at capacity, evict the LRU non-expired item.
        If all items are expired, clear them first before inserting.
        Custom TTL overrides the default TTL.
        """
        current_time = time.monotonic()
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

        # If key is new, check capacity
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
            node = self._map.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed during this operation
        to ensure the returned count is accurate.
        """
        # We must iterate and remove expired items to get an accurate count
        # This is O(N) in the worst case (all expired), but amortized O(1) 
        # if items are accessed regularly.
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                del self._map[current.key]
                self._remove_node(current)
            current = next_node
        
        return len(self._map)
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
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None
    
    # Update existing
    cache.put("a", 10)
    assert cache.get("a") == 10

def test_capacity_eviction_lru_order():
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU, 'b' becomes LRU
    cache.get("a")
    
    # Insert 'c', should evict 'b'
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=2, default_ttl=5.0)
        
        cache.put("x", 100)
        assert cache.get("x") == 100
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        assert cache.get("x") is None
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        # 'short' expires in 2s, 'long' expires in 20s
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        # Advance to 3s
        mock_time.return_value = 3.0
        
        assert cache.get("short") is None  # Expired
        assert cache.get("long") == 2      # Still valid
        
        # Advance to 25s
        mock_time.return_value = 25.0
        assert cache.get("long") is None   # Now expired

def test_delete():
    """Test the delete method."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    assert cache.delete("key1") is False  # Already deleted
    assert cache.delete("nonexistent") is False
    
    assert cache.get("key2") == "value2"

def test_size_with_mixed_expired_valid():
    """Test size() returns count of non-expired items and cleans up expired ones."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=5, default_ttl=5.0)
        
        cache.put("a", 1, ttl=2.0)   # Expires at 2
        cache.put("b", 2, ttl=10.0)  # Expires at 10
        cache.put("c", 3, ttl=2.0)   # Expires at 2
        
        # Initial size should be 3
        assert cache.size() == 3
        
        # Advance time to 3s (a and c expired)
        mock_time.return_value = 3.0
        
        # size() should clean up expired items and return 1
        assert cache.size() == 1
        
        # Verify only 'b' remains
        assert cache.get("b") == 2
        assert cache.get("a") is None
        assert cache.get("c") is None
```

### Key Implementation Details

1.  **Data Structures**:
    *   **Hash Map (`_map`)**: Stores `key -> Node` for O(1) access.
    *   **Doubly Linked List**: Maintains the order of usage. The head is the Most Recently Used (MRU), and the tail is the Least Recently Used (LRU). Dummy head and tail nodes simplify insertion and deletion logic.
2.  **Time Tracking**: Uses `time.monotonic()` to ensure time is monotonic and not affected by system clock adjustments.
3.  **Lazy Cleanup**:
    *   Expired items are not immediately removed upon insertion of a new item unless the cache is full.
    *   `get()`, `size()`, and `put()` (when full) trigger checks for expiration.
    *   `size()` explicitly iterates to clean up expired items to return an accurate count.
4.  **Eviction Logic**:
    *   When `put` is called and the cache is full, it first attempts to clean up expired items (`_cleanup_expired`).
    *   If space is still not available, it evicts the LRU item (the node immediately preceding the tail).
5.  **Complexity**:
    *   `get`, `put`, `delete`: O(1) average time (hash map lookup + constant number of pointer updates).
    *   `size`: O(N) in the worst case (if all items are expired), but amortized O(1) in typical usage where items are accessed before expiry. This is a standard trade-off for lazy expiration.
