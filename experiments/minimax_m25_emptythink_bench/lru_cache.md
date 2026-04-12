# MiniMax-M2.5 UD-Q3_K_XL (empty-think) — LRU Cache with TTL

## Output (12127 chars)

Here is a complete implementation of the `TTLCache` class meeting all the requirements, along with the requested pytest test suite.

### Implementation

```python
import time
from typing import Any, Optional, Dict
from collections import OrderedDict

class TTLCache:
    """
    A Least Recently Used (LRU) cache with time-to-live (TTL) expiration.
    
    Uses a hash map for O(1) lookups and a doubly-linked list to maintain
    access order. Expired items are lazily cleaned up on access.
    """
    
    class _Node:
        """Internal node for the doubly-linked list."""
        __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')
        
        def __init__(self, key: str, value: Any, expires_at: float):
            self.key = key
            self.value = value
            self.expires_at = expires_at
            self.prev: Optional['_Node'] = None
            self.next: Optional['_Node'] = None

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        
        # Hash map: key -> _Node
        self._cache: Dict[str, '_Node'] = {}
        
        # Doubly-linked list to track LRU order
        # Head is most recently used, Tail is least recently used
        self._head: Optional['_Node'] = None
        self._tail: Optional['_Node'] = None

    def _is_expired(self, node: '_Node') -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.expires_at

    def _move_to_front(self, node: '_Node') -> None:
        """Move a node to the front (most recently used) of the list."""
        if node is self._head:
            return
        
        # Remove node from its current position
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            # Node was tail
            self._tail = node.prev
        
        # Add to front
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        
        if self._tail is None:
            self._tail = node

    def _remove_node(self, node: '_Node') -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

    def _evict_lru(self) -> None:
        """Evict the least recently used (tail) non-expired item."""
        if not self._tail:
            return
        
        # Find the last non-expired node
        node = self._tail
        while node and self._is_expired(node):
            # If tail is expired, move tail back
            self._tail = node.prev
            if self._tail:
                self._tail.next = None
            else:
                self._head = None
            del self._cache[node.key]
            node = self._tail
        
        # If we exhausted the list and everything was expired
        if node is None:
            self._cache.clear()
            return

        # Evict the LRU node
        self._remove_node(node)
        del self._cache[node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns the value if the key exists and is not expired, otherwise None.
        Accessing a key makes it the most recently used item.
        """
        node = self._cache.get(key)
        if node is None:
            return None
        
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None
        
        # Move to front (most recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the cache is at capacity, the least recently used non-expired item
        is evicted. If all items are expired, they are cleared first.
        Custom ttl overrides the default_ttl.
        """
        # Check if key exists
        existing_node = self._cache.get(key)
        
        if existing_node:
            # Update existing node
            if self._is_expired(existing_node):
                # Treat as new insertion if expired
                self._remove_node(existing_node)
                del self._cache[key]
                existing_node = None
            else:
                existing_node.value = value
                existing_node.expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
                self._move_to_front(existing_node)
                return

        # Determine expiration time
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        
        # Create new node
        new_node = self._Node(key, value, expires_at)
        self._cache[key] = new_node
        self._move_to_front(new_node)
        
        # Evict if over capacity
        if len(self._cache) > self._capacity:
            self._evict_lru()

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed and was removed, False otherwise.
        """
        node = self._cache.get(key)
        if node is None:
            return False
        
        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup of expired items encountered at the tail.
        """
        # Clean up expired items from the tail
        while self._tail and self._is_expired(self._tail):
            self._tail = self._tail.prev
            if self._tail:
                self._tail.next = None
            else:
                self._head = None
        
        # Reconstruct cache map without expired items (lazy approach: 
        # actually we need to remove them from map too)
        # The above loop only adjusts the list. We need to clean the map.
        # A full cleanup is expensive (O(n)), but size() is allowed to do work.
        # However, to keep O(1) average for other ops, we usually clean only on access.
        # For size(), we can iterate the linked list which is O(current_items).
        
        count = 0
        current = self._head
        valid_keys = []
        while current:
            if not self._is_expired(current):
                count += 1
                valid_keys.append(current.key)
            current = current.next
        
        # Rebuild cache dict to exclude expired (should be rare)
        self._cache = {k: self._cache[k] for k in valid_keys}
        return count
```

### Tests

```python
import pytest
from unittest.mock import patch
import time

# Assuming the TTLCache class is in the same file or imported
# from your_module import TTLCache 

def test_basic_get_put():
    """Test basic get and put operations."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        cache.put("b", 2)
        assert cache.get("b") == 2
        assert cache.get("a") == 1

def test_capacity_eviction_lru_order():
    """Test that eviction removes the least recently used item."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Access 'a' to make it most recent
        cache.get("a")
        
        # Add one more, should evict 'b' (least recent)
        cache.put("d", 4)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

def test_ttl_expiry():
    """Test that items expire after TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key", "value")
        assert cache.get("key") == "value"
        
        # Advance time past expiry
        mock_time.return_value = 6.0
        assert cache.get("key") is None

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=2.0)
        
        cache.put("short", "val", ttl=1.0)
        cache.put("long", "val", ttl=10.0)
        
        # At t=1.5, short should expire, long should not
        mock_time.return_value = 1.5
        assert cache.get("short") is None
        assert cache.get("long") == "val"
        
        # At t=3.0, long should expire
        mock_time.return_value = 3.0
        assert cache.get("long") is None

def test_delete():
    """Test deletion of existing and non-existing keys."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        
        assert cache.delete("a") is False # Already deleted
        assert cache.delete("missing") is False

def test_size_with_mixed_expired_valid():
    """Test size calculation with a mix of expired and valid items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=10, default_ttl=5.0)
        
        cache.put("a", 1)      # Expires at 5
        cache.put("b", 2)      # Expires at 5
        cache.put("c", 3, ttl=100) # Expires at 100
        
        mock_time.return_value = 6.0
        
        # 'a' and 'b' expired, 'c' valid
        assert cache.size() == 1
        
        # Add more valid items
        cache.put("d", 4)
        assert cache.size() == 2
        
        # Verify we can still evict correctly if needed
        cache.put("e", 5)
        cache.put("f", 6)
        # Capacity is 10, we have 4 valid, adding more is fine
        assert cache.size() == 4
```

### Key Implementation Details

1.  **Data Structures**:
    *   **Hash Map (`_cache`)**: Maps keys to node objects for O(1) access.
    *   **Doubly Linked List**: Maintains LRU order. `self._head` is the Most Recently Used (MRU), `self._tail` is the Least Recently Used (LRU).

2.  **Time Management**:
    *   Uses `time.monotonic()` to track expiration times (`expires_at`).
    *   Expiration checks happen on `get`, `put`, and `size`.

3.  **Eviction Logic**:
    *   On `put`, if `len(self._cache) > capacity`, `_evict_lru` is called.
    *   `_evict_lru` removes items from the tail. It handles the edge case where *all* items in the cache are expired (it clears the map and resets the list).

4.  **Lazy Cleanup**:
    *   `get` removes expired items immediately upon access.
    *   `size` iterates the list to count valid items and rebuilds the map to ensure consistency.

5.  **Complexity**:
    *   All operations (`get`, `put`, `delete`, `size`) are O(1) average, assuming hash map collisions are negligible. `size()` iterates over valid nodes which is O(k) where k is current valid items, but this is necessary to return an accurate count without maintaining a separate counter that might drift due to lazy expiration.
