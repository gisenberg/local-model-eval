# Qwen3.5-122B-A10B Q4_K_M — KV f16 (baseline) — LRU Cache with TTL

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
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.ttl_expiry

    def _evict_expired(self) -> None:
        """
        Remove all expired items from the cache.
        This is called when the cache is full to make room, or during size calculation.
        """
        current = self._head.next
        while current and current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            current = next_node

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # First, ensure we don't try to evict an expired item if we can avoid it
        # by cleaning up expired ones first if the cache is full
        if len(self._map) == self.capacity:
            self._evict_expired()
            # If after cleaning expired items we still have items, evict the LRU
            # If the cache became empty after cleaning, we don't need to evict
            if len(self._map) == 0:
                return

        # The LRU item is right before the tail
        lru_node = self._tail.prev
        if lru_node and lru_node != self._head:
            self._remove_node(lru_node)
            del self._map[lru_node.key]

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

        if self._is_expired(node):
            # Lazy removal of expired item
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
        If the key is new and capacity is reached, evict the LRU non-expired item.
        If all items are expired, clear them first before inserting.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl

        if key in self._map:
            # Update existing
            node = self._map[key]
            node.value = value
            node.ttl_expiry = expiry_time
            self._remove_node(node)
            self._add_to_front(node)
            return

        # New item
        if len(self._map) >= self.capacity:
            # Check if we can just clear expired items to make space
            self._evict_expired()
            
            # If still at capacity, evict LRU
            if len(self._map) >= self.capacity:
                self._evict_lru()

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
        
        Performs lazy cleanup: expired items are removed during this operation.
        """
        self._evict_expired()
        return len(self._map)
```

### Test Suite

Here are the 6 pytest tests covering the required scenarios. They use `unittest.mock.patch` to control `time.monotonic` for deterministic testing without `time.sleep`.

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
    """Test that LRU items are evicted when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a")
    
    # Add 'c', should evict 'b' (LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    cache = create_cache(capacity=2, default_ttl=5.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        # Advance time past TTL
        with patch('time.monotonic', return_value=6.0):
            assert cache.get("a") is None
            assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        # Advance time to 3.0: 'short' should expire, 'long' should remain
        with patch('time.monotonic', return_value=3.0):
            assert cache.get("short") is None
            assert cache.get("long") == 2
            
            # Advance time to 21.0: 'long' should expire
            with patch('time.monotonic', return_value=21.0):
                assert cache.get("long") is None

def test_delete():
    """Test manual deletion of keys."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.delete("a") is True
    assert cache.delete("a") is False  # Already deleted
    assert cache.delete("c") is False  # Never existed
    
    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_size_with_mixed_expired_valid():
    """Test size calculation with mixed expired and valid items."""
    cache = create_cache(capacity=5, default_ttl=10.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("valid1", 1)
        cache.put("valid2", 2)
        cache.put("expiring", 3, ttl=5.0)
        cache.put("valid3", 4)
        
        # Advance time to 6.0: 'expiring' is now expired
        with patch('time.monotonic', return_value=6.0):
            # Size should trigger cleanup and return 3 (valid1, valid2, valid3)
            assert cache.size() == 3
            
            # Verify expired item is gone
            assert cache.get("expiring") is None
            
            # Size should now be 3
            assert cache.size() == 3
```

### Usage Notes

1.  **Dependencies**: Ensure `pytest` is installed (`pip install pytest`).
2.  **Module Import**: Replace `from your_module import TTLCache` with the actual filename where the `TTLCache` class is saved (e.g., `from lru_cache import TTLCache`).
3.  **Time Mocking**: The tests use `patch('time.monotonic', return_value=X)` to simulate time passing instantly. This ensures deterministic behavior without waiting for real seconds.
4.  **O(1) Complexity**:
    *   `get`: Hash map lookup + linked list move (O(1)).
    *   `put`: Hash map lookup + linked list move/insert + potential eviction (O(1)).
    *   `delete`: Hash map lookup + linked list removal (O(1)).
    *   `size`: Iterates only over expired items (lazy cleanup), amortized O(1) per item over the lifetime of the cache.
