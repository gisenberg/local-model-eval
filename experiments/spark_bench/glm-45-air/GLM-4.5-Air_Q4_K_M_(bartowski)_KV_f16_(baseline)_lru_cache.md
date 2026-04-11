# GLM-4.5-Air Q4_K_M (bartowski) — KV f16 (baseline) — LRU Cache with TTL

## Output

Here's an implementation of the TTLCache class with the specified requirements, followed by pytest tests:

```python
import time
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

class Node:
    """Doubly-linked list node for LRU cache."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = None
        self.next = None

class TTLCache:
    """LRU cache with time-based expiration using doubly-linked list and hash map."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """Initialize cache with given capacity and default TTL in seconds."""
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
        
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head = Node("", "", 0)  # dummy head
        self.tail = Node("", "", 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _is_expired(self, node: Node, current_time: float) -> bool:
        """Check if a node has expired."""
        return current_time >= node.expiry
    
    def _cleanup_expired(self, current_time: float) -> None:
        """Remove all expired nodes from the cache."""
        # Iterate through the list and remove expired nodes
        node = self.tail.prev
        while node != self.head:
            if self._is_expired(node, current_time):
                prev_node = node.prev
                self._remove_node(node)
                del self.cache[node.key]
                node = prev_node
            else:
                break
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for key if exists and not expired, else None. Updates LRU order."""
        current_time = time.monotonic()
        node = self.cache.get(key)
        
        if node is None:
            return None
        
        if self._is_expired(node, current_time):
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to front (most recently used)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert/update key-value pair with optional TTL. Evicts LRU if at capacity."""
        current_time = time.monotonic()
        self._cleanup_expired(current_time)
        
        # If key exists, update value and TTL
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = current_time + (ttl if ttl is not None else self.default_ttl)
            self._remove_node(node)
            self._add_to_front(node)
            return
        
        # If at capacity, evict LRU
        if len(self.cache) >= self.capacity:
            lru_node = self.tail.prev
            if lru_node != self.head:
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
        
        # Add new node
        expiry = current_time + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)
    
    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if key existed."""
        node = self.cache.pop(key, None)
        if node is not None:
            self._remove_node(node)
            return True
        return False
    
    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup)."""
        current_time = time.monotonic()
        self._cleanup_expired(current_time)
        return len(self.cache)
```

Here are the pytest tests:

```python
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None

def test_capacity_eviction():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic', return_value=0):
        cache = TTLCache(2, 1)
        cache.put("a", 1)
        cache.put("b", 2)
    
    with patch('time.monotonic', return_value=1):
        assert cache.get("a") is None  # Expired
        assert cache.get("b") is None  # Expired
    
    with patch('time.monotonic', return_value=0.5):
        cache.put("c", 3)
        assert cache.get("c") == 3

def test_custom_ttl():
    with patch('time.monotonic', return_value=0):
        cache = TTLCache(2, 10)
        cache.put("a", 1, ttl=1)  # Custom TTL
        cache.put("b", 2)         # Default TTL
    
    with patch('time.monotonic', return_value=1):
        assert cache.get("a") is None  # Expired (custom TTL)
        assert cache.get("b") == 2    # Still valid (default TTL)

def test_delete():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_size_with_expired():
    with patch('time.monotonic', return_value=0):
        cache = TTLCache(3, 1)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
    
    with patch('time.monotonic', return_value=1):
        assert cache.size() == 0  # All expired
    
    with patch('time.monotonic', return_value=0.5):
        cache.put("d", 4)
        assert cache.size() == 1  # Only "d" is valid
```

### Explanation:

1. **Data Structures**:
   - `Node` class represents doubly-linked list nodes with key, value, expiry time, and prev/next pointers.
   - `TTLCache` uses a dictionary (`cache`) for O(1) access and a doubly-linked list for LRU ordering.

2. **Key Methods**:
   - `get()`: Checks expiry, updates LRU order, returns value or None.
   - `put()`: Handles updates, evictions, and adds new nodes with appropriate TTL.
   - `delete()`: Removes nodes from both dictionary and linked list.
   - `size()`: Performs lazy cleanup and returns current count of valid items.

3. **Time Handling**:
   - Uses `time.monotonic()` for consistent time tracking.
   - `_cleanup_expired()` removes expired nodes during operations.

4. **Tests**:
   - Mock `time.monotonic()` to control time deterministically.
   - Cover all requirements including LRU eviction, TTL expiry, custom TTL, deletion, and size with mixed expired/valid items.

This implementation ensures O(1) average time complexity for all operations while maintaining LRU order and TTL expiration.
