# Gemma 4 31B-IT Q4_K_M (f16 @ 32K, new base default ub) — lru_cache

**Hypothesis:** Refresh baseline: f16 at 32K works on new base. Is this faster or slower than turbo4's 11.8?

## Output

Here is the complete implementation of an LRU cache with TTL.

### Implementation Details
To achieve $O(1)$ average time complexity, I used a **Doubly Linked List** to track the order of access (LRU) and a **Hash Map** (Python dictionary) to store references to the nodes. 

The TTL is handled via **lazy cleanup**: when `get` or `put` is called, the system checks if the accessed item has expired. If it has, it is treated as if it doesn't exist and is removed.

```python
import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly linked list."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Evicts the least recently used item when capacity is reached.
    Items expire after their specific or default TTL.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        :param capacity: Maximum number of items to store.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail to simplify boundary conditions
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node immediately after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has exceeded its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache. 
        Returns None if key is missing or expired.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (Mark as recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.
        :param ttl: Optional override for the default_ttl.
        """
        if key in self.cache:
            self.delete(key)
        
        # Calculate expiry time
        ttl_value = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + ttl_value
        
        # Evict LRU if capacity reached
        if len(self.cache) >= self.capacity:
            # The node before the dummy tail is the least recently used
            lru_node = self.tail.prev
            self.delete(lru_node.key)
            
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """Remove a key from the cache. Returns True if key existed."""
        if key not in self.cache:
            return False
        
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return len(self.cache)
```

### Test Suite
The following tests use `pytest` and `unittest.mock.patch` to simulate the passage of time without actually sleeping.

```python
import pytest
from unittest.mock import patch

# Assuming the class above is in the same file or imported
# from ttl_cache import TTLCache

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    
    # Advance time past TTL
    mock_time.return_value = 111.0
    assert cache.get("a") is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=100)
    cache.put("short", 1, ttl=5)
    
    mock_time.return_value = 106.0
    assert cache.get("short") is None

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=100)
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make 'b' the LRU
    cache.get("a")
    
    # Put 'c', should evict 'b'
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

@patch('time.monotonic')
def test_delete(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("non_existent") is False

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("a", 2) # Update value
    assert cache.get("a") == 2
    assert cache.size() == 1
```

### Complexity Analysis
- **Time Complexity**: 
    - `get(key)`: $O(1)$ average. Dictionary lookup is $O(1)$, and updating the doubly linked list is $O(1)$.
    - `put(key, value)`: $O(1)$ average. Dictionary insertion and linked list manipulation are $O(1)$.
    - `delete(key)`: $O(1)$ average.
    - `size()`: $O(1)$.
- **Space Complexity**: $O(N)$ where $N$ is the capacity of the cache.
