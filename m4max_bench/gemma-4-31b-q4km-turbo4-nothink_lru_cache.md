# Gemma 4 31B-IT Q4_K_M (turbo4 KV) — lru_cache

## Output

Here is the complete implementation of an LRU Cache with TTL.

### Implementation Details
To achieve $O(1)$ average time complexity for all operations, I use a combination of a **Hash Map** (for fast lookup) and a **Doubly Linked List** (to track the order of access for LRU eviction). 

The TTL is handled via "lazy cleanup": when a key is accessed via `get`, the code checks if the current time exceeds the stored expiration time. If it does, the item is treated as missing and deleted.

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
    LRU Cache with Time-To-Live (TTL) expiration.
    Evicts the Least Recently Used item when capacity is reached.
    Items expire after their specific TTL or the default TTL.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail to simplify linked list operations
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has exceeded its TTL."""
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
        
        # Move to front (LRU logic)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.
        :param ttl: Optional override for the default TTL.
        """
        # Use provided ttl or fallback to default
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._remove(node)
            self._add_to_front(node)
        else:
            # Create new node
            if len(self.cache) >= self.capacity:
                # Evict Least Recently Used (node before dummy tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)
            
            new_node = Node(key, value, expiry_time)
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
        """Return the current number of items in the cache."""
        # Note: This includes expired items that haven't been lazily cleaned yet
        return len(self.cache)
```

### Test Suite
I have used `pytest` and `unittest.mock.patch` to simulate the passage of time without actually calling `time.sleep()`.

```python
import pytest
from unittest.mock import patch

# Assuming the class above is in the same file or imported
# from cache_impl import TTLCache

@pytest.fixture
def cache():
    # Capacity 2, Default TTL 10 seconds
    return TTLCache(2, 10)

def test_basic_put_get(cache):
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1

def test_lru_eviction(cache):
    """Test that the least recently used item is evicted when capacity is reached."""
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' becomes most recent, 'b' is now LRU
    cache.put("c", 3) # Should evict 'b'
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiration():
    """Test that items expire after the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        
        cache.put("a", 1)
        
        # Advance time by 5s (not expired)
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Advance time by 11s (expired)
        mock_time.return_value = 116.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    """Test that a specific TTL override works."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100) # Long default
        
        cache.put("short", 1, ttl=2) # Short custom TTL
        
        mock_time.return_value = 103.0
        assert cache.get("short") is None

def test_delete(cache):
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("nonexistent") is False

def test_update_refreshes_lru_and_ttl():
    """Updating a key should move it to front and reset its expiry."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        
        cache.put("a", 1)
        mock_time.return_value = 105.0
        cache.put("a", 2) # Update 'a' at t=105, new expiry is 115
        
        mock_time.return_value = 112.0 
        # If it used original TTL (110), it would be expired. 
        # But it should be alive until 115.
        assert cache.get("a") == 2
```

### Complexity Analysis
- **`get(key)`**: $O(1)$ average. Hash map lookup is $O(1)$, and moving a node in a doubly linked list is $O(1)$.
- **`put(key, value)`**: $O(1)$ average. Hash map insertion/update is $O(1)$, and adding/removing nodes from the linked list is $O(1)$.
- **`delete(key)`**: $O(1)$ average.
- **`size()`**: $O(1)$.
