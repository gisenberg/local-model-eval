# Gemma 4 31B 4bit MLX — lru_cache

## Thinking (3070 chars)

```


*   Goal: Implement an LRU (Least Recently Used) cache with TTL (Time To Live) in Python.
*   Requirements:
    *   Class `TTLCache` with `__init__(capacity, default_ttl)`.
    *   Methods: `get(key)`, `put(key, value, ttl=None)`, `delete(key)`, `size()`.
    *   Complexity: $O(1)$ average time.
    *   Implementation details: Doubly-linked list + hash map (no `OrderedDict`).
    *   Time tracking: `time.monotonic()`.
    *   Cleanup: Lazy cleanup on access.
    *   Coding standards: Type hints, docstrings.
    *   Testing: 6 `pytest` tests using `unittest.mock.patch` on `time.monotonic`.

    *   *LRU Logic:* Use a doubly-linked list to track usage. The head is the most recently used (MRU), and the tail is the least recently used (LRU).
    *   *TTL Logic:* Store the expiration timestamp along with the value in the node.
    *   *Hash Map:* Maps keys to nodes in the doubly-linked list for $O(1)$ access.
    *   *Lazy Cleanup:* When `get` or `put` is called, check if the accessed node has expired. If so, treat it as if it doesn't exist (delete it).

    *   `Node` class: `key`, `value`, `expiry`, `prev`, `next`.
    *   `TTLCache` class:
        *   `capacity`: Max items.
        *   `default_ttl`: Default time to live.
        *   `cache`: Dictionary `{key: Node}`.
        *   `head`, `tail`: Dummy nodes to simplify edge cases.

    *   `_remove(node)`: Unlinks a node from the list.
    *   `_add_to_front(node)`: Inserts a node after the dummy head.
    *   `_is_expired(node)`: Checks `time.monotonic() > node.expiry`.

    *   `get(key)`:
        1. Check if key exists.
        2. If exists, check if expired.
        3. If expired, delete and return `None`.
        4. If not expired, move to front and return value.
    *   `put(key, value, ttl=None)`:
        1. If key exists, remove old node.
        2. Calculate expiry using `ttl` or `default_ttl`.
        3. If size exceeds capacity, remove the tail (LRU).
        4. Add new node to front.
    *   `delete(key)`:
        1. If key exists, remove node and return `True`.
        2. Else return `False`.
    *   `size()`:
        1. Return length of the dictionary. *Self-correction:* Since it's lazy cleanup, `size()` might return expired items. However, usually, `size()` in these contexts refers to the current map size. To be strictly accurate, one could iterate and clean, but that's $O(N)$. I'll stick to the map size as per standard LRU implementations, but I'll ensure `get` and `put` handle the lazy cleanup.

    *   Need to mock `time.monotonic`.
    *   Test cases:
        1. Basic put/get.
        2. LRU eviction (capacity limit).
        3. TTL expiration (lazy cleanup).
        4. Custom TTL per item.
        5. Deletion.
        6. Updating an existing key (resets LRU position and TTL).

    *   *Edge case:* What if the item being evicted by LRU is already expired? That's fine.
    *   *Edge case:* What if the item being accessed is expired? Delete it and return `None`.
    *   *Edge case:* `put` on an existing key. Update value, update TTL, move to front.
```

## Output

```python
import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Uses a doubly-linked list for LRU tracking and a hash map for O(1) access.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache.
        
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail to avoid null checks during node manipulation
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache. 
        Returns None if the key is missing or expired.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (MRU)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.
        
        :param key: The key to store.
        :param value: The value to store.
        :param ttl: Optional custom TTL for this specific item.
        """
        if key in self.cache:
            self.delete(key)
        
        # Evict LRU if capacity reached
        if len(self.cache) >= self.capacity:
            # The node before the dummy tail is the LRU
            lru_node = self.tail.prev
            if lru_node:
                self.delete(lru_node.key)
        
        # Calculate expiration time
        duration = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + duration
        
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """Remove an item from the cache. Returns True if item existed."""
        if key not in self.cache:
            return False
        
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

# ==============================================================================
# Pytest Tests
# ==============================================================================
import pytest
from unittest.mock import patch

def test_basic_put_get():
    """Test that basic put and get operations work."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1

def test_lru_eviction():
    """Test that the least recently used item is evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' becomes MRU, 'b' becomes LRU
    cache.put("c", 3) # 'b' should be evicted
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that items expire after the default TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    
    cache.put("a", 1) # Expires at 105.0
    
    mock_time.return_value = 104.0
    assert cache.get("a") == 1 # Not expired yet
    
    mock_time.return_value = 106.0
    assert cache.get("a") is None # Expired

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    """Test that items can have a custom TTL overriding the default."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=100)
    
    cache.put("short", 1, ttl=1) # Expires at 101.0
    cache.put("long", 2, ttl=10) # Expires at 110.0
    
    mock_time.return_value = 102.0
    assert cache.get("short") is None
    assert cache.get("long") == 2

def test_delete_functionality():
    """Test that the delete method removes items correctly."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.size() == 0
    assert cache.delete("non_existent") is False

@patch('time.monotonic')
def test_update_resets_lru_and_ttl(mock_time):
    """Test that updating a key resets its position and expiration."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    
    cache.put("a", 1) # Expires 110
    cache.put("b", 2) # Expires 110
    
    mock_time.return_value = 105.0
    cache.put("a", 10, ttl=10) # 'a' updated, now expires 115, and is MRU
    
    mock_time.return_value = 112.0
    # 'b' should have expired (110), 'a' should still be alive (115)
    assert cache.get("b") is None
    assert cache.get("a") == 10
```
