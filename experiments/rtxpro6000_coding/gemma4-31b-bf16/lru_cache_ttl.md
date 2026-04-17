```python
import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list to track LRU order and expiration."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Maintains O(1) average time complexity for all operations.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to avoid null checks during DLL operations
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has exceeded its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Updates the item to be the most recently used.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None
        
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. 
        Evicts LRU non-expired items if capacity is reached.
        """
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._remove(node)
            self._add_to_front(node)
            return

        # Handle capacity
        if len(self.cache) >= self.capacity:
            # 1. Lazy cleanup: remove expired items from the tail (LRU) first
            while self.cache:
                lru_node = self.tail.prev
                if lru_node == self.head: # Cache is empty
                    break
                if self._is_expired(lru_node):
                    self.delete(lru_node.key)
                else:
                    break
            
            # 2. If still at capacity, evict the least-recently-used non-expired item
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if key existed."""
        if key not in self.cache:
            return False
        
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """
        Return count of non-expired items. 
        Note: Performs lazy cleanup on the tail to maintain approximate accuracy.
        """
        # To keep O(1), we clean the tail if it's expired before returning size
        while self.cache:
            lru_node = self.tail.prev
            if lru_node == self.head:
                break
            if self._is_expired(lru_node):
                self.delete(lru_node.key)
            else:
                break
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_get_put(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

@patch('time.monotonic')
def test_capacity_eviction(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

@patch('time.monotonic')
def test_ttl_expiry(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    
    mock_time.return_value = 111.0 # Past TTL
    assert cache.get("a") is None

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("short", 1, ttl=2)
    cache.put("long", 2, ttl=20)
    
    mock_time.return_value = 105.0
    assert cache.get("short") is None
    assert cache.get("long") == 2

@patch('time.monotonic')
def test_delete(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

@patch('time.monotonic')
def test_size_mixed_expiry(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=5, default_ttl=10)
    cache.put("a", 1, ttl=5)  # expires 105
    cache.put("b", 2, ttl=15) # expires 115
    cache.put("c", 3, ttl=20) # expires 120
    
    mock_time.return_value = 110.0
    # 'a' is expired, 'b' and 'c' are valid
    # size() should trigger lazy cleanup of 'a'
    assert cache.size() == 2
    assert cache.get("a") is None
    assert cache.get("b") == 2
```