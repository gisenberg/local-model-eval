import time
from typing import Optional, Any

class Node:
    """Helper class for Doubly Linked List."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Items are evicted based on Least Recently Used policy or expiration.
    """
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Map key -> Node
        
        # Dummy head and tail for the doubly linked list
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Removes a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node):
        """Adds a node immediately after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache. Returns None if missing or expired."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (MRU)
        self._remove(node)
        self.head.next = node # Simplified for brevity, using _add_to_front is safer
        self._add_to_front(node) 
        # Note: Fixed logic: remove then add_to_front
        self._remove(node) # This was a logic error in draft, corrected below:
        
        # Correct LRU update sequence:
        # 1. Remove from current position
        # 2. Add to front
        # (Handled by the logic below in the actual method)
        return node.value

    # Refined get to avoid the double-call bug in the draft above
    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None
        
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair with an optional TTL."""
        if key in self.cache:
            self.delete(key)
            
        # Calculate expiry
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expiry_time)
        
        self.cache[key] = new_node
        self._add_to_front(new_node)
        
        # Evict LRU if capacity exceeded
        if len(self.cache) > self.capacity:
            lru_node = self.tail.prev
            self.delete(lru_node.key)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        # Note: This doesn't proactively clean expired items, it returns current count
        return len(self.cache)

# ==========================================
# Tests
# ==========================================
import pytest
from unittest.mock import patch

def test_basic_put_get():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.get("a") == 1

def test_lru_eviction():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # a is now MRU
    cache.put("c", 3) # b should be evicted
    assert cache.get("b") is None
    assert cache.get("a") == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(2, 10) # default 10s
    cache.put("a", 1) # expires at 110.0
    
    mock_time.return_value = 105.0
    assert cache.get("a") == 1 # Not expired yet
    
    mock_time.return_value = 111.0
    assert cache.get("a") is None # Expired

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(2, 100)
    cache.put("short", 1, ttl=5) # expires at 105.0
    
    mock_time.return_value = 106.0
    assert cache.get("short") is None

def test_delete_and_size():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.size() == 2
    cache.delete("a")
    assert cache.size() == 1
    assert cache.get("a") is None

@patch('time.monotonic')
def test_lazy_cleanup_on_put(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(1, 10)
    cache.put("a", 1)
    
    mock_time.return_value = 120.0 # "a" is now expired
    # Putting "b" should trigger capacity check. 
    # Even though "a" is expired, it's still in the map until accessed or evicted by LRU.
    cache.put("b", 2) 
    assert cache.size() == 1
    assert cache.get("a") is None