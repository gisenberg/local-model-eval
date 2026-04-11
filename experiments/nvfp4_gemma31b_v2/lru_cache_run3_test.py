import time
from typing import Any, Optional

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
    Items are considered expired based on time.monotonic().
    """
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Map[key, Node]
        
        # Dummy head and tail to simplify list operations
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache. Returns None if expired or not found."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (LRU update)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair with a specific or default TTL."""
        if key in self.cache:
            self.delete(key)
            
        # Calculate expiry time
        ttl_value = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + ttl_value
        
        # Handle capacity
        if len(self.cache) >= self.capacity:
            # Evict the least recently used (tail.prev)
            lru_node = self.tail.prev
            self.delete(lru_node.key)
            
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Return current number of items in cache (including potentially expired ones)."""
        return len(self.cache)

import pytest
from unittest.mock import patch

def test_basic_put_get():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1

def test_lru_eviction():
    # Capacity 2: "a" and "b" added, then "c" added should evict "a"
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiration():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0  # Start time
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1) # Expires at 110.0
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1 # Still valid
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None # Expired
        assert cache.size() == 0

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        # Override default TTL with a short one (1s)
        cache.put("short", "val", ttl=1) 
        
        mock_time.return_value = 101.1
        assert cache.get("short") is None

def test_lru_update_on_get():
    # Verify that getting an item moves it to the front, preventing eviction
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    
    cache.get("a") # "a" is now most recent, "b" is least recent
    cache.put("c", 3) # Should evict "b"
    
    assert cache.get("b") is None
    assert cache.get("a") == 1

def test_delete_method():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.size() == 1
    cache.delete("a")
    assert cache.size() == 0
    assert cache.get("a") is None