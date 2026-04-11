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
        
        # Dummy head and tail for the doubly linked list
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
        """Add a node right after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value from cache. 
        Returns None if key doesn't exist or is expired.
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
        Insert or update a value. 
        If key exists, update value and move to front.
        If cache is full, evict the LRU item.
        """
        if key in self.cache:
            self.delete(key)
            
        # Calculate expiry
        ttl_value = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + ttl_value
        
        # Evict LRU if capacity reached
        if len(self.cache) >= self.capacity:
            # Evict the node before the dummy tail
            lru_node = self.tail.prev
            self.delete(lru_node.key)
            
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove an item from the cache."""
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
    # Capacity 2, should evict "a" when "c" is added
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # Access "a" to make it MRU
    cache.put("c", 3) # Should evict "b"
    
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

def test_ttl_expiration():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1) # Expires at 110.0
        
        # Move time forward by 5 seconds
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Move time forward past expiry (111 seconds)
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        # Override default_ttl with a short 5s TTL
        cache.put("short", "val", ttl=5)
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None

def test_delete_functionality():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.size() == 0

def test_update_existing_key():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("a", 2) # Update value
    assert cache.get("a") == 2
    assert cache.size() == 1