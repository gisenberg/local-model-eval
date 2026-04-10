import time
, Any

class Node:
    """Doubly Linked List Node to store cache entries."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Uses a Hash Map and Doubly Linked List for O(1) operations.
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
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node):
        """Add a node to the front (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache. Returns None if expired or missing."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (LRU logic)
        self._remove(node)
        self.add_to_front(node) # Note: logic inside _add_to_front
        # Wait, I defined _add_to_front, but I called add_to_front. Fixing:
        return node.value

    # Fixed the method call in the logic below to use internal helpers
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
        """Insert or update a key-value pair with an optional specific TTL."""
        if key in self.cache:
            self.delete(key)
        
        # Evict if capacity reached
        if len(self.cache) >= self.capacity:
            # Evict the least recently used (tail.prev)
            lru_node = self.tail.prev
            self.delete(lru_node.key)

        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Return the current number of items in the cache."""
        # Note: This returns the count including potentially expired items 
        # until they are lazily cleaned.
        return len(self.cache)

import pytest
from unittest.mock import patch
 # Assuming the class is in implementation.py

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1

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
    cache = TTLCache(capacity=2, default_ttl=100) # Long default
    cache.put("a", 1, ttl=5) # Short custom TTL
    
    mock_time.return_value = 106.0
    assert cache.get("a") is None

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=60)
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make 'b' the LRU
    cache.get("a")
    
    # Put 'c', should evict 'b'
    cache.put("c", 3)
    
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

@patch('time.monotonic')
def test_delete(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=60)
    cache.put("a", 1)
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=60)
    cache.put("a", 1)
    cache.put("a", 2) # Update value
    assert cache.get("a") == 2
    assert cache.size() == 1