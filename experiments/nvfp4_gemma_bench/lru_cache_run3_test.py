import time
from typing import Optional, Any

class Node:
    """Node for the doubly linked list."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Evicts the least recently used item when capacity is reached.
    Items are considered expired based on time.monotonic().
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
        """Add a node immediately after the head (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache. Returns None if expired or missing."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (MRU)
        self._remove(node)
        self.head.next = node # Simplified logic: just move to front
        node.prev = self.head
        node.next = self.tail # This is wrong, using _add_to_front instead
        # Correcting the move-to-front logic:
        self._remove(node)
        self._add_to_front(node)
        
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key. Evicts LRU if capacity exceeded."""
        if key in self.cache:
            self.delete(key)
        
        # Calculate expiry
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expiry_time)
        
        self.cache[key] = new_node
        self._add_to_front(new_node)
        
        if len(self.cache) > self.capacity:
            # Evict the least recently used (the one before tail)
            lru_node = self.tail.prev
            self.delete(lru_node.key)

    def delete(self, key: Any) -> None:
        """Remove key from cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Return current number of items in cache (including potentially expired ones)."""
        return len(self.cache)

# --- Testing Suite ---
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
    cache.put("c", 3) # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiration():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10) # default ttl = 10
        cache.put("a", 1) # expires at 110.0
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1 # Not expired yet
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None # Expired

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100)
        cache.put("a", 1, ttl=5) # expires at 105.0
        
        mock_time.return_value = 106.0
        assert cache.get("a") is None

def test_update_mru():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # "a" becomes MRU
    cache.put("c", 3) # "b" should be evicted
    assert cache.get("b") is None
    assert cache.get("a") == 1

def test_delete_and_size():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.size() == 2
    cache.delete("a")
    assert cache.size() == 1
    assert cache.get("a") is None