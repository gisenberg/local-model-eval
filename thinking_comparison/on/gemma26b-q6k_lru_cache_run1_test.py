import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list representing a cache entry."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU Cache with time-based expiration (TTL).
    Uses a doubly-linked list and a hash map to maintain O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        :param capacity: Maximum number of non-expired items allowed.
        :param default_ttl: Default time in seconds until an item expires.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to simplify doubly-linked list operations
        self.head: Node = Node("", None, 0.0)
        self.tail: Node = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Return the value if the key exists and is not expired, else None.
        Accessing a key makes it the most recently used.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() >= node.expiry:
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If capacity is reached, evicts the least-recently-used non-expired item.
        If all items are expired, clears them first.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        if len(self.cache) >= self.capacity:
            # 1. Try to clear expired items from the tail (LRU side) to make room
            while self.tail.prev != self.head and self.tail.prev.expiry <= now:
                self.delete(self.tail.prev.key)
            
            # 2. If still at capacity, evict the LRU (the tail)
            if len(self.cache) >= self.capacity:
                self.delete(self.tail.prev.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if it existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return the count of items in the cache (lazy cleanup applied)."""
        return len(self.cache)

# --- Tests ---

import pytest
from unittest.mock import patch

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.size() == 2

def test_capacity_eviction():
    """Test that the least recently used item is evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3)  # Should evict 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after their TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        
        # Advance time past TTL
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    """Test that custom TTL overrides the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("default", 1) # expires at 110
        cache.put("custom", 2, ttl=2) # expires at 102
        
        mock_time.return_value = 105.0
        assert cache.get("default") == 1
        assert cache.get("custom") is None

def test_delete():
    """Test the delete functionality."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed():
    """Test size with a mix of valid and expired items (lazy cleanup)."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        
        cache.put("a", 1) # valid
        cache.put("b", 2, ttl=1) # expires at 101
        cache.put("c", 3) # valid
        
        mock_time.return_value = 105.0
        # 'b' is expired but not yet removed (lazy cleanup)
        # size() returns len(cache) which includes 'b' until accessed
        assert cache.size() == 3
        
        # Accessing 'b' triggers lazy cleanup
        assert cache.get("b") is None
        assert cache.size() == 2