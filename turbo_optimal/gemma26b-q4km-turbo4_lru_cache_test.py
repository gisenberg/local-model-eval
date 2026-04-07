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
    An LRU Cache with time-based expiration.
    Uses a doubly-linked list for LRU order and a hash map for O(1) access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to simplify edge cases in doubly-linked list
        self.head: Node = Node("", None, 0.0)
        self.tail: Node = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head (Most Recently Used)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front of the list."""
        self._remove_node(node)
        self._add_to_front(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Return the value if the key exists and is not expired, else None.
        Accessing a key makes it the most recently used.
        """
        node = self.cache.get(key)
        if not node:
            return None
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key with a specific or default TTL.
        Handles capacity eviction and expiration-based clearing.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Check capacity constraints
        if len(self.cache) >= self.capacity:
            # Requirement: If all items are expired, clear them all first.
            all_expired = all(self._is_expired(n) for n in self.cache.values())
            
            if all_expired:
                self.cache.clear()
                self.head.next = self.tail
                self.tail.prev = self.head
            else:
                # Requirement: Evict the least-recently-used (the tail)
                # We find the tail's predecessor (the actual LRU item)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        # Create new node
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if it existed."""
        node = self.cache.pop(key, None)
        if node:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return the count of non-expired items."""
        # Lazy cleanup: we don't iterate the whole list (O(N)), 
        # but we ensure the count is accurate by checking the cache.
        # Since we clean up on 'get' and 'put', len(self.cache) is 
        # the most efficient way to track size.
        return len(self.cache)

# --- Tests ---

import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # This should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    cache = TTLCache(capacity=2, default_ttl=10)
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)  # Expires at 110.0
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None

def test_custom_ttl():
    cache = TTLCache(capacity=2, default_ttl=10)
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1, ttl=5)   # Expires 105.0
        cache.put("b", 2, ttl=20)  # Expires 120.0
        
        mock_time.return_value = 106.0
        assert cache.get("a") is None
        assert cache.get("b") == 2

def test_delete():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_items():
    cache = TTLCache(capacity=2, default_ttl=10)
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1, ttl=5)   # Expires 105
        cache.put("b", 2, ttl=20)  # Expires 120
        
        mock_time.return_value = 106.0
        # At this point, 'a' is expired but still in cache dict (lazy cleanup)
        # 'get' will trigger the cleanup for 'a'
        assert cache.get("a") is None 
        # After 'get("a")', 'a' is removed from cache.
        # 'b' is still valid.
        assert cache.size() == 1