import time
from typing import Any, Optional

class Node:
    """Doubly-linked list node to store cache entries."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL Cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Maps key -> Node
        
        # Dummy head and tail for the doubly-linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add(self, node: Node) -> None:
        """Add a node to the front (most-recently-used) of the list."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node
        self.tail.prev = node if self.capacity == 1 and self.tail.prev == self.head else self.tail.prev
        # Correcting tail pointer logic for edge cases
        self.tail.prev = node if self.tail.prev == self.head else self.tail.prev
        # Re-establishing tail link correctly
        curr = self.head.next
        while curr and curr.next != self.tail:
            curr = curr.next
        if curr:
            curr.next = self.tail
            self.tail.prev = curr

    def _add_to_front(self, node: Node) -> None:
        """Helper to place node immediately after head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has exceeded its expiry time."""
        return time.monotonic() > node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
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

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a value. Evicts LRU non-expired item if at capacity.
        If all items are expired, clears them first.
        """
        current_time = time.monotonic()
        expiry = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._remove(node)
            self._add_to_front(node)
            return

        # Handle capacity
        if len(self.cache) >= self.capacity:
            # 1. Try to find and remove any expired items first
            # We check the tail (LRU) for expiration
            while self.cache and self._is_expired(self.tail.prev):
                self.delete(self.tail.prev.key)
            
            # 2. If still at capacity, evict the LRU non-expired item
            if len(self.cache) >= self.capacity:
                # If the tail is still there and not expired, it's the LRU
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        # Create and add new node
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Return count of items currently in cache (lazy cleanup)."""
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, no expiration
    cache = TTLCache(2, 100.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10.0) # Expires at 110.0
        cache.put("a", 1)
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1 # Not expired yet
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None # Expired

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100.0) # Default long TTL
        cache.put("short", 1, ttl=1.0) # Expires at 101.0
        
        mock_time.return_value = 102.0
        assert cache.get("short") is None

def test_delete():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10.0)
        cache.put("a", 1) # Exp 110
        cache.put("b", 2) # Exp 110
        cache.put("c", 3) # Exp 110
        
        mock_time.return_value = 111.0
        # size() returns current map size due to lazy cleanup
        # but accessing 'a' should trigger cleanup
        assert cache.size() == 3 
        cache.get("a") # triggers cleanup of 'a'
        assert cache.size() == 2