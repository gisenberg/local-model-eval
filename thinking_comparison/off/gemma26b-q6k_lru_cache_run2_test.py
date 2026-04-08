import time
from typing import Any, Optional, Dict


class Node:
    """A node in the doubly-linked list representing a cache entry."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    An LRU Cache with time-based expiration.
    Uses a Doubly Linked List for O(1) LRU management and a Hash Map for O(1) access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """Initialize the cache with a fixed capacity and default TTL."""
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Sentinel nodes for the doubly linked list
        self.head = Node("", None, 0)  # Most Recently Used
        self.tail = Node("", None, 0)  # Least Recently Used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() >= node.expiry

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        keys_to_del = [k for k, n in self.cache.items() if self._is_expired(n)]
        for k in keys_to_del:
            self.delete(k)

    def get(self, key: str) -> Optional[Any]:
        """Return value if exists and not expired, else None. Updates LRU status."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key. Evicts LRU if capacity is reached."""
        expiry_duration = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + expiry_duration

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._move_to_front(node)
            return

        # Check if we need to evict
        if len(self.cache) >= self.capacity:
            # First, try to clear all expired items
            self._cleanup_expired()
            
            # If still at capacity, evict the LRU (tail.prev)
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return count of non-expired items."""
        # Lazy cleanup: remove expired items before returning size
        self._cleanup_expired()
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
    """Test that the Least Recently Used item is evicted."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3)  # Should evict 'b'
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiry():
    """Test that items expire after their TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1)
        
        # Advance time but not enough to expire
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Advance time past TTL
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    """Test that per-key TTL overrides default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        cache.put("short", "val", ttl=5)
        cache.put("long", "val", ttl=50)
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None
        assert cache.get("long") == "val"

def test_delete():
    """Test the delete operation."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False

def test_size_mixed_expired():
    """Test size returns only non-expired items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        
        cache.put("a", 1) # expires 110
        cache.put("b", 2) # expires 110
        cache.put("c", 3, ttl=50) # expires 150
        
        mock_time.return_value = 115.0
        # 'a' and 'b' are expired, 'c' is valid
        assert cache.size() == 1
        assert "c" in cache.cache