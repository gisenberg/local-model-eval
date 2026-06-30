import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    Uses a custom doubly-linked list and a hash map for O(1) average time complexity.
    Expired items are cleaned up lazily on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        
        # Sentinel nodes for clean DLL operations
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return node.expiry <= time.monotonic()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        Returns None if the key is not found or has expired.
        Moves the accessed item to the MRU position.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        if self._is_expired(node):
            self._remove(node)
            del self._cache[key]
            return None
            
        self._remove(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If the cache is at capacity, the LRU item is evicted.
        Uses custom ttl if provided, otherwise falls back to default_ttl.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._remove(node)
            self._add_to_head(node)
            return

        if len(self._cache) >= self.capacity:
            lru_node = self._tail.prev
            self._remove(lru_node)
            del self._cache[lru_node.key]

        new_node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self._add_to_head(new_node)
        self._cache[key] = new_node

    def delete(self, key: str) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove(node)
            del self._cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch
import time

# Assuming TTLCache is imported from the module above
# 
def test_basic_put_get():
    """Test basic insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration():
    """Test lazy cleanup when TTL expires."""
    # put at t=0 (expires at t=5), get at t=6 -> should return None
    with patch('time.monotonic', side_effect=[0.0, 6.0]):
        cache = TTLCache(2, 5.0)
        cache.put('a', 1)
        assert cache.get('a') is None
        assert cache.size() == 0

def test_lru_eviction():
    """Test LRU eviction when capacity is exceeded."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3

def test_custom_ttl_override():
    """Test custom TTL overriding default TTL."""
    # put at t=0 with ttl=2, get at t=1 (valid), get at t=3 (expired)
    with patch('time.monotonic', side_effect=[0.0, 1.0, 3.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1, ttl=2.0)
        assert cache.get('a') == 1
        assert cache.get('a') is None

def test_delete():
    """Test explicit deletion of a key."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.size() == 1

def test_size_and_update():
    """Test that updating existing keys doesn't increase size."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 100.0)
        cache.put('a', 1)
        cache.put('a', 2)  # Update existing
        cache.put('b', 3)
        assert cache.size() == 2
        assert cache.get('a') == 2
        assert cache.get('b') == 3