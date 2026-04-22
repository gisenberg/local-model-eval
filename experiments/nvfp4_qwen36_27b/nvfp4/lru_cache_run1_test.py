import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) pointer manipulation."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a custom doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Dummy head and tail simplify boundary conditions in the linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after the dummy head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL using monotonic time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed key to the most recently used position.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        # Lazy cleanup: remove if expired
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None

        # Move to head (most recently used)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and resets TTL. Moves to most recently used.
        If cache exceeds capacity, evicts least recently used item.
        """
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl
        new_node = _Node(key, value, expires_at)

        self._add_to_head(new_node)
        self._cache[key] = new_node

        if len(self._cache) > self._capacity:
            # Evict LRU (node immediately before dummy tail)
            lru_node = self._tail.prev
            self._remove_node(lru_node)
            del self._cache[lru_node.key]

    def delete(self, key: Any) -> bool:
        """Remove key from cache. Returns True if key existed, False otherwise."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            return True
        return False

    def size(self) -> int:
        """Return the current number of valid items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=2, default_ttl=10.0)

def test_basic_put_and_get(cache):
    """Test standard insertion and retrieval."""
    with patch('time.monotonic', return_value=100.0):
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration(cache):
    """Test that entries expire after default TTL."""
    # put(t=100), get(t=100 -> valid), get(t=115 -> expired)
    with patch('time.monotonic', side_effect=[100.0, 100.0, 115.0]):
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.get('a') is None  # Lazy cleanup triggers here
        assert cache.size() == 0

def test_custom_ttl_override(cache):
    """Test that custom TTL overrides default TTL."""
    # put(ttl=5, t=100), get(t=100 -> valid), get(t=105 -> expired)
    with patch('time.monotonic', side_effect=[100.0, 100.0, 105.0]):
        cache.put('a', 1, ttl=5.0)
        assert cache.get('a') == 1
        assert cache.get('a') is None  # Expires at 105, not 110

def test_lru_eviction(cache):
    """Test LRU eviction when capacity is exceeded."""
    with patch('time.monotonic', return_value=100.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a' (LRU)
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_delete_operation(cache):
    """Test explicit deletion and non-existent key handling."""
    with patch('time.monotonic', return_value=100.0):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.delete('a') is True
        assert cache.delete('nonexistent') is False
        assert cache.get('a') is None
        assert cache.size() == 1

def test_size_and_lazy_cleanup(cache):
    """Test that size reflects lazy cleanup behavior."""
    # put(t=100), put(t=100), get(t=115 -> expires 'a')
    with patch('time.monotonic', side_effect=[100.0, 100.0, 115.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.size() == 2
        assert cache.get('a') is None  # Triggers lazy cleanup
        assert cache.size() == 1       # Size updates after cleanup