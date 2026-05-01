import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support.

    Uses a hash map and doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Add a node immediately after the head sentinel."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head of the list."""
        self._remove(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node immediately before the tail sentinel."""
        node = self._tail.prev
        self._remove(node)
        return node

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key.

        Returns the value if the key exists and is not expired, otherwise None.
        Moves the accessed item to the most recently used position.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if time.monotonic() >= node.expiration:
            # Lazy cleanup: remove expired entry
            self._remove(node)
            del self._cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL. Falls back to default_ttl if None.
        """
        if key in self._cache:
            node = self._cache[key]
            if time.monotonic() >= node.expiration:
                # Lazy cleanup on put: treat expired as miss
                self._remove(node)
                del self._cache[key]
            else:
                node.value = value
                node.expiration = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_head(node)
                return

        if len(self._cache) >= self.capacity:
            evicted = self._pop_tail()
            del self._cache[evicted.key]

        effective_ttl = ttl if ttl is not None else self.default_ttl
        node = _Node(key, value, time.monotonic() + effective_ttl)
        self._cache[key] = node
        self._add_to_head(node)

    def delete(self, key: Any) -> None:
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

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired entries return None and are removed on access."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that custom TTL in put() overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL, but before default
    assert cache.get('a') is None

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion removes key from cache and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_size_reflects_lazy_cleanup(mock_time):
    """Test that size() updates correctly after lazy cleanup triggers."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    
    mock_time.return_value = 6.0
    cache.get('a')  # Triggers lazy cleanup for 'a'
    assert cache.size() == 1