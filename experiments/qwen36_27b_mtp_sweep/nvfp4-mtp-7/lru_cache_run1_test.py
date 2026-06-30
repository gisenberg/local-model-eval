import time
from typing import Any, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: K, value: V, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Uses a doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy cleanup for expired entries.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for entries.
        """
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[K, _Node] = {}
        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_front(self, node: _Node) -> None:
        """Insert node immediately after head (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _pop_tail(self) -> _Node:
        """Remove and return the node immediately before tail (LRU position)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to the front (MRU position)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if a node has expired."""
        return now >= node.expires_at

    def get(self, key: K) -> Optional[V]:
        """
        Retrieve a value by key. Returns None if key is missing or expired.
        Moves accessed item to MRU position. Lazy cleanup on access.
        """
        if key not in self._cache:
            return None
        node = self._cache[key]
        now = time.monotonic()
        if self._is_expired(node, now):
            self._remove_node(node)
            del self._cache[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists and is not expired, updates value and resets TTL.
        Evicts LRU items if capacity is exceeded. Lazy cleanup on eviction.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = now + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            if self._is_expired(node, now):
                self._remove_node(node)
                del self._cache[key]
            else:
                node.value = value
                node.expires_at = expires_at
                self._move_to_front(node)
                return

        new_node = _Node(key, value, expires_at)
        self._add_to_front(new_node)
        self._cache[key] = new_node

        # Evict LRU items if over capacity
        while len(self._cache) > self._capacity:
            evict_node = self._pop_tail()
            if evict_node.key in self._cache:
                del self._cache[evict_node.key]

    def delete(self, key: K) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

# Import your cache class here in a real project
# 

@patch('time.monotonic', side_effect=[1.0, 1.0])
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1


@patch('time.monotonic', side_effect=[0.0, 4.0, 6.0])
def test_ttl_expiration_on_get(mock_time):
    """Test that expired entries return None and are lazily cleaned."""
    cache = TTLCache(2, 5.0)
    cache.put('k', 'v')          # t=0, expires at 5.0
    assert cache.get('k') == 'v' # t=4, still valid
    assert cache.get('k') is None  # t=6, expired & cleaned


@patch('time.monotonic', side_effect=[0.0] * 6)
def test_lru_eviction(mock_time):
    """Test that least recently used items are evicted when capacity is exceeded."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3


@patch('time.monotonic', side_effect=[0.0, 1.0, 11.0])
def test_custom_ttl_override(mock_time):
    """Test that custom TTL overrides default TTL."""
    cache = TTLCache(2, 10.0)
    cache.put('k', 'v', ttl=20.0)  # Expires at t=20
    assert cache.get('k') == 'v'   # t=1, valid
    assert cache.get('k') == 'v'   # t=11, still valid


@patch('time.monotonic', side_effect=[0.0, 0.0])
def test_delete_operation(mock_time):
    """Test explicit deletion and size tracking."""
    cache = TTLCache(2, 10.0)
    cache.put('k', 'v')
    cache.delete('k')
    assert cache.get('k') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[0.0, 0.0, 5.0])
def test_size_with_lazy_cleanup(mock_time):
    """Test that size() reflects lazy cleanup of expired entries."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    # Access 'a' at t=5 triggers lazy cleanup
    assert cache.get('a') is None
    assert cache.size() == 1