import time
from typing import Any, Optional


class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    Uses a custom doubly-linked list and hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed or evicted.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def _add_to_front(self, node: _Node) -> None:
        """Insert node immediately after head (most recently used)."""
        nxt = self._head.next
        node.next = nxt
        node.prev = self._head
        self._head.next = node
        nxt.prev = node
        self._size += 1

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._size -= 1

    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to the front of the list."""
        self._remove_node(node)
        self._add_to_front(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the tail node (least recently used)."""
        if self._size == 0:
            raise IndexError("Cache is empty")
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return node.expiry <= time.monotonic()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed key to front (LRU update). Lazy cleanup on access.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found/expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and refreshes TTL.
        If cache is full, evicts LRU entry.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + effective_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if self._size >= self.capacity:
                evicted = self._pop_tail()
                del self.cache[evicted.key]

            node = _Node(key, value, expiry)
            self.cache[key] = node
            self._add_to_front(node)

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The cache key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch


@pytest.fixture
def cache():
    return TTLCache(2, 10.0)


def test_basic_put_get(cache):
    """Test standard insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        assert cache.get('a') == 1


def test_ttl_expiration(cache):
    """Test that entries are lazily cleaned up when expired."""
    with patch('time.monotonic', side_effect=[0.0, 11.0]):
        cache.put('a', 1)
        assert cache.get('a') is None


def test_lru_eviction(cache):
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3


def test_update_refreshes_ttl(cache):
    """Test that updating an existing key refreshes its TTL."""
    with patch('time.monotonic', side_effect=[0.0, 5.0, 14.0]):
        cache.put('a', 1)
        cache.put('a', 2)  # Refreshes expiry to 15.0
        assert cache.get('a') == 2  # Still valid at t=14.0


def test_delete(cache):
    """Test explicit deletion and size tracking."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 0


def test_size_tracking_with_lazy_cleanup(cache):
    """Test that size accurately reflects lazy cleanup on access."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 11.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.size() == 2
        cache.get('a')  # Triggers lazy cleanup, removes expired 'a'
        assert cache.size() == 1