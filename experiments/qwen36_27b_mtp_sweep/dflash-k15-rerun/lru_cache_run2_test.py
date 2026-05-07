import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.

    Uses a hash map and a doubly-linked list to achieve O(1) average time
    complexity for get, put, and delete operations. Implements lazy cleanup
    by checking expiration times on access rather than using background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cached items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes for the doubly-linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the tail sentinel (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired entry
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Lazy cleanup: treat expired key as new insertion
                self._remove_node(node)
                del self.cache[key]
            else:
                # Update existing valid entry
                node.value = value
                node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_head(node)
                return

        # Evict least recently used item if at capacity
        if len(self.cache) >= self.capacity:
            tail = self._pop_tail()
            del self.cache[tail.key]

        # Insert new node
        actual_ttl = ttl if ttl is not None else self.default_ttl
        node = _Node(key, value, time.monotonic() + actual_ttl)
        self._add_to_head(node)
        self.cache[key] = node

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1, ttl=2.0)
    mock_time.return_value = 3.0
    assert cache.get('a') is None

@patch('time.monotonic')
def test_capacity_eviction_lru(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (least recently used)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_delete_operation(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_lazy_cleanup_on_put_full_capacity(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(1, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0
    # 'a' is expired. Putting 'b' should trigger lazy cleanup and evict 'a'
    cache.put('b', 2)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1