import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a doubly-linked list and a hash map for O(1) average time complexity
    on get, put, and delete operations. Implements lazy cleanup for expired entries.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes eliminate boundary checks in linked list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the tail (LRU position)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiration

    def _cleanup_node(self, node: _Node) -> None:
        """Remove a node from both the linked list and the hash map."""
        self._remove_node(node)
        del self.cache[node.key]

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
            self._cleanup_node(node)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl if None.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiration = time.monotonic() + effective_ttl
            self._move_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU node
                lru_node = self._pop_tail()
                del self.cache[lru_node.key]

            new_node = _Node(key, value, time.monotonic() + effective_ttl)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to remove.
        """
        if key in self.cache:
            node = self.cache[key]
            self._cleanup_node(node)

    def size(self) -> int:
        """Return the current number of items in the cache (including lazy-expired entries)."""
        return len(self.cache)

import pytest
from unittest.mock import patch
import time

# Adjust the patch target to match your actual module name if not running in __main__
# e.g., @patch('ttl_cache.time.monotonic')
@patch('time.monotonic')
def test_basic_put_and_get(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('key1', 'value1')
    assert cache.get('key1') == 'value1'
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('key1', 'value1')
    mock_monotonic.return_value = 6.0
    assert cache.get('key1') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('key1', 'value1', ttl=2.0)
    mock_monotonic.return_value = 3.0
    assert cache.get('key1') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction_on_capacity(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_delete_operation(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    cache.delete('nonexistent')  # Should handle gracefully without raising

@patch('time.monotonic')
def test_update_existing_key_resets_ttl(mock_monotonic):
    mock_monotonic.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('key1', 'v1')
    mock_monotonic.return_value = 3.0
    cache.put('key1', 'v2', ttl=10.0)
    mock_monotonic.return_value = 8.0
    assert cache.get('key1') == 'v2'
    assert cache.size() == 1