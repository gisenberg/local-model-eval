import time
from typing import Any, Optional


class Node:
    """Node for the doubly-linked list."""

    def __init__(self, key: Any, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class TTLCache:
    """LRU Cache with TTL (Time To Live)."""

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Hash map to store key-node pairs
        self.head = Node(None, None)  # Dummy head node
        self.tail = Node(None, None)  # Dummy tail node
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _add_node(self, node: Node) -> None:
        """Add a node right after the head (most recent)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: Node) -> None:
        """Move a node to the head (most recent)."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Node:
        """Pop the tail node (least recently used)."""
        res = self.tail.prev
        self._remove_node(res)
        return res

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on current time and its TTL."""
        return time.monotonic() > (node.ttl or self.default_ttl) + (node.ttl or 0)

    def get(self, key: Any) -> Any:
        """
        Get the value for the given key if it exists and is not expired.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or -1 if the key is not found or expired.
        """
        node = self.cache.get(key)
        if not node:
            return -1
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return -1
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update the value for the given key with an optional TTL.

        Args:
            key: The key to add or update.
            value: The value to associate with the key.
            ttl: Time-to-live in seconds for this entry. If None, uses the default TTL.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        node = Node(key, value, ttl)
        self.cache[key] = node
        self._add_node(node)
        self.size += 1
        if self.size > self.capacity:
            tail = self._pop_tail()
            del self.cache[tail.key]
            self.size -= 1

    def delete(self, key: Any) -> None:
        """
        Delete the entry with the given key if it exists.

        Args:
            key: The key to delete.
        """
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1

    def size(self) -> int:
        """Return the current size of the cache."""
        return self.size


# Pytest tests using unittest.mock.patch on time.monotonic
import pytest
from unittest.mock import patch


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_get_and_expiration(mock_monotonic):
    cache = TTLCache(2, 2)  # Capacity 2, default TTL 2 seconds
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.get('a') == 1  # Should be valid
    assert cache.get('b') == 2  # Should be valid
    assert cache.get('c') == -1  # Should not exist
    # After 2 seconds, 'a' should expire
    assert cache.get('a') == -1  # Expired


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_put_and_expiration(mock_monotonic):
    cache = TTLCache(2, 1)  # Capacity 2, default TTL 1 second
    cache.put('a', 1)
    assert cache.get('a') == 1  # Should be valid
    # After 1 second, 'a' should expire
    assert cache.get('a') == -1  # Expired


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_delete(mock_monotonic):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') == -1  # Should not exist


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_size_and_capacity(mock_monotonic):
    cache = TTLCache(2, 1)
    assert cache.size() == 0
    cache.put('a', 1)
    assert cache.size() == 1
    cache.put('b', 2)
    assert cache.size() == 2
    cache.put('c', 3)  # Should evict 'a'
    assert cache.size() == 2
    assert cache.get('a') == -1  # 'a' should be evicted


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_lru_eviction(mock_monotonic):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # 'a' should be evicted due to LRU
    assert cache.get('a') == -1  # 'a' should be gone
    assert cache.get('b') == 2  # 'b' should still be there
    assert cache.get('c') == 3  # 'c' should be the most recent


@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_custom_ttl(mock_monotonic):
    cache = TTLCache(2, 10)  # Default TTL 10 seconds
    cache.put('a', 1, ttl=1)  # Custom TTL 1 second
    assert cache.get('a') == 1  # Should be valid
    # After 1 second, 'a' should expire
    assert cache.get('a') == -1  # Expired