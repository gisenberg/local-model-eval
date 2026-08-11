import time
from typing import Any, Optional


class Node:
    """Node for the doubly linked list."""
    def __init__(self, key: Any = None, value: Any = None, expire_at: float = 0):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: 'Node' = None
        self.next: 'Node' = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    Uses a doubly linked list and a hash map for O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")
        if default_ttl < 0:
            raise ValueError("Default TTL must be non-negative.")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: dict[Any, Node] = {}
        # Sentinel nodes to simplify edge cases
        self.head: Node = Node()
        self.tail: Node = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: Node) -> None:
        """Add a node right after the head (most recently used)."""
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
        """Move a node to the head (most recently used)."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Node:
        """Remove and return the least recently used node."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired."""
        return node.expire_at != 0 and time.monotonic() >= node.expire_at

    def get(self, key: Any) -> Any:
        """
        Get an item from the cache by key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or -1 if not found or expired.
        """
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                self.delete(key)
                return -1
            self._move_to_head(node)
            return node.value
        return -1

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put an item into the cache.

        Args:
            key: The key to set.
            value: The value to associate with the key.
            ttl: Optional time-to-live in seconds. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        actual_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = current_time + actual_ttl if actual_ttl > 0 else 0

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
        else:
            node = Node(key, value, expire_at)
            self.cache[key] = node
            self._add_node(node)

            if len(self.cache) > self.capacity:
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]

    def delete(self, key: Any) -> None:
        """
        Delete an item from the cache by key.

        Args:
            key: The key to delete.
        """
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)

    def size(self) -> int:
        """
        Return the number of items currently in the cache.

        Returns:
            The number of items in the cache.
        """
        return len(self.cache)

import time
import pytest
from unittest.mock import patch


def test_get_existing_key():
    """Test getting an existing key returns its value."""
    with patch('time.monotonic', return_value=1000.0):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put('a', 1)
        assert cache.get('a') == 1


def test_get_nonexistent_key():
    """Test getting a non-existent key returns -1."""
    with patch('time.monotonic', return_value=1000.0):
        cache = TTLCache(capacity=2, default_ttl=5)
        assert cache.get('nonexistent') == -1


def test_put_and_evict_lru():
    """Test that putting more items than capacity evicts the LRU item."""
    with patch('time.monotonic', return_value=1000.0):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        assert cache.get('a') == -1
        assert cache.get('b') == 2
        assert cache.get('c') == 3


def test_ttl_expiration():
    """Test that items expire after their TTL."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put('a', 1)

        # Item should be valid at t=1004
        mock_monotonic.return_value = 1004.0
        assert cache.get('a') == 1

        # Item should be expired at t=1006
        mock_monotonic.return_value = 1006.0
        assert cache.get('a') == -1


def test_custom_ttl():
    """Test that a custom TTL overrides the default."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1, ttl=2)  # Custom TTL of 2 seconds

        # Item should be valid at t=1001
        mock_monotonic.return_value = 1001.0
        assert cache.get('a') == 1

        # Item should be expired at t=1003
        mock_monotonic.return_value = 1003.0
        assert cache.get('a') == -1


def test_delete_key():
    """Test deleting a key removes it from the cache."""
    with patch('time.monotonic', return_value=1000.0):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') == -1
        assert cache.size() == 0