from __future__ import annotations
import time
from typing import Optional, Any, Dict, Tuple


class Node:
    """A node in the doubly-linked list."""
    __slots__ = ('key', 'value', 'expire_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class TTLCache:
    """
    An LRU (Least Recently Used) cache with Time-To-Live (TTL) support.

    Attributes:
        capacity (int): Maximum number of items.
        default_ttl (float): Default time-to-live in seconds.
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("Default TTL cannot be negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node(None, None, 0)  # Dummy head
        self.tail = Node(None, None, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front (most recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() > node.expire_at

    def get(self, key: Any) -> Any:
        """
        Get the value for a key. Returns -1 if not found or expired.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or -1 if not found/expired.
        """
        if key not in self.cache:
            return -1

        node = self.cache[key]

        if self._is_expired(node):
            self.delete(key)
            return -1

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        Args:
            key: The key.
            value: The value.
            ttl: Optional custom TTL. Defaults to default_ttl.
        """
        expire_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict least recently used (tail.prev)
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

            new_node = Node(key, value, expire_at)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Delete a key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Return current number of valid (non-expired) items."""
        return len(self.cache)

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries cleaned up.
        """
        removed = 0
        keys_to_delete = [k for k, v in self.cache.items() if self._is_expired(v)]
        for k in keys_to_delete:
            self.delete(k)
            removed += 1
        return removed

import pytest
from unittest.mock import patch
import time

# Import your class here
from typing import Any


# Mock time.monotonic
@patch('__main__.time.monotonic')
def test_get_existing_key(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1)
    assert cache.get("a") == 1


@patch('__main__.time.monotonic')
def test_get_expired_key(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1)
    mock_time.return_value = 106.0  # After expiration
    assert cache.get("a") == -1


@patch('__main__.time.monotonic')
def test_put_updates_existing(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1)
    cache.put("a", 2)
    assert cache.get("a") == 2


@patch('__main__.time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # Access 'a' so it becomes MRU
    cache.put("c", 3)  # Should evict 'b'
    assert cache.get("b") == -1
    assert cache.get("a") == 1
    assert cache.get("c") == 3


@patch('__main__.time.monotonic')
def test_delete_key(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1)
    cache.delete("a")
    assert cache.get("a") == -1


@patch('__main__.time.monotonic')
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("a", 1, ttl=10)
    mock_time.return_value = 109.0
    assert cache.get("a") == 1
    mock_time.return_value = 111.0
    assert cache.get("a") == -1