import time
from typing import Optional, Dict, Any
from collections import defaultdict

class Node:
    def __init__(self, key: Any = None, value: Any = None, expire_at: float = 0):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: 'Node' = None
        self.next: 'Node' = None

class TTLCache:
    """
    LRU Cache with TTL (Time To Live) support.

    Args:
        capacity (int): Maximum number of items in cache.
        default_ttl (float): Default time-to-live in seconds.
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}

        # Dummy head and tail nodes
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from doubly linked list."""
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add node right after dummy head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move existing node to front (most recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def _pop_tail(self) -> Node:
        """Remove and return the least recently used node."""
        node = self.tail.prev
        self._remove(node)
        return node

    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired."""
        return node.expire_at > 0 and time.monotonic() >= node.expire_at

    def _lazy_cleanup(self) -> None:
        """Remove expired nodes from cache."""
        current = self.head.next
        while current != self.tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove(current)
                del self.cache[current.key]
            current = next_node

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns -1 if not found or expired.

        Args:
            key: Key to lookup.

        Returns:
            Value if found and not expired, else -1.
        """
        self._lazy_cleanup()
        if key not in self.cache:
            return -1

        node = self.cache[key]
        if self._is_expired(node):
            self._remove(node)
            del self.cache[key]
            return -1

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put key-value pair into cache.

        Args:
            key: Key to insert/update.
            value: Value to store.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        self._lazy_cleanup()

        if ttl is None:
            ttl = self.default_ttl

        expire_at = time.monotonic() + ttl if ttl > 0 else 0

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]

            new_node = Node(key, value, expire_at)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """
        Delete key from cache.

        Args:
            key: Key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Return current number of items in cache.

        Returns:
            Number of items.
        """
        self._lazy_cleanup()
        return len(self.cache)

import pytest
from unittest.mock import patch
import time

class TestTTLCache:

    @patch('your_module.time.monotonic')
    def test_get_returns_value(self, mock_monotonic):
        """Test basic get functionality."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1)
        assert cache.get('a') == 1

    @patch('your_module.time.monotonic')
    def test_get_returns_default_for_missing_key(self, mock_monotonic):
        """Test get returns -1 for missing key."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        assert cache.get('missing') == -1

    @patch('your_module.time.monotonic')
    def test_put_over_capacity_evicts_lru(self, mock_monotonic):
        """Test LRU eviction when capacity exceeded."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        assert cache.get('a') == -1
        assert cache.get('b') == 2
        assert cache.get('c') == 3

    @patch('your_module.time.monotonic')
    def test_expired_key_returns_default(self, mock_monotonic):
        """Test expired keys are treated as missing."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1)

        # Move time forward beyond TTL
        mock_monotonic.return_value = 115.0
        assert cache.get('a') == -1

    @patch('your_module.time.monotonic')
    def test_custom_ttl_overrides_default(self, mock_monotonic):
        """Test custom TTL overrides default."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1, ttl=5)

        # Within custom TTL
        mock_monotonic.return_value = 104.0
        assert cache.get('a') == 1

        # Beyond custom TTL but within default
        mock_monotonic.return_value = 106.0
        assert cache.get('a') == -1

    @patch('your_module.time.monotonic')
    def test_delete_removes_key(self, mock_monotonic):
        """Test delete functionality."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') == -1
        assert cache.size() == 0