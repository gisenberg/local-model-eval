import time
from typing import Optional, Dict, Any

class Node:
    """Doubly linked list node for LRU cache."""
    def __init__(self, key: Any = None, value: Any = None, expiry: float = 0):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    """
    LRU Cache with TTL support using doubly-linked list and hash map.

    Args:
        capacity: Maximum number of items in cache
        default_ttl: Default time-to-live in seconds
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("TTL must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node()  # Dummy head
        self.tail = Node()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move existing node to front (most recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def _evict_lru(self) -> None:
        """Remove least recently used item."""
        if self.head.next != self.tail:
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self.cache[lru_node.key]

    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns None if not found or expired.

        Args:
            key: Key to lookup

        Returns:
            Value if found and not expired, None otherwise
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy expiration check
        if self._is_expired(node):
            self._remove(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put key-value pair into cache.

        Args:
            key: Key to insert
            value: Value to insert
            ttl: Time-to-live in seconds. If None, uses default_ttl
        """
        if ttl is None:
            ttl = self.default_ttl

        expiry = time.monotonic() + ttl if ttl > 0 else float('inf')

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            # Add new node
            if len(self.cache) >= self.capacity:
                self._evict_lru()

            node = Node(key, value, expiry)
            self.cache[key] = node
            self._add_to_front(node)

    def delete(self, key: Any) -> bool:
        """
        Delete key from cache.

        Args:
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of items in cache
        """
        return len(self.cache)

# Tests
import pytest
from unittest.mock import patch

class TestTTLCache:

    @patch('time.monotonic')
    def test_get_returns_value(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')
        assert cache.get('key1') == 'value1'

    @patch('time.monotonic')
    def test_get_returns_none_for_missing_key(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        assert cache.get('missing') is None

    @patch('time.monotonic')
    def test_put_updates_existing_key(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')
        cache.put('key1', 'value2')
        assert cache.get('key1') == 'value2'

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # Should evict key1
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1', ttl=5)

        mock_time.return_value = 106.0  # After expiration
        assert cache.get('key1') is None

    @patch('time.monotonic')
    def test_delete_key(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None
        assert cache.delete('missing') is False