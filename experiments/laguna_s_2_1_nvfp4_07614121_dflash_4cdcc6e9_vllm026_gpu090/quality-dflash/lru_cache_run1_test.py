import time
from typing import Optional, Dict, Any


class Node:
    """Doubly linked list node for LRU tracking."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support using doubly-linked list and hash map.

    All operations (get, put, delete) have O(1) average time complexity.
    Uses lazy cleanup - expired items are only removed when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize TTLCache.

        Args:
            capacity: Maximum number of items in cache.
            default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node(None, None, 0)  # Dummy head
        self.tail = Node(None, None, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front of the linked list (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front of the linked list."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Check if a node is expired."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns None if key doesn't exist or is expired.

        Args:
            key: The key to look up.

        Returns:
            Value associated with key, or None if not found/expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            del self.cache[key]
            self._remove_node(node)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put key-value pair into cache with optional TTL override.

        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional time-to-live override. Uses default_ttl if None.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = current_time + effective_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used (tail.prev)
                lru_node = self.tail.prev
                del self.cache[lru_node.key]
                self._remove_node(lru_node)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if key was deleted, False if not found.
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        del self.cache[key]
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Get current number of items in cache.

        Returns:
            Number of items currently in cache.
        """
        return len(self.cache)

import pytest
from unittest.mock import patch
import time

# Assuming the TTLCache class is in ttl_cache.py
#

class TestTTLCache:
    @patch('time.monotonic')
    def test_get_existing_key(self, mock_time):
        """Test getting an existing key returns correct value."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')

        result = cache.get('key1')
        assert result == 'value1'

    @patch('time.monotonic')
    def test_get_nonexistent_key(self, mock_time):
        """Test getting a non-existent key returns None."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)

        result = cache.get('nonexistent')
        assert result is None

    @patch('time.monotonic')
    def test_put_and_get(self, mock_time):
        """Test putting and getting a value."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')

        assert cache.get('key1') == 'value1'
        assert cache.size() == 1

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        """Test that LRU eviction works correctly."""
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
        """Test that TTL expiration works correctly."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 5)
        cache.put('key1', 'value1')

        # Move time forward beyond TTL
        mock_time.return_value = 106.0
        assert cache.get('key1') is None

    @patch('time.monotonic')
    def test_delete_key(self, mock_time):
        """Test deleting a key from cache."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('key1', 'value1')

        result = cache.delete('key1')
        assert result is True
        assert cache.get('key1') is None
        assert cache.size() == 0