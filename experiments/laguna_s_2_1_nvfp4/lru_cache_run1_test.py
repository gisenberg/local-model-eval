import time
from typing import Any, Optional, Dict, Tuple

class Node:
    """Doubly linked list node for LRU cache"""
    def __init__(self, key: Any = None, value: Any = None,
                 expire_time: float = 0.0):
        self.key = key
        self.value = value
        self.expire_time = expire_time
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with TTL support using doubly-linked list and hash map.

    Time Complexity: O(1) average for get, put, delete operations.
    Uses lazy cleanup for expired entries.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize TTLCache.

        Args:
            capacity: Maximum number of items in cache.
            default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node()  # Dummy head
        self.tail = Node()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
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
        self._remove_node(node)
        self._add_to_front(node)

    def _remove_from_back(self) -> None:
        """Remove least recently used node."""
        if self.head.next != self.tail:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

    def _cleanup_expired(self) -> None:
        """Lazy cleanup of expired entries from the back."""
        current_time = time.monotonic()
        while self.tail.prev != self.head:
            node = self.tail.prev
            if node.expire_time <= current_time:
                self._remove_node(node)
                del self.cache[node.key]
            else:
                break

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns -1 if not found or expired.

        Args:
            key: Key to lookup.

        Returns:
            Value if found and not expired, -1 otherwise.
        """
        if key not in self.cache:
            return -1

        node = self.cache[key]
        current_time = time.monotonic()

        # Check expiration
        if node.expire_time <= current_time:
            self._remove_node(node)
            del self.cache[key]
            return -1

        # Move to front (most recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put key-value pair into cache with optional TTL.

        Args:
            key: Key to insert/update.
            value: Value to store.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_time = current_time + effective_ttl

        # Clean up expired entries first
        self._cleanup_expired()

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expire_time = expire_time
            self._move_to_front(node)
        else:
            # Add new node
            if len(self.cache) >= self.capacity:
                self._remove_from_back()

            new_node = Node(key, value, expire_time)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete key from cache.

        Args:
            key: Key to delete.

        Returns:
            True if key was deleted, False if not found.
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Get current size of cache.

        Returns:
            Number of items currently in cache.
        """
        self._cleanup_expired()
        return len(self.cache)

# test_ttl_cache.py
import pytest
from unittest.mock import patch

class TestTTLCache:

    @patch('ttl_cache.time.monotonic')
    def test_get_returns_value_when_not_expired(self, mock_time):
        """Test that get returns correct value when entry is not expired."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")

        mock_time.return_value = 105.0
        result = cache.get("key1")
        assert result == "value1"

    @patch('ttl_cache.time.monotonic')
    def test_get_returns_minus_one_when_expired(self, mock_time):
        """Test that get returns -1 when entry is expired."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")

        mock_time.return_value = 115.0  # 15 seconds later, expired
        result = cache.get("key1")
        assert result == -1

    @patch('ttl_cache.time.monotonic')
    def test_put_with_custom_ttl(self, mock_time):
        """Test that put uses custom TTL when provided."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1", ttl=5.0)  # Custom TTL of 5 seconds

        mock_time.return_value = 104.0  # 4 seconds later, still valid
        assert cache.get("key1") == "value1"

        mock_time.return_value = 106.0  # 6 seconds later, expired
        assert cache.get("key1") == -1

    @patch('ttl_cache.time.monotonic')
    def test_lru_eviction_order(self, mock_time):
        """Test that LRU eviction removes least recently used item."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add third item, should evict key2 (LRU)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == -1
        assert cache.get("key3") == "value3"

    @patch('ttl_cache.time.monotonic')
    def test_delete_removes_key(self, mock_time):
        """Test that delete removes key from cache."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")

        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") == -1

        # Deleting non-existent key
        result = cache.delete("key1")
        assert result is False

    @patch('ttl_cache.time.monotonic')
    def test_size_reflects_expired_entries(self, mock_time):
        """Test that size() does not count expired entries."""
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.size() == 2

        # Advance time to expire one entry
        mock_time.return_value = 115.0
        assert cache.size() == 1  # key1 expired, key2 still valid