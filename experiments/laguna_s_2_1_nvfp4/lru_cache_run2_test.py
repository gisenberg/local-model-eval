import time
from typing import Any, Optional

class Node:
    """Doubly linked list node for LRU cache."""
    def __init__(self, key: Any = None, value: Any = None, expire_at: float = 0):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with TTL support using doubly-linked list and hash map.

    Time complexity: O(1) average for all operations
    Space complexity: O(capacity)
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize TTLCache.

        Args:
            capacity: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # key -> Node
        self.head = Node()  # dummy head
        self.tail = Node()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove node from doubly linked list."""
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move existing node to front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _remove_from_back(self) -> Node:
        """Remove and return least recently used node."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired."""
        return time.monotonic() >= node.expire_at

    def get(self, key: Any) -> Any:
        """
        Get value by key. Updates LRU position if found and not expired.

        Args:
            key: Cache key

        Returns:
            Value if found and not expired, None otherwise
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
            key: Cache key
            value: Value to store
            ttl: Optional TTL override (uses default_ttl if None)
        """
        current_time = time.monotonic()
        expire_at = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                lru_node = self._remove_from_back()
                del self.cache[lru_node.key]

            new_node = Node(key, value, expire_at)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if key not found
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Get current cache size (excludes expired items).

        Returns:
            Number of valid items in cache
        """
        # Lazy cleanup of expired items
        current_time = time.monotonic()
        expired_keys = [
            key for key, node in self.cache.items()
            if current_time >= node.expire_at
        ]

        for key in expired_keys:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

        return len(self.cache)

# Tests
import pytest
from unittest.mock import patch

class TestTTLCache:
    @patch('time.monotonic')
    def test_get_returns_value(self, mock_time):
        """Test basic get operation returns correct value."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    @patch('time.monotonic')
    def test_get_returns_none_for_missing_key(self, mock_time):
        """Test get returns None for missing key."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        assert cache.get("missing") is None

    @patch('time.monotonic')
    def test_put_evicts_lru_when_full(self, mock_time):
        """Test LRU eviction when cache is full."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Make key1 recently used
        cache.put("key3", "value3")  # Should evict key2
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        """Test item expiration based on TTL."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 5)
        cache.put("key1", "value1")
        mock_time.return_value = 106.0  # Past TTL
        assert cache.get("key1") is None

    @patch('time.monotonic')
    def test_delete_removes_item(self, mock_time):
        """Test delete operation removes item."""
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False

    @patch('time.monotonic')
    def test_size_counts_valid_items(self, mock_time):
        """Test size() counts only non-expired items."""
        mock_time.return_value = 100.0
        cache = TTLCache(3, 10)
        cache.put("key1", "value1")
        cache.put("key2", "value2", ttl=5)
        mock_time.return_value = 106.0  # key2 expired
        assert cache.size() == 1