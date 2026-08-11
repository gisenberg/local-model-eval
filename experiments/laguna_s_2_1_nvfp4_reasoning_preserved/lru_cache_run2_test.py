import time
from typing import Optional, Any, Dict


class Node:
    """Doubly linked list node."""
    def __init__(self, key: Any = None, value: Any = None, expiry: float = 0):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support using doubly-linked list and hash map.

    Methods:
        get(key): Get value by key, returns None if not found or expired.
        put(key, value, ttl=None): Insert/update key-value pair with optional TTL.
        delete(key): Remove key from cache.
        size(): Return current number of items in cache.
    """

    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node()  # dummy head
        self.tail = Node()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
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

    def _cleanup_expired(self) -> None:
        """Lazy cleanup of expired nodes from the back."""
        current_time = time.monotonic()
        while self.tail.prev != self.head and self.tail.prev.expiry <= current_time:
            expired_node = self.tail.prev
            self._remove(expired_node)
            del self.cache[expired_node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Get value by key.

        Args:
            key: Key to lookup.

        Returns:
            Value if found and not expired, None otherwise.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        current_time = time.monotonic()

        if node.expiry <= current_time:
            # Expired
            self._remove(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.

        Args:
            key: Key to insert/update.
            value: Value to store.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl

        current_time = time.monotonic()
        expiry = current_time + ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU (least recently used is at tail.prev)
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

            new_node = Node(key, value, expiry)
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
        self._remove(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Return current number of items in cache.

        Returns:
            Number of items currently stored.
        """
        self._cleanup_expired()
        return len(self.cache)

import pytest
from unittest.mock import patch
import time


class TestTTLCache:
    @patch('time.monotonic')
    def test_get_and_put(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('a', 1)
        assert cache.get('a') == 1
        mock_time.return_value = 105.0
        assert cache.get('a') == 1

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')  # 'a' becomes MRU
        cache.put('c', 3)  # should evict 'b'
        assert cache.get('b') is None
        assert cache.get('a') == 1
        assert cache.get('c') == 3

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 5)
        cache.put('a', 1)
        mock_time.return_value = 104.0
        assert cache.get('a') == 1
        mock_time.return_value = 106.0
        assert cache.get('a') is None

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('a', 1, ttl=3)
        mock_time.return_value = 102.0
        assert cache.get('a') == 1
        mock_time.return_value = 104.0
        assert cache.get('a') is None

    @patch('time.monotonic')
    def test_delete(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put('a', 1)
        assert cache.delete('a') is True
        assert cache.get('a') is None
        assert cache.delete('a') is False

    @patch('time.monotonic')
    def test_size_with_expired_items(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(3, 5)
        cache.put('a', 1)
        cache.put('b', 2)
        mock_time.return_value = 106.0
        assert cache.size() == 0
        cache.put('c', 3)
        assert cache.size() == 1