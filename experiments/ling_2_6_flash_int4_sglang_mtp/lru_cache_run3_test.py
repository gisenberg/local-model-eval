import time
from typing import Any, Optional


class Node:
    """Doubly linked list node."""
    def __init__(self, key: Any, value: Any, ttl: float):
        self.key: Any = key
        self.value: Any = value
        self.expires_at: float = ttl  # absolute expiration time
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None


class TTLCache:
    """
    A thread-unsafe LRU Cache with TTL (Time-To-Live) support.

    Features:
    - O(1) average time complexity for get and put operations.
    - Lazy expiration: expired items are removed only when accessed.
    - Doubly-linked list + hash map for efficient operations.

    Args:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for items.
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: dict[Any, Node] = {}
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front of the doubly linked list."""
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node

    def _is_expired(self, node: Node, current_time: float) -> bool:
        """Check if a node has expired based on current time."""
        return current_time >= node.expires_at

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front."""
        self._remove_node(node)
        self._add_to_front(node)

    def get(self, key: Any, current_time: Optional[float] = None) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key: Key to look up.
            current_time: Optional time for testing purposes.

        Returns:
            The value if the key exists and is not expired, else None.
        """
        if current_time is None:
            current_time = time.monotonic()

        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node, current_time):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None, current_time: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: Key to insert or update.
            value: Value to store.
            ttl: Optional custom TTL in seconds. Uses default_ttl if not provided.
            current_time: Optional time for testing purposes.
        """
        if current_time is None:
            current_time = time.monotonic()

        if ttl is None:
            ttl = self.default_ttl

        expires_at = current_time + ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                self._evict_lru(current_time)

            new_node = Node(key, value, expires_at)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def _evict_lru(self, current_time: float) -> None:
        """Evict the least recently used (and not expired) item."""
        current = self.tail
        while current:
            if self._is_expired(current, current_time):
                prev_node = current.prev
                self._remove_node(current)
                del self.cache[current.key]
                return
            current = current.prev

        # If all items are expired, evict the LRU (tail)
        if self.tail:
            del self.cache[self.tail.key]
            self._remove_node(self.tail)

    def delete(self, key: Any) -> None:
        """
        Delete a key from the cache.

        Args:
            key: Key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            int: Number of items currently stored.
        """
        return len(self.cache)

import time
import unittest
from unittest.mock import patch


class TestTTLCache(unittest.TestCase):
    def test_put_and_get(self):
        with patch('time.monotonic', return_value=1000):
            cache = TTLCache(2, 10)
            cache.put('a', 1)
            self.assertEqual(cache.get('a'), 1)

    def test_get_expired_key(self):
        with patch('time.monotonic', side_effect=[1000, 1011]):
            cache = TTLCache(2, 10)
            cache.put('a', 1)
            self.assertIsNone(cache.get('a'))

    def test_update_existing_key(self):
        with patch('time.monotonic', return_value=1000):
            cache = TTLCache(2, 10)
            cache.put('a', 1)
            cache.put('a', 2)
            self.assertEqual(cache.get('a'), 2)

    def test_lru_eviction(self):
        with patch('time.monotonic', return_value=1000):
            cache = TTLCache(2, 10)
            cache.put('a', 1)
            cache.put('b', 2)
            cache.put('c', 3)  # Should evict 'a'
            self.assertIsNone(cache.get('a'))
            self.assertEqual(cache.get('b'), 2)
            self.assertEqual(cache.get('c'), 3)

    def test_delete_key(self):
        with patch('time.monotonic', return_value=1000):
            cache = TTLCache(2, 10)
            cache.put('a', 1)
            cache.delete('a')
            self.assertIsNone(cache.get('a'))

    def test_size(self):
        with patch('time.monotonic', return_value=1000):
            cache = TTLCache(2, 10)
            self.assertEqual(cache.size(), 0)
            cache.put('a', 1)
            self.assertEqual(cache.size(), 1)
            cache.put('b', 2)
            self.assertEqual(cache.size(), 2)
            cache.put('c', 3)
            self.assertEqual(cache.size(), 2)


if __name__ == '__main__':
    unittest.main()