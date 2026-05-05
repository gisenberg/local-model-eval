import time
from typing import Any, Optional


class Node:
    """Doubly linked list node."""
    def __init__(self, key: Any, value: Any, expires_at: float):
        self.key: Any = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None


class TTLCache:
    """
    A thread-unsafe LRU Cache with TTL (Time-To-Live) support.

    Features:
    - O(1) average time complexity for get/put/delete
    - Lazy expiration of expired entries
    - Doubly linked list + hash map (no OrderedDict)

    Args:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for entries.
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

    def _is_expired(self, node: Node, current_time: float) -> bool:
        return current_time >= node.expires_at

    def _move_to_head(self, node: Node) -> None:
        if node == self.head:
            return
        if node == self.tail:
            self.tail = node.prev
            if self.tail:
                self.tail.next = None
        else:
            if node.prev:
                node.prev.next = node.next
            if node.next:
                node.next.prev = node.prev

        node.prev = None
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node

        if not self.tail:
            self.tail = node

    def _add_to_head(self, node: Node) -> None:
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node

    def _remove_node(self, node: Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _remove_from_cache(self, key: Any) -> None:
        node = self.cache.pop(key)
        self._remove_node(node)

    def get(self, key: Any, current_time: Optional[float] = None) -> Optional[Any]:
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
            self._remove_from_cache(key)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None, current_time: Optional[float] = None) -> None:
        """
        Insert or update a value in the cache.

        Args:
            key: Key to insert or update.
            value: Value to store.
            ttl: Optional custom TTL in seconds. Uses default_ttl if not provided.
            current_time: Optional time for testing purposes.
        """
        if current_time is None:
            current_time = time.monotonic()

        expires_at = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the LRU (tail)
                lru_node = self.tail
                if lru_node:
                    self._remove_from_cache(lru_node.key)

            new_node = Node(key, value, expires_at)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        if key not in self.cache:
            return False
        self._remove_from_cache(key)
        return True

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            Number of items currently in the cache.
        """
        return len(self.cache)

import unittest
from unittest.mock import patch


class TestTTLCache(unittest.TestCase):

    def test_put_and_get(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(2, 10.0)
            cache.put('a', 1)
            self.assertEqual(cache.get('a'), 1)

    def test_get_expired(self):
        with patch('time.monotonic', side_effect=[100.0, 111.0]):
            cache = TTLCache(2, 10.0)
            cache.put('a', 1)
            self.assertIsNone(cache.get('a'))

    def test_put_overwrites(self):
        with patch('time.monotonic', side_effect=[100.0, 100.0]):
            cache = TTLCache(2, 10.0)
            cache.put('a', 1)
            cache.put('a', 2)
            self.assertEqual(cache.get('a'), 2)

    def test_lru_eviction(self):
        with patch('time.monotonic', side_effect=[100.0, 100.0, 100.0, 100.0]):
            cache = TTLCache(2, 100.0)
            cache.put('a', 1)
            cache.put('b', 2)
            cache.put('c', 3)  # Should evict 'a'
            self.assertIsNone(cache.get('a'))
            self.assertEqual(cache.get('b'), 2)
            self.assertEqual(cache.get('c'), 3)

    def test_delete(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(2, 10.0)
            cache.put('a', 1)
            self.assertTrue(cache.delete('a'))
            self.assertFalse(cache.delete('a'))
            self.assertIsNone(cache.get('a'))

    def test_size(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(2, 10.0)
            self.assertEqual(cache.size(), 0)
            cache.put('a', 1)
            self.assertEqual(cache.size(), 1)
            cache.put('b', 2)
            self.assertEqual(cache.size(), 2)
            cache.put('c', 3)  # evict 'a'
            self.assertEqual(cache.size(), 2)


if __name__ == '__main__':
    unittest.main()