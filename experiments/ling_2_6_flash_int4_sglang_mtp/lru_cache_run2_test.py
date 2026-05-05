import time
from typing import Any, Optional


class Node:
    def __init__(self, key: Any, value: Any, ttl_expiry: float):
        self.key: Any = key
        self.value: Any = value
        self.ttl_expiry: float = ttl_expiry
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None


class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize a TTL-based LRU Cache.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default Time-To-Live in seconds for each item.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: dict[Any, Node] = {}

        # Doubly linked list sentinel nodes
        self.head = Node("", None, float('inf'))
        self.tail = Node("", None, float('inf'))
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove node from its current position."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_front(self, node: Node) -> None:
        """Move existing node to front."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Check if node has expired based on current time."""
        return time.monotonic() >= node.ttl_expiry

    def _cleanup_expired(self) -> None:
        """Lazy cleanup: remove expired nodes from the end."""
        current = self.tail.prev
        while current != self.head and self._is_expired(current):
            prev_node = current.prev
            self._remove_node(current)
            del self.cache[current.key]
            current = prev_node

    def get(self, key: Any) -> Any:
        """
        Retrieve value for the given key if it exists and is not expired.

        Args:
            key (Any): Key to look up.

        Returns:
            Any: Value if key exists and is valid, else None.
        """
        self._cleanup_expired()

        if key in self.cache:
            node = self.cache[key]
            if not self._is_expired(node):
                self._move_to_front(node)
                return node.value
            else:
                # Remove expired node
                self._remove_node(node)
                del self.cache[key]
        return None

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional TTL.

        Args:
            key (Any): Key to insert or update.
            value (Any): Value to store.
            ttl (Optional[float]): Time-To-Live in seconds. Uses default_ttl if None.
        """
        self._cleanup_expired()

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl_expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict least recently used (node before tail)
                lru_node = self.tail.prev
                self._remove_node(lru_node)
                del self.cache[lru_node.key]

            ttl_expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            new_node = Node(key, value, ttl_expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.

        Args:
            key (Any): Key to delete.

        Returns:
            bool: True if key was deleted, False otherwise.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the current number of valid items in the cache.

        Returns:
            int: Number of items.
        """
        self._cleanup_expired()
        return len(self.cache)

import time
import unittest
from unittest.mock import patch


class TestTTLCache(unittest.TestCase):
    def test_put_and_get(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            self.assertEqual(cache.get('a'), 1)

    def test_get_expired_key(self):
        with patch('time.monotonic', side_effect=[100.0, 111.0]):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            self.assertIsNone(cache.get('a'))  # expired

    def test_update_key(self):
        with patch('time.monotonic', side_effect=[100.0, 105.0, 106.0]):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            cache.put('a', 2)  # update
            self.assertEqual(cache.get('a'), 2)

    def test_delete_key(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            self.assertTrue(cache.delete('a'))
            self.assertIsNone(cache.get('a'))

    def test_cache_eviction(self):
        with patch('time.monotonic', return_value=100.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            cache.put('b', 2)
            cache.put('c', 3)  # should evict 'a'
            self.assertIsNone(cache.get('a'))
            self.assertEqual(cache.get('b'), 2)
            self.assertEqual(cache.get('c'), 3)

    def test_size(self):
        with patch('time.monotonic', side_effect=[100.0, 111.0]):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            cache.put('b', 2)
            self.assertEqual(cache.size(), 2)
            cache.get('a')  # access to trigger cleanup
            self.assertEqual(cache.size(), 1)  # 'b' is expired


if __name__ == '__main__':
    unittest.main()