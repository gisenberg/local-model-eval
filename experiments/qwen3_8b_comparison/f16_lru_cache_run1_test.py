from typing import Optional, Any, Dict, List
import time

class Node:
    def __init__(self, key: Any, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL Cache with a given capacity and default TTL.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size = 0

    def _add_to_head(self, node: Node):
        """Add a node to the head of the doubly linked list."""
        if self.head:
            self.head.prev = node
        node.next = self.head
        self.head = node
        if not self.tail:
            self.tail = node

    def _remove_node(self, node: Node):
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _move_to_head(self, node: Node):
        """Move a node to the head of the list (marking it as recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve the value for a key from the cache.

        Args:
            key (Any): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and hasn't expired, else None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if time.monotonic() >= node.ttl:
            self.delete(key)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None):
        """
        Add or update a key-value pair in the cache.

        Args:
            key (Any): The key to store.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. Defaults to default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl = time.monotonic() + ttl
            self._move_to_head(node)
            return

        if self.size >= self.capacity:
            # Remove the least recently used item (tail)
            if self.tail:
                self.delete(self.tail.key)

        node = Node(key, value, time.monotonic() + ttl)
        self._add_to_head(node)
        self.cache[key] = node
        self.size += 1

    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.

        Args:
            key (Any): The key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size -= 1
        return True

    def size(self) -> int:
        """
        Get the current number of items in the cache.

        Returns:
            int: The number of items in the cache.
        """
        return self.size

import unittest
from unittest.mock import patch
from datetime import timedelta
from typing import Optional, Any

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(3, 1.0)

    @patch('time.monotonic', return_value=0.0)
    def test_get(self, mock_monotonic):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

    @patch('time.monotonic', return_value=1.5)
    def test_get_expired(self, mock_monotonic):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), None)

    @patch('time.monotonic', return_value=0.0)
    def test_put(self, mock_monotonic):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.size(), 1)

    @patch('time.monotonic', return_value=0.0)
    def test_put_ttl(self, mock_monotonic):
        self.cache.put("key1", "value1", ttl=2.0)
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.size(), 1)

    @patch('time.monotonic', return_value=0.0)
    def test_delete(self, mock_monotonic):
        self.cache.put("key1", "value1")
        self.assertTrue(self.cache.delete("key1"))
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic', return_value=0.0)
    def test_size(self, mock_monotonic):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.cache.delete("key1")
        self.assertEqual(self.cache.size(), 0)

if __name__ == "__main__":
    unittest.main()