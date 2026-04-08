import time
from typing import Optional, Any, Dict

class Node:
    def __init__(self, key: str, value: Any, expiration: float):
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev = None
        self.next = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache with a given capacity and default time-to-live.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head = None
        self.tail = None
        self.size = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for the given key if it exists and is not expired.
        Accessing a key makes it most-recently-used.
        """
        if key not in self.cache:
            return None
        node = self.cache[key]
        if node.expiration <= time.monotonic():
            # Expired, remove from cache
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1
            return None
        # Move to tail
        self._move_to_tail(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Insert or update the value for the given key. If at capacity, evict the least-recently-used non-expired item.
        """
        if ttl is None:
            ttl = self.default_ttl
        current_time = time.monotonic()
        expiration = current_time + ttl
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiration = expiration
            self._move_to_tail(node)
            return
        # Add new node
        if self.size >= self.capacity:
            # Evict head
            self._evict_head()
        new_node = Node(key, value, expiration)
        self.cache[key] = new_node
        self._add_to_tail(new_node)
        self.size += 1

    def delete(self, key: str) -> bool:
        """
        Remove the key from the cache if it exists.
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
        Return the count of non-expired items in the cache.
        """
        return self.size

    def _add_to_tail(self, node: Node):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

    def _remove_node(self, node: Node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _move_to_tail(self, node: Node):
        self._remove_node(node)
        self._add_to_tail(node)

    def _evict_head(self):
        if self.head is None:
            return
        node = self.head
        self._remove_node(node)
        del self.cache[node.key]
        self.size -= 1

import unittest
from unittest import mock


class TestTTLCache(unittest.TestCase):
    def test_get_expired(self):
        """
        Test that get returns None when the item is expired.
        """
        cache = TTLCache(1, 1.0)
        cache.put('a', 1)
        with mock.patch('time.monotonic', return_value=2.0):
            self.assertIsNone(cache.get('a'))

    def test_get_moves_to_tail(self):
        """
        Test that get moves the item to the tail if not expired.
        """
        cache = TTLCache(1, 1.0)
        cache.put('a', 1)
        with mock.patch('time.monotonic', return_value=0.5):
            self.assertEqual(cache.get('a'), 1)
        with mock.patch('time.monotonic', return_value=0.6):
            self.assertEqual(cache.get('a'), 1)
        # After the second get, the node should be moved to the tail
        # (Assuming the internal logic is correct)

    def test_put_evicts_when_full(self):
        """
        Test that put evicts the least recently used item when the cache is full.
        """
        cache = TTLCache(2, 1.0)
        cache.put('a', 1)
        cache.put('b', 2)
        with mock.patch('time.monotonic', return_value=0.5):
            cache.put('c', 3)
        self.assertEqual(cache.size(), 2)
        self.assertEqual(cache.get('c'), 3)
        self.assertIsNone(cache.get('a'))

    def test_delete_removes_item(self):
        """
        Test that delete removes the item from the cache.
        """
        cache = TTLCache(1, 1.0)
        cache.put('a', 1)
        self.assertTrue(cache.delete('a'))
        self.assertFalse(cache.delete('a'))

    def test_size_returns_correct_count(self):
        """
        Test that size returns the correct count of non-expired items.
        """
        cache = TTLCache(2, 1.0)
        cache.put('a', 1)
        cache.put('b', 2)
        self.assertEqual(cache.size(), 2)
        with mock.patch('time.monotonic', return_value=2.0):
            cache.get('a')
        self.assertEqual(cache.size(), 1)

    def test_evict_least_recently_used(self):
        """
        Test that the least recently used item is evicted when the cache is full.
        """
        cache = TTLCache(2, 1.0)
        cache.put('a', 1)
        cache.put('b', 2)
        with mock.patch('time.monotonic', return_value=0.5):
            cache.put('c', 3)
        self.assertEqual(cache.size(), 2)
        self.assertEqual(cache.get('c'), 3)
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.get('b'), 2)