import time
from typing import Optional, Any, Dict
from unittest.mock import patch

class Node:
    def __init__(self):
        self.key = None
        self.value = None
        self.expiration_time = 0.0
        self.prev = None
        self.next = None


class TTLCache:
    """LRU cache with time-based expiration."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache with given capacity and default TTL.
        
        Args:
            capacity (int): Maximum number of items in the cache.
            default_ttl (float): Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Maps keys to Node objects
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size_counter = 0  # Tracks number of non-expired items

    def _move_to_head(self, node):
        """Move a node to the head of the linked list (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _add_to_head(self, node):
        """Add a node to the head of the linked list."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _evict(self):
        """Evict the least recently used item (tail.prev)."""
        node = self.tail.prev
        current_time = time.monotonic()
        if node.expiration_time > current_time:
            self.size_counter -= 1
        self._remove_node(node)
        del self.cache[node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for the given key if it exists and is not expired.
        
        Args:
            key (str): Key to retrieve.
        
        Returns:
            Optional[Any]: Value if key exists and is not expired, else None.
        """
        current_time = time.monotonic()
        node = self.cache.get(key)
        if not node:
            return None
        if node.expiration_time <= current_time:
            self._remove_node(node)
            del self.cache[key]
            self.size_counter -= 1
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Insert or update a key-value pair in the cache.
        
        Args:
            key (str): Key to insert/update.
            value (Any): Value to store.
            ttl (Optional[float]): Time-to-live in seconds. If None, use default_ttl.
        """
        current_time = time.monotonic()
        if key in self.cache:
            node = self.cache[key]
            if node.expiration_time <= current_time:
                self._remove_node(node)
                del self.cache[key]
                self.size_counter -= 1
            else:
                self._move_to_head(node)
            node.value = value
            if ttl is None:
                node.expiration_time = current_time + self.default_ttl
            else:
                node.expiration_time = current_time + ttl
            self._move_to_head(node)
        else:
            if self.capacity == 0:
                return
            if self.size_counter >= self.capacity:
                self._evict()
            new_node = Node()
            new_node.key = key
            new_node.value = value
            if ttl is None:
                new_node.expiration_time = current_time + self.default_ttl
            else:
                new_node.expiration_time = current_time + ttl
            self.cache[key] = new_node
            self._add_to_head(new_node)
            self.size_counter += 1

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key (str): Key to delete.
        
        Returns:
            bool: True if the key existed and was removed, else False.
        """
        if key not in self.cache:
            return False
        node = self.cache[key]
        current_time = time.monotonic()
        if node.expiration_time > current_time:
            self.size_counter -= 1
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.
        
        Returns:
            int: Count of non-expired items.
        """
        return self.size_counter

import unittest
from unittest.mock import patch
  # Replace with the actual module name


class TestTTLCache(unittest.TestCase):
    def test_get_existing_not_expired(self):
        """Test get returns value when not expired."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(1, 1.0)
            cache.put('key', 1)
            self.assertEqual(cache.get('key'), 1)
            self.assertEqual(cache.size(), 1)

    def test_get_expired(self):
        """Test get returns None when expired."""
        with patch('time.monotonic', return_value=1.0):
            cache = TTLCache(1, 1.0)
            cache.put('key', 1)
            self.assertIsNone(cache.get('key'))
            self.assertEqual(cache.size(), 0)

    def test_put_update(self):
        """Test put updates value and moves to head."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(1, 1.0)
            cache.put('key', 1)
            cache.put('key', 2)
            self.assertEqual(cache.get('key'), 2)
            self.assertEqual(cache.size(), 1)

    def test_eviction(self):
        """Test eviction when cache is full."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(1, 1.0)
            cache.put('key1', 1)
            cache.put('key2', 2)
            self.assertEqual(cache.size(), 1)
            self.assertIsNone(cache.get('key1'))
            self.assertEqual(cache.size(), 1)

    def test_delete(self):
        """Test delete removes key from cache."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(1, 1.0)
            cache.put('key', 1)
            cache.delete('key')
            self.assertEqual(cache.size(), 0)
            self.assertIsNone(cache.get('key'))

    def test_size(self):
        """Test size returns correct count of non-expired items."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(2, 1.0)
            cache.put('key1', 1)
            cache.put('key2', 2)
            self.assertEqual(cache.size(), 2)
            cache.get('key1')
            cache.put('key3', 3)
            self.assertEqual(cache.size(), 2)