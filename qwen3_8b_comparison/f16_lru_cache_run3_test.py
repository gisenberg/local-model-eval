from typing import Optional, Any, Dict
import time
from collections import deque

class TTLCache:
    """
    A thread-safe LRU cache with Time-to-Live (TTL) support.

    Attributes:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for items without explicit TTL.
        cache (Dict): Hash map mapping keys to (value, expiration_time) tuples.
        lru_queue (deque): Doubly-linked list to maintain LRU order.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache with given capacity and default TTL.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items without explicit TTL.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, (Any, float)] = {}
        self.lru_queue: deque = deque()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the given key from the cache.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value associated with the key, or None if the key is not found or expired.
        """
        if key not in self.cache:
            return None

        value, expiration_time = self.cache[key]
        if time.monotonic() >= expiration_time:
            self.delete(key)
            return None

        # Move the accessed key to the end of the queue (LRU)
        self.lru_queue.remove(key)
        self.lru_queue.append(key)
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update an item in the cache with an optional TTL.

        Args:
            key (str): The key to store.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. If None, use default TTL.
        """
        if key in self.cache:
            self.lru_queue.remove(key)
        else:
            if len(self.cache) >= self.capacity:
                self.delete(self.lru_queue.popleft())

        expiration_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self.cache[key] = (value, expiration_time)
        self.lru_queue.append(key)

    def delete(self, key: str) -> bool:
        """
        Delete the item associated with the given key from the cache.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was found and deleted, False otherwise.
        """
        if key not in self.cache:
            return False

        del self.cache[key]
        if key in self.lru_queue:
            self.lru_queue.remove(key)
        return True

    def size(self) -> int:
        """
        Return the number of items currently in the cache.

        Returns:
            int: The number of items in the cache.
        """
        return len(self.cache)

import unittest
from unittest.mock import patch
  # Replace with the actual module name

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(3, 10.0)

    def test_get(self):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), None)

    def test_put(self):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.cache.put("key2", "value2")
        self.assertEqual(self.cache.size(), 2)
        self.cache.put("key3", "value3")
        self.assertEqual(self.cache.size(), 3)
        self.cache.put("key4", "value4")
        self.assertEqual(self.cache.size(), 3)

    def test_delete(self):
        self.cache.put("key1", "value1")
        self.assertTrue(self.cache.delete("key1"))
        self.assertFalse(self.cache.delete("key1"))

    def test_size(self):
        self.assertEqual(self.cache.size(), 0)
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.cache.put("key2", "value2")
        self.assertEqual(self.cache.size(), 2)

    @patch("time.monotonic", return_value=0.0)
    def test_lazy_cleanup(self, mock_monotonic):
        self.cache.put("key1", "value1", ttl=0.1)
        self.cache.get("key1")
        self.assertEqual(self.cache.size(), 1)

        # Simulate time passing to trigger expiration
        mock_monotonic.return_value = 0.11
        self.cache.get("key1")
        self.assertEqual(self.cache.size(), 0)

if __name__ == "__main__":
    unittest.main()

cache = TTLCache(3, 10.0)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # Output: 1
cache.put("d", 4)      # "a" is evicted
print(cache.get("a"))  # Output: None