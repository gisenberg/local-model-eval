from typing import Optional, Any
import time
from collections import deque

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL cache.

        Args:
            capacity (int): Maximum number of items in the cache.
            default_ttl (float): Default time-to-live (in seconds) for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # key -> (value, ttl, timestamp)
        self.lru_list = deque()  # for LRU ordering
        self.size = 0  # current number of items in cache

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the given key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and is not expired, else None.
        """
        if key not in self.cache:
            return None

        value, ttl, timestamp = self.cache[key]
        current_time = time.monotonic()
        if current_time - timestamp > ttl:
            # Expired
            self.delete(key)
            return None

        # Move to the end of the LRU list (mark as recently used)
        self.lru_list.remove(key)
        self.lru_list.append(key)
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update the key-value pair in the cache.

        Args:
            key (str): The key to insert/update.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. If None, use default_ttl.
        """
        if key in self.cache:
            self.cache[key] = (value, ttl if ttl is not None else self.default_ttl, time.monotonic())
            self.lru_list.remove(key)
            self.lru_list.append(key)
            return

        if self.size >= self.capacity:
            # Evict the LRU item
            lru_key = self.lru_list.popleft()
            self.delete(lru_key)

        self.cache[key] = (value, ttl if ttl is not None else self.default_ttl, time.monotonic())
        self.lru_list.append(key)
        self.size += 1

    def delete(self, key: str) -> bool:
        """
        Delete the key from the cache.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        if key not in self.cache:
            return False

        self.lru_list.remove(key)
        del self.cache[key]
        self.size -= 1
        return True

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            int: The current size of the cache.
        """
        return self.size

import unittest
from unittest.mock import patch
from datetime import timedelta
import time

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(2, 1.0)

    @patch("time.monotonic", return_value=0.0)
    def test_get_nonexistent_key(self, mock_time):
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)

    @patch("time.monotonic", return_value=0.0)
    def test_get_expired_key(self, mock_time):
        self.cache.put("key1", "value1", ttl=0.5)
        time.sleep(1)  # Wait for it to expire
        result = self.cache.get("key1")
        self.assertIsNone(result)

    @patch("time.monotonic", return_value=0.0)
    def test_put_with_ttl(self, mock_time):
        self.cache.put("key1", "value1", ttl=1.0)
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")

    @patch("time.monotonic", return_value=0.0)
    def test_delete_key(self, mock_time):
        self.cache.put("key1", "value1")
        self.cache.delete("key1")
        self.assertEqual(self.cache.size(), 0)

    @patch("time.monotonic", return_value=0.0)
    def test_size(self, mock_time):
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.assertEqual(self.cache.size(), 2)

    @patch("time.monotonic", return_value=0.0)
    def test_eviction_on_put(self, mock_time):
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")  # Should evict key1
        self.assertEqual(self.cache.size(), 2)
        self.assertNotIn("key1", self.cache.cache)

if __name__ == "__main__":
    unittest.main()