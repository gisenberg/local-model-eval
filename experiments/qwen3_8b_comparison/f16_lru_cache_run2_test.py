from typing import Optional, Any, Dict, List
import time
from collections import deque

class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-to-Live (TTL) support.
    
    Attributes:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for items not specified with TTL.
        cache (Dict): Hash map to store key-value pairs.
        lru_list (Deque): Doubly-linked list to maintain the order of access.
        time_to_expire (Dict): Maps keys to their expiration times.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items not specified with TTL.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Any] = {}
        self.lru_list: deque = deque()
        self.time_to_expire: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for the given key from the cache.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and hasn't expired, else None.
        """
        if key not in self.cache:
            return None

        # Check if the item has expired
        if self._is_expired(key):
            self.delete(key)
            return None

        # Move the key to the end of the LRU list (mark as recently used)
        self.lru_list.remove(key)
        self.lru_list.append(key)

        return self.cache[key]

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update the value for the given key in the cache.

        Args:
            key (str): The key to store.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. If None, use default_ttl.
        """
        if key in self.cache:
            self.lru_list.remove(key)
            self.lru_list.append(key)
            self.cache[key] = value
            self.time_to_expire[key] = time.monotonic() + (ttl or self.default_ttl)
            return

        # If cache is full, remove the least recently used item
        if len(self.cache) >= self.capacity:
            lru_key = self.lru_list.popleft()
            self.delete(lru_key)

        self.lru_list.append(key)
        self.cache[key] = value
        self.time_to_expire[key] = time.monotonic() + (ttl or self.default_ttl)

    def delete(self, key: str) -> bool:
        """
        Delete the item with the given key from the cache.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        if key not in self.cache:
            return False

        self.lru_list.remove(key)
        del self.cache[key]
        del self.time_to_expire[key]
        return True

    def size(self) -> int:
        """
        Return the number of items currently in the cache.

        Returns:
            int: The number of items.
        """
        return len(self.cache)

    def _is_expired(self, key: str) -> bool:
        """
        Check if the item with the given key has expired.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the item has expired, False otherwise.
        """
        return time.monotonic() >= self.time_to_expire.get(key, 0)

import unittest
from unittest.mock import patch
from typing import Optional, Any
  # Replace with the actual module name

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(3, 10.0)

    @patch("time.monotonic", return_value=100.0)
    def test_get_existing_key(self, mock_time):
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")

    @patch("time.monotonic", return_value=100.0)
    def test_get_expired_key(self, mock_time):
        self.cache.put("key1", "value1", ttl=0.1)
        time.sleep(0.1)  # Wait for the key to expire
        result = self.cache.get("key1")
        self.assertIsNone(result)

    @patch("time.monotonic", return_value=100.0)
    def test_put_new_key(self, mock_time):
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.assertEqual(self.cache.get("key1"), "value1")

    @patch("time.monotonic", return_value=100.0)
    def test_put_update_key(self, mock_time):
        self.cache.put("key1", "value1")
        self.cache.put("key1", "value2")
        self.assertEqual(self.cache.get("key1"), "value2")

    @patch("time.monotonic", return_value=100.0)
    def test_delete_key(self, mock_time):
        self.cache.put("key1", "value1")
        result = self.cache.delete("key1")
        self.assertTrue(result)
        self.assertEqual(self.cache.size(), 0)

    @patch("time.monotonic", return_value=100.0)
    def test_size(self, mock_time):
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        self.assertEqual(self.cache.size(), 3)
        self.cache.delete("key1")
        self.assertEqual(self.cache.size(), 2)

if __name__ == "__main__":
    unittest.main()