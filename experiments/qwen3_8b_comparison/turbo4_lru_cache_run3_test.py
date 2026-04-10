from typing import Optional, Any, Dict
import time
from collections import deque

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL cache with a given capacity and default TTL.
        
        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Any] = {}
        self.items: deque = deque()
        self.ttl_map: Dict[str, float] = {}  # key: expiration time

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for the given key if it exists and hasn't expired.
        
        Args:
            key (str): The key to look up.
        
        Returns:
            Optional[Any]: The value if found and not expired, else None.
        """
        if key not in self.cache:
            return None

        # Check if the item is expired
        if self._is_expired(key):
            self.delete(key)
            return None

        # Move the accessed item to the end of the deque (LRU)
        self.items.remove(key)
        self.items.append(key)
        return self.cache[key]

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update an item in the cache with a given TTL.
        
        Args:
            key (str): The key to store.
            value (Any): The value to store.
            ttl (float): Time-to-live in seconds. If None, use default TTL.
        """
        if key in self.cache:
            self.items.remove(key)
            self.items.append(key)
            self.cache[key] = value
            if ttl is not None:
                self.ttl_map[key] = time.monotonic() + ttl
            else:
                self.ttl_map[key] = time.monotonic() + self.default_ttl
            return

        # If cache is full, remove the LRU item
        if len(self.items) >= self.capacity:
            lru_key = self.items.popleft()
            self.delete(lru_key)

        self.items.append(key)
        self.cache[key] = value
        self.ttl_map[key] = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

    def delete(self, key: str) -> bool:
        """
        Delete the item with the given key from the cache.
        
        Args:
            key (str): The key to delete.
        
        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        if key in self.cache:
            del self.cache[key]
            self.items.remove(key)
            del self.ttl_map[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the number of items currently in the cache.
        
        Returns:
            int: Number of items in the cache.
        """
        return len(self.items)

    def _is_expired(self, key: str) -> bool:
        """
        Check if the item with the given key has expired.
        
        Args:
            key (str): The key to check.
        
        Returns:
            bool: True if expired, False otherwise.
        """
        return time.monotonic() >= self.ttl_map[key]

import unittest
from unittest.mock import patch
from datetime import timedelta
from typing import Optional

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(2, 1.0)

    @patch("time.monotonic")
    def test_get_existing_key(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")

    @patch("time.monotonic")
    def test_get_expired_key(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.cache.put("key1", "value1", ttl=0.5)
        mock_monotonic.return_value = 1.0  # Now it's expired
        result = self.cache.get("key1")
        self.assertIsNone(result)

    @patch("time.monotonic")
    def test_put_with_ttl(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.cache.put("key1", "value1", ttl=1.0)
        self.assertEqual(self.cache.cache["key1"], "value1")
        self.assertEqual(self.cache.ttl_map["key1"], 1.0)

    @patch("time.monotonic")
    def test_put_with_default_ttl(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.ttl_map["key1"], 1.0)

    @patch("time.monotonic")
    def test_delete_key(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.cache.put("key1", "value1")
        self.assertTrue(self.cache.delete("key1"))
        self.assertFalse("key1" in self.cache.cache)

    @patch("time.monotonic")
    def test_size(self, mock_monotonic):
        mock_monotonic.return_value = 0.0
        self.assertEqual(self.cache.size(), 0)
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.cache.put("key2", "value2")
        self.assertEqual(self.cache.size(), 2)
        self.cache.put("key3", "value3")
        self.assertEqual(self.cache.size(), 2)  # LRU eviction