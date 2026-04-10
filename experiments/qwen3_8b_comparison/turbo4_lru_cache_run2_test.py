from typing import Any, Optional, Dict, List
import time
from collections import deque

class TTLCache:
    """
    A cache that supports LRU (Least Recently Used) eviction and TTL (Time to Live) expiration.

    Attributes:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for items without an explicit TTL.
        _cache: Dictionary mapping keys to (value, ttl, timestamp) tuples.
        _lru_order: Deque to maintain the order of access (LRU).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items without an explicit TTL.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: Dict[str, Any] = {}
        self._lru_order: deque = deque()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the given key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and hasn't expired, else None.
        """
        if key not in self._cache:
            return None

        value, ttl, timestamp = self._cache[key]
        if self._is_expired(timestamp, ttl):
            self.delete(key)
            return None

        # Move the key to the end of the deque to mark it as recently used
        self._lru_order.remove(key)
        self._lru_order.append(key)
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update the value for the given key with an optional TTL.

        Args:
            key (str): The key to store.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. If None, use default_ttl.
        """
        if key in self._cache:
            # Update the value and TTL
            self._cache[key] = (value, ttl if ttl is not None else self.default_ttl, time.monotonic())
            # Move the key to the end of the deque to mark it as recently used
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return

        if len(self._cache) >= self.capacity:
            # Evict the least recently used item
            lru_key = self._lru_order.popleft()
            self.delete(lru_key)

        self._cache[key] = (value, ttl if ttl is not None else self.default_ttl, time.monotonic())
        self._lru_order.append(key)

    def delete(self, key: str) -> bool:
        """
        Delete the item with the given key.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        if key not in self._cache:
            return False

        del self._cache[key]
        if key in self._lru_order:
            self._lru_order.remove(key)
        return True

    def size(self) -> int:
        """
        Get the current number of items in the cache.

        Returns:
            int: Number of items currently in the cache.
        """
        return len(self._cache)

    def _is_expired(self, timestamp: float, ttl: float) -> bool:
        """
        Check if an item has expired based on its timestamp and TTL.

        Args:
            timestamp (float): The time the item was inserted or updated.
            ttl (float): Time-to-live in seconds.

        Returns:
            bool: True if the item has expired, False otherwise.
        """
        return time.monotonic() - timestamp > ttl

import unittest
from unittest.mock import patch
from datetime import timedelta
from typing import Any, Optional, Dict, List
  # Replace with your actual module name

class TestTTLCache(unittest.TestCase):
    def test_get_existing_key(self):
        cache = TTLCache(1, 10)
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    @patch("time.monotonic")
    def test_get_expired_key(self, mock_monotonic):
        mock_monotonic.return_value = 100.0
        cache = TTLCache(1, 1)
        cache.put("key1", "value1", ttl=1)
        mock_monotonic.return_value = 101.0
        self.assertIsNone(cache.get("key1"))

    @patch("time.monotonic")
    def test_put_with_ttl(self, mock_monotonic):
        mock_monotonic.return_value = 100.0
        cache = TTLCache(1, 10)
        cache.put("key1", "value1", ttl=2)
        mock_monotonic.return_value = 102.0
        self.assertEqual(cache.get("key1"), "value1")
        mock_monotonic.return_value = 103.0
        self.assertIsNone(cache.get("key1"))

    @patch("time.monotonic")
    def test_put_no_ttl(self, mock_monotonic):
        mock_monotonic.return_value = 100.0
        cache = TTLCache(1, 10)
        cache.put("key1", "value1")
        mock_monotonic.return_value = 101.0
        self.assertEqual(cache.get("key1"), "value1")
        mock_monotonic.return_value = 110.0
        self.assertIsNone(cache.get("key1"))

    @patch("time.monotonic")
    def test_delete_key(self, mock_monotonic):
        cache = TTLCache(1, 10)
        cache.put("key1", "value1")
        self.assertTrue(cache.delete("key1"))
        self.assertFalse(cache.delete("key1"))

    @patch("time.monotonic")
    def test_size(self, mock_monotonic):
        cache = TTLCache(3, 10)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        self.assertEqual(cache.size(), 3)