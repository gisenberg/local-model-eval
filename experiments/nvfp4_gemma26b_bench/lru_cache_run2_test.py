import time
from typing import Any, Dict, Optional

class Node:
    """A node in the doubly-linked list."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Uses a Doubly-Linked List for LRU eviction and a Dict for O(1) access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Args:
            capacity: Maximum number of items in cache.
            default_ttl: Default lifetime of an item in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Sentinel nodes for the Doubly-Linked List
        self.head = Node(None, None, 0)  # Most Recently Used
        self.tail = Node(None, None, 0)  # Least Recently Used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node right after the head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Moves an existing node to the front (MRU)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node, current_time: float) -> bool:
        """Checks if a node's TTL has passed."""
        return current_time >= node.expiry

    def get(self, key: Any) -> Any:
        """
        Returns the value of the key. 
        If key is expired or not found, returns None and removes it.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        now = time.monotonic()

        if self._is_expired(node, now):
            self.delete(key)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key.
        If capacity is reached, evicts the Least Recently Used (LRU) item.
        """
        now = time.monotonic()
        actual_ttl = ttl if ttl is not None else self.default_ttl
        expiry = now + actual_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU (the node before tail)
                lru_node = self.tail.prev
                self.delete(lru_node.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Removes a key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)

    def size(self) -> int:
        """Returns the current number of valid (non-expired) items."""
        now = time.monotonic()
        # Note: This is a 'lazy' size. It doesn't clean up expired items 
        # unless get() or put() is called, but we filter for the count.
        return sum(1 for node in self.cache.values() if not self._is_expired(node, now))

import unittest
from unittest.mock import patch

class TestTTLCache(unittest.TestCase):

    @patch('time.monotonic')
    def test_basic_put_get(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1)
        self.assertEqual(cache.get("a"), 1)

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1) # Expires at 110.0
        
        # Advance time to 111.0
        mock_time.return_value = 111.0
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.size(), 0)

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access "a" to make it MRU, "b" becomes LRU
        cache.get("a")
        
        # Add "c", should evict "b"
        cache.put("c", 3)
        
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        # Put with short TTL
        cache.put("short", "val", ttl=5) # Expires at 105.0
        
        mock_time.return_value = 106.0
        self.assertIsNone(cache.get("short"))

    @patch('time.monotonic')
    def test_delete(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1)
        cache.delete("a")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.size(), 0)

    @patch('time.monotonic')
    def test_size_with_lazy_cleanup(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=10, default_ttl=10)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        mock_time.return_value = 111.0 # Both expired
        # size() should reflect that items are expired
        self.assertEqual(cache.size(), 0)

if __name__ == '__main__':
    unittest.main()