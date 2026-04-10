import time
, Any

class Node:
    """Doubly Linked List Node to store cache data."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Evicts the Least Recently Used item when capacity is reached.
    Items are lazily removed if they have expired.
    """
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Map key -> Node
        self.size = 0
        
        # Dummy head and tail for the doubly linked list
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Removes a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node):
        """Adds a node immediately after the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node):
        """Moves an existing node to the front (most recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has exceeded its TTL."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve item from cache. Returns None if expired or missing."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update item. Evicts LRU if capacity is exceeded."""
        if key in self.cache:
            self.delete(key)

        # Calculate expiry time
        ttl_value = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + ttl_value
        
        # Handle capacity
        if self.size >= self.capacity:
            # Evict the least recently used (the one before dummy tail)
            lru_node = self.tail.prev
            if lru_node != self.head:
                self.delete(lru_node.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self.size += 1

    def delete(self, key: Any) -> None:
        """Remove a specific key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            self.size -= 1

    def current_size(self) -> int:
        """Returns the number of items currently in the cache."""
        return self.size

import unittest
from unittest.mock import patch

class TestTTLCache(unittest.TestCase):

    @patch('time.monotonic')
    def test_basic_put_get(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        self.assertEqual(cache.get("a"), 1)

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1) # Expires at 110.0
        
        mock_time.return_value = 105.0
        self.assertEqual(cache.get("a"), 1) # Still valid
        
        mock_time.return_value = 111.0
        self.assertIsNone(cache.get("a")) # Expired
        self.assertEqual(cache.current_size(), 0)

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")    # Access 'a' to make it MRU
        cache.put("c", 3) # Should evict 'b'
        
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        cache.put("short", "val", ttl=1) # Expires at 101.0
        
        mock_time.return_value = 102.0
        self.assertIsNone(cache.get("short"))

    @patch('time.monotonic')
    def test_delete(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.delete("a")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.current_size(), 0)

    @patch('time.monotonic')
    def test_update_existing_key(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("a", 2) # Update value
        self.assertEqual(cache.get("a"), 2)
        self.assertEqual(cache.current_size(), 1)

if __name__ == "__main__":
    unittest.main()