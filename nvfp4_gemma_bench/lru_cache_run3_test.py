import time
, Optional

class Node:
    """Doubly Linked List Node to store cache entries."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Evicts the least recently used item when capacity is reached.
    Items are considered expired if current time >= expiry time.
    """
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Map[key, Node]
        
        # Dummy head and tail for the doubly linked list
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has passed its expiry time."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache. Returns None if expired or not found."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self.delete(key)
            return None
        
        # Move to front (LRU update)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a value with a specific or default TTL."""
        if key in self.cache:
            self.delete(key)
        
        # Calculate expiry time
        duration = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + duration
        
        # Evict LRU if capacity reached
        if len(self.cache) >= self.capacity:
            # Evict the node just before the tail dummy
            lru_node = self.tail.prev
            self.delete(lru_node.key)
            
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a specific key from the cache."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        """Return the current number of items in the cache."""
        # Note: This returns the raw size. 
        # In a real scenario, you might want to filter expired items here.
        return len(self.cache)

import unittest
from unittest.mock import patch
 # Assuming the code above is in ttllru.py

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
        
        mock_time.return_value = 111.0 # Move time past TTL
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.size(), 0)

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make 'b' the LRU
        cache.get("a")
        
        # Put 'c', should evict 'b'
        cache.put("c", 3)
        
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        # Set a very short TTL for this specific key
        cache.put("short", "val", ttl=1) 
        
        mock_time.return_value = 101.1
        self.assertIsNone(cache.get("short"))

    @patch('time.monotonic')
    def test_update_existing_key(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.put("a", 2) # Update value
        self.assertEqual(cache.get("a"), 2)
        self.assertEqual(cache.size(), 1)

    @patch('time.monotonic')
    def test_delete(self, mock_time):
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=60)
        cache.put("a", 1)
        cache.delete("a")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.size(), 0)

if __name__ == "__main__":
    unittest.main()