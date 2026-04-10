import time
, Optional, Dict, Tuple

class Node:
    """
    Represents a node in the doubly linked list.
    Stores the key, value, and the expiration timestamp.
    """
    def __init__(self, key: Any, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    """
    An LRU Cache implementation with Time-To-Live (TTL) support.

    Uses a combination of a hash map (for O(1) lookups) and a doubly-linked list
    (for O(1) LRU operations).
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initializes the TTLCache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default time-to-live in seconds for entries
                          if no specific TTL is provided during put.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        
        # Map: key -> Node reference
        self.cache: Dict[Any, Node] = {}
        
        # Doubly Linked List setup: Sentinel nodes for head (MRU) and tail (LRU)
        self.head = Node(None, None, 0.0)  # Most Recently Used (MRU)
        self.tail = Node(None, None, 0.0)  # Least Recently Used (LRU)
        
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self._size: int = 0

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _add_to_head(self, node: Node) -> None:
        """Adds a node right after the head (making it MRU)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        """Moves an existing node to the head of the list (MRU)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _cleanup_expired(self) -> None:
        """
        Performs lazy cleanup by checking the LRU end of the list.
        Removes expired items without iterating the whole cache.
        """
        current = self.tail.prev
        while current != self.head and current.expiry_time < time.monotonic():
            # Found an expired item at the LRU end
            self._remove_node(current)
            del self.cache[current.key]
            self._size -= 1
            current = current.prev

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves an item from the cache. Updates its usage status (MRU).

        Args:
            key: The key of the item to retrieve.

        Returns:
            The value if found and not expired, otherwise None.
        """
        node = self.cache.get(key)
        if not node:
            return None

        # Check for expiration upon access
        if node.expiry_time < time.monotonic():
            self.delete(key)  # Clean up expired item
            return None

        # Valid item: Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates an item in the cache.

        Args:
            key: The key to store the value under.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default_ttl if None.
        """
        # Determine expiry time
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + ttl_to_use

        if key in self.cache:
            # Update existing item
            node = self.cache[key]
            node.value = value
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # New item
            if self._size >= self.capacity:
                # Evict LRU item (which is just before the tail sentinel)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]
                    self._size -= 1
            
            # Create and insert new node
            new_node = Node(key, value, expiry_time)
            self.cache[key] = new_node
            self._add_to_head(new_node)
            self._size += 1

    def delete(self, key: Any) -> bool:
        """
        Removes a specific key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        node = self.cache.get(key)
        if not node:
            return False
        
        self._remove_node(node)
        del self.cache[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        # Run cleanup before reporting size to ensure accuracy
        self._cleanup_expired()
        return self._size

# Example Usage (optional, for local testing)
if __name__ == '__main__':
    # Cache capacity 3, default TTL 5 seconds
    cache = TTLCache(capacity=3, default_ttl=5.0)

    print("--- Initial Puts ---")
    cache.put("a", 100)
    cache.put("b", 200)
    cache.put("c", 300)
    print(f"Size after 3 puts: {cache.size()}") # 3

    print("\n--- Accessing (MRU update) ---")
    print(f"Get 'a': {cache.get('a')}") # Access 'a', moves it to MRU
    print(f"Size: {cache.size()}") # 3

    print("\n--- Eviction Test ---")
    # 'b' is now LRU
    cache.put("d", 400) # 'b' should be evicted
    print(f"Get 'b' after eviction: {cache.get('b')}") # None
    print(f"Size after eviction: {cache.size()}") # 3 (a, c, d)

    print("\n--- TTL Test ---")
    # Put 'e' with a short TTL (1 second)
    cache.put("e", 500, ttl=1.0)
    print(f"Size before wait: {cache.size()}") # 4 (if capacity was larger, but here it's 3)
    
    # Wait for TTL to pass
    print("Waiting 1.5 seconds...")
    time.sleep(1.5)
    
    # Accessing 'e' should trigger cleanup and return None
    print(f"Get 'e' after TTL: {cache.get('e')}") # None
    print(f"Size after expired access: {cache.size()}") # Should be 3 (d, a, c)

import unittest
from unittest.mock import patch
import time


class TestTTLCache(unittest.TestCase):

    def setUp(self):
        """Setup a fresh cache before each test."""
        self.capacity = 2
        self.default_ttl = 10.0
        self.cache = TTLCache(self.capacity, self.default_ttl)
        # Set a base time for mocking
        self.base_time = 1000.0

    @patch('time.monotonic', return_value=1000.0)
    def test_01_basic_put_and_get(self, mock_time):
        """Test basic insertion and retrieval."""
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        
        retrieved = self.cache.get("key1")
        self.assertEqual(retrieved, "value1")
        self.assertEqual(self.cache.size(), 1) # Size should not change on successful get

    @patch('time.monotonic', return_value=1000.0)
    def test_02_lru_eviction(self, mock_time):
        """Test that the Least Recently Used item is evicted when capacity is reached."""
        self.cache.put("k1", 1)  # MRU: k1
        self.cache.put("k2", 2)  # MRU: k2, LRU: k1
        
        # Access k1 to make it MRU
        self.cache.get("k1") # MRU: k1, LRU: k2
        
        # Add k3, which should evict k2 (the current LRU)
        self.cache.put("k3", 3) # MRU: k3, LRU: k2 (evicted)
        
        self.assertEqual(self.cache.size(), 2)
        self.assertIsNone(self.cache.get("k2"))
        self.assertEqual(self.cache.get("k1"), 1)
        self.assertEqual(self.cache.get("k3"), 3)

    @patch('time.monotonic', return_value=1000.0)
    def test_03_ttl_expiration_on_get(self, mock_time):
        """Test that an item expires and is removed when accessed after TTL."""
        # Put item with 1 second TTL
        self.cache.put("exp_key", "data", ttl=1.0)
        
        # Advance time past the TTL
        mock_time.return_value = 1001.1
        
        # Accessing it should trigger cleanup and return None
        result = self.cache.get("exp_key")
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic', return_value=1000.0)
    def test_04_ttl_update_on_put(self, mock_time):
        """Test that putting an item again resets its TTL."""
        self.cache.put("ttl_key", "v1", ttl=1.0) # Expires at 1001.0
        
        # Advance time slightly, but not past TTL
        mock_time.return_value = 1000.5
        
        # Update the item, resetting TTL to 10 seconds from 1000.5
        self.cache.put("ttl_key", "v2", ttl=10.0) 
        
        # Advance time past the *original* expiration time (1001.0)
        mock_time.return_value = 1002.0
        
        # The item should still be valid because the TTL was reset
        result = self.cache.get("ttl_key")
        self.assertEqual(result, "v2")
        self.assertEqual(self.cache.size(), 1)

    @patch('time.monotonic', return_value=1000.0)
    def test_05_lazy_cleanup_on_size(self, mock_time):
        """Test that expired items are cleaned up when size() is called."""
        # Put item with 0.1s TTL
        self.cache.put("expired", "data", ttl=0.1)
        
        # Advance time past TTL
        mock_time.return_value = 1000.2
        
        # Size() should trigger cleanup
        size = self.cache.size()
        self.assertEqual(size, 0)
        self.assertIsNone(self.cache.get("expired"))

    @patch('time.monotonic', return_value=1000.0)
    def test_06_delete_functionality(self, mock_time):
        """Test explicit deletion of an item."""
        self.cache.put("del_key", "to_delete")
        self.assertEqual(self.cache.size(), 1)
        
        # Delete the item
        success = self.cache.delete("del_key")
        self.assertTrue(success)
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.get("del_key"))
        
        # Try deleting a non-existent key
        self.assertFalse(self.cache.delete("non_existent"))


if __name__ == '__main__':
    unittest.main()