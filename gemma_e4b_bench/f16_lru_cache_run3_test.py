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
    An LRU Cache implementation that supports Time-To-Live (TTL) expiration.

    Uses a combination of a hash map (for O(1) lookups) and a doubly-linked list
    (for O(1) LRU management).
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initializes the TTLCache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
            
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        
        # Hash map: key -> Node
        self.cache: Dict[Any, Node] = {}
        
        # Doubly Linked List setup (Head is MRU, Tail is LRU)
        # Sentinel nodes to simplify boundary checks
        self.head = Node(None, None, -1)  # Most Recently Used (MRU) end
        self.tail = Node(None, None, -1)  # Least Recently Used (LRU) end
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_head(self, node: Node) -> None:
        """Adds a node right after the head (making it MRU)."""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        """Moves an existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _cleanup_expired(self) -> None:
        """
        Lazily cleans up expired items starting from the LRU end (tail).
        This is called during get/put operations.
        """
        current = self.tail.prev
        while current != self.head:
            node: Node = current
            
            # Check if the node has expired
            if time.monotonic() >= node.expiry_time:
                # Expired: Remove from cache and list
                self._remove_node(node)
                del self.cache[node.key]
                current = node.prev
            else:
                # Found a non-expired item, stop cleanup
                break

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves a value from the cache. If found, marks it as MRU and checks TTL.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if not found or expired.
        """
        node = self.cache.get(key)
        if not node:
            return None

        # 1. Check TTL (Lazy Cleanup)
        if time.monotonic() >= node.expiry_time:
            # Expired: Remove it immediately
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # 2. Valid: Update LRU status
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair in the cache.

        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default_ttl if None.
        """
        # Determine expiration time
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
            # Check capacity and evict LRU if necessary
            if len(self.cache) >= self.capacity:
                # Evict the LRU item (node just before the tail sentinel)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]
            
            # Create and add new node
            new_node = Node(key, value, expiry_time)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """
        Removes a key-value pair from the cache.

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
        return True

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return len(self.cache)

# Example Usage (Optional, for local testing)
if __name__ == '__main__':
    cache = TTLCache(capacity=3, default_ttl=2)
    
    print("--- Initial Puts ---")
    cache.put("a", 100)
    cache.put("b", 200)
    cache.put("c", 300)
    print(f"Size after 3 puts: {cache.size()}") # 3

    # Access 'a' to make it MRU
    print(f"Get 'a': {cache.get('a')}") # 100
    
    # Put 'd', should evict 'b' (LRU)
    print("\n--- Eviction Test ---")
    cache.put("d", 400)
    print(f"Size after eviction: {cache.size()}") # 3
    print(f"Get 'b' (should be evicted): {cache.get('b')}") # None
    print(f"Get 'c' (should exist): {cache.get('c')}") # 300

    # Test TTL
    print("\n--- TTL Test ---")
    cache.put("e", 500, ttl=0.1) # Short TTL
    print(f"Size before wait: {cache.size()}") # 4 (if capacity allowed, but here it's 3)
    
    # Wait for TTL to pass
    print("Waiting 0.2 seconds...")
    time.sleep(0.2)
    
    # Access 'e' - should trigger cleanup and return None
    print(f"Get 'e' after expiry: {cache.get('e')}") # None
    print(f"Size after TTL cleanup: {cache.size()}") # Should be 3 (if 'e' was the LRU)

import unittest
from unittest.mock import patch, MagicMock
import time


class TestTTLCache(unittest.TestCase):
    
    def setUp(self):
        """Setup a fresh cache before each test."""
        self.capacity = 3
        self.default_ttl = 1.0
        self.cache = TTLCache(self.capacity, self.default_ttl)

    # --- Test Basic Functionality (LRU & Put/Get) ---

    def test_basic_put_and_get(self):
        """Test standard insertion and retrieval."""
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_lru_eviction(self):
        """Test that the Least Recently Used item is evicted when capacity is reached."""
        self.cache.put("k1", 1)  # MRU: k1
        self.cache.put("k2", 2)  # MRU: k2, LRU: k1
        self.cache.put("k3", 3)  # MRU: k3, LRU: k1
        
        # Access k1 to make it MRU
        self.cache.get("k1") # MRU: k1, LRU: k2
        
        # Add k4, should evict k2
        self.cache.put("k4", 4) # MRU: k4, LRU: k3
        
        self.assertEqual(self.cache.size(), 3)
        self.assertIsNone(self.cache.get("k2")) # k2 was evicted
        self.assertEqual(self.cache.get("k1"), 1) # k1 should remain
        self.assertEqual(self.cache.get("k4"), 4)

    def test_update_existing_key(self):
        """Test that updating a key moves it to the MRU position."""
        self.cache.put("k1", 1)
        self.cache.put("k2", 2)
        self.cache.put("k3", 3) # LRU: k1
        
        # Update k1
        self.cache.put("k1", 100) # MRU: k1, LRU: k2
        
        # Add k4, should evict k2
        self.cache.put("k4", 4)
        
        self.assertEqual(self.cache.get("k1"), 100)
        self.assertIsNone(self.cache.get("k2")) # k2 evicted
        self.assertEqual(self.cache.size(), 3)

    # --- Test TTL Functionality ---

    @patch('time.monotonic')
    def test_ttl_expiration_on_get(self, mock_monotonic):
        """Test that an item expires and is removed when accessed via get()."""
        # Set initial time
        start_time = 1000.0
        mock_monotonic.return_value = start_time
        
        # Put item with 0.1s TTL
        self.cache.put("expiring_key", "data", ttl=0.1)
        
        # Advance time past TTL
        future_time = start_time + 0.2
        mock_monotonic.return_value = future_time
        
        # Accessing it should trigger cleanup and return None
        result = self.cache.get("expiring_key")
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic')
    def test_ttl_does_not_expire_before_time(self, mock_monotonic):
        """Test that an item remains valid if accessed before its TTL expires."""
        start_time = 1000.0
        mock_monotonic.return_value = start_time
        
        # Put item with 1.0s TTL
        self.cache.put("valid_key", "data", ttl=1.0)
        
        # Advance time slightly, but less than TTL
        future_time = start_time + 0.5
        mock_monotonic.return_value = future_time
        
        # Accessing it should succeed
        result = self.cache.get("valid_key")
        
        self.assertEqual(result, "data")
        self.assertEqual(self.cache.size(), 1)

    @patch('time.monotonic')
    def test_ttl_cleanup_on_put(self, mock_monotonic):
        """Test that expired items are cleaned up when a new item is put."""
        start_time = 1000.0
        mock_monotonic.return_value = start_time
        
        # Put an expired item (TTL=0.1)
        self.cache.put("expired_key", "old_data", ttl=0.1)
        
        # Advance time past TTL
        future_time = start_time + 0.2
        mock_monotonic.return_value = future_time
        
        # Put a new item. This should trigger cleanup of 'expired_key'.
        self.cache.put("new_key", "fresh_data")
        
        self.assertEqual(self.cache.size(), 1)
        self.assertIsNone(self.cache.get("expired_key")) # Should be gone
        self.assertEqual(self.cache.get("new_key"), "fresh_data")

    # --- Test Deletion and Size ---

    def test_delete_existing_key(self):
        """Test successful deletion of an existing key."""
        self.cache.put("k1", 1)
        self.cache.put("k2", 2)
        self.assertTrue(self.cache.delete("k1"))
        self.assertEqual(self.cache.size(), 1)
        self.assertIsNone(self.cache.get("k1"))

    def test_delete_nonexistent_key(self):
        """Test deletion attempt on a key that does not exist."""
        self.cache.put("k1", 1)
        self.assertFalse(self.cache.delete("k99"))
        self.assertEqual(self.cache.size(), 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)