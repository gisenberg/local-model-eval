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
        Initializes the TTL Cache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default time-to-live in seconds for entries
                         that do not specify an individual TTL.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
            
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        
        # Dictionary mapping key to its corresponding Node object
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail nodes for the doubly linked list
        self.head = Node(None, None, 0.0)
        self.tail = Node(None, None, 0.0)
        
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Internal helper to remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Internal helper to add a node right after the head (Most Recently Used)."""
        # Node goes between head and head.next
        next_node = self.head.next
        
        node.prev = self.head
        node.next = next_node
        
        self.head.next = node
        next_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Moves an existing node to the front of the list (MRU)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _cleanup_expired(self) -> None:
        """
        Performs lazy cleanup: removes expired items from the tail end 
        (Least Recently Used side) if they are found during access or put.
        Note: A full scan is not performed on every call for O(1) average time.
        """
        # We only check the nodes near the tail for simplicity in this lazy model.
        # A more rigorous implementation might iterate from the tail until a non-expired item is found.
        current = self.tail.prev
        while current != self.head:
            if current.expiry_time < time.monotonic():
                # Item is expired, remove it
                self._remove_node(current)
                del self.cache[current.key]
                current = current.prev
            else:
                # Since the list is ordered by usage, if this one isn't expired, 
                # we stop checking the tail for this call.
                break


    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves an item from the cache. If found, marks it as MRU.
        Checks for expiration upon retrieval.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The value associated with the key, or None if not found or expired.
        """
        node = self.cache.get(key)
        if not node:
            return None

        # 1. Check for expiration
        if node.expiry_time < time.monotonic():
            # Expired: remove it and return None
            self._remove_node(node)
            del self.cache[key]
            return None

        # 2. Valid: Move to front (MRU) and return value
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates an item in the cache.

        Args:
            key: The key of the item.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default_ttl if None.
        """
        current_time = time.monotonic()
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + ttl_to_use

        if key in self.cache:
            # Update existing item
            node = self.cache[key]
            node.value = value
            node.expiry_time = expiry_time
            self._move_to_front(node)
        else:
            # New item
            if len(self.cache) >= self.capacity:
                # Cache is full, evict LRU item (node just before tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]
            
            # Create and insert new node
            new_node = Node(key, value, expiry_time)
            self.cache[key] = new_node
            self._add_to_front(new_node)
            
        # Perform cleanup after insertion/update to handle potential immediate expirations
        self._cleanup_expired()


    def delete(self, key: Any) -> bool:
        """
        Removes a specific key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        # We rely on the cache dictionary size, which is accurate after cleanup.
        return len(self.cache)

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Capacity 3, Default TTL 5 seconds
    cache = TTLCache(capacity=3, default_ttl=5.0)
    
    print("--- Initial Puts ---")
    cache.put("A", 100)
    cache.put("B", 200)
    cache.put("C", 300)
    print(f"Size after 3 puts: {cache.size()}") # Expected: 3

    print("\n--- LRU Test (Access B) ---")
    print(f"Get B: {cache.get('B')}") # B becomes MRU
    
    print("\n--- Eviction Test (Put D) ---")
    # A is now LRU, so A should be evicted
    cache.put("D", 400) 
    print(f"Size after eviction: {cache.size()}") # Expected: 3
    print(f"Get A (should be None): {cache.get('A')}") # Expected: None
    print(f"Get D: {cache.get('D')}") # Expected: 400

    print("\n--- TTL Test ---")
    # Put E with a very short TTL (0.1s)
    cache.put("E", 500, ttl=0.1)
    print(f"Size before wait: {cache.size()}") # Expected: 3
    
    time.sleep(0.2)
    
    # Access E - should be expired
    print(f"Get E after wait (should be None): {cache.get('E')}") # Expected: None
    print(f"Size after expired get: {cache.size()}") # Expected: 2 (E removed)

    # Check cleanup mechanism (though lazy cleanup is triggered by access)
    cache.put("F", 600)
    print(f"Size after F put: {cache.size()}") # Expected: 3 (D, C, F)

import unittest
from unittest.mock import patch
import time


class TestTTLCache(unittest.TestCase):
    
    def setUp(self):
        """Set up a fresh cache before each test."""
        self.capacity = 3
        self.default_ttl = 10.0
        self.cache = TTLCache(self.capacity, self.default_ttl)

    @patch('time.monotonic', return_value=100.0)
    def test_01_put_and_get_basic(self, mock_time):
        """Test basic put and get functionality."""
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.size(), 1)
        
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")
        self.assertEqual(self.cache.size(), 1) # Accessing doesn't change size

    @patch('time.monotonic', return_value=100.0)
    def test_02_lru_eviction(self, mock_time):
        """Test that the Least Recently Used item is evicted when capacity is reached."""
        self.cache.put("A", 1) # MRU: A
        self.cache.put("B", 2) # MRU: B, LRU: A
        self.cache.put("C", 3) # MRU: C, LRU: A
        
        # Access A to make it MRU
        self.cache.get("A") # MRU: A, LRU: B
        
        # Put D, B should be evicted
        self.cache.put("D", 4) # MRU: D, LRU: C
        
        self.assertEqual(self.cache.size(), 3)
        self.assertIsNone(self.cache.get("B")) # B was evicted
        self.assertEqual(self.cache.get("A"), 1) # A is still present
        self.assertEqual(self.cache.get("D"), 4)

    @patch('time.monotonic', return_value=100.0)
    def test_03_ttl_expiration_on_get(self, mock_time):
        """Test that an item expires and is removed upon retrieval."""
        # Set TTL to 0.1 seconds
        self.cache.put("expiring_key", "secret", ttl=0.1)
        
        # Advance time past the TTL
        mock_time.return_value = 100.2 
        
        result = self.cache.get("expiring_key")
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic', return_value=100.0)
    def test_04_ttl_default_ttl_usage(self, mock_time):
        """Test that the default_ttl is used when no TTL is provided."""
        self.cache.put("default_key", "value")
        
        # Advance time past the default TTL (10.0s)
        mock_time.return_value = 110.1
        
        result = self.cache.get("default_key")
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic', return_value=100.0)
    def test_05_delete_functionality(self, mock_time):
        """Test successful deletion and attempting to delete a non-existent key."""
        self.cache.put("to_delete", "data")
        self.assertTrue(self.cache.delete("to_delete"))
        self.assertEqual(self.cache.size(), 0)
        
        # Try deleting again
        self.assertFalse(self.cache.delete("to_delete"))
        self.assertEqual(self.cache.size(), 0)

    @patch('time.monotonic', return_value=100.0)
    def test_06_update_and_move_to_front(self, mock_time):
        """Test that updating a key moves it to the MRU position."""
        self.cache.put("A", 1) # LRU: A
        self.cache.put("B", 2) # LRU: A, MRU: B
        
        # Update A
        self.cache.put("A", 100) # A is updated and moved to MRU
        
        # Put C, B should now be evicted because A is MRU
        self.cache.put("C", 3) # LRU: B, MRU: C
        
        self.assertEqual(self.cache.get("A"), 100) # A is present
        self.assertIsNone(self.cache.get("B")) # B was evicted
        self.assertEqual(self.cache.get("C"), 3)


if __name__ == '__main__':
    unittest.main()