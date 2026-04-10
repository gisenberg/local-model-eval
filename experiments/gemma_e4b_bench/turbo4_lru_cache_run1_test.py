import time
, Optional, Dict, Tuple

class Node:
    """
    A node for the doubly linked list.
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

    Uses a doubly linked list for O(1) LRU operations and a hash map for O(1) lookups.
    Cleanup of expired items is done lazily upon access.
    """
    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.default_ttl = default_ttl
        
        # Map: key -> Node object
        self.cache: Dict[Any, Node] = {}
        
        # Doubly Linked List setup (Head is MRU, Tail is LRU)
        # Sentinel nodes simplify boundary checks
        self.head = Node(None, None, 0.0)  # Most Recently Used (MRU)
        self.tail = Node(None, None, 0.0)  # Least Recently Used (LRU)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the linked list."""
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
        """Moves an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[Node]:
        """Removes and returns the LRU node (just before the tail sentinel)."""
        lru_node = self.tail.prev
        if lru_node == self.head:
            return None  # Cache is empty
        
        self._remove_node(lru_node)
        return lru_node

    def _is_expired(self, node: Node) -> bool:
        """Checks if the node's TTL has passed."""
        return node.expiry_time < time.monotonic()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves an item by key. Updates its position to MRU if found and not expired.
        Returns None if the key is not found or the item is expired.
        """
        node = self.cache.get(key)
        if not node:
            return None

        # Lazy cleanup: Check for expiration
        if self._is_expired(node):
            self.delete(key)  # Remove expired item
            return None

        # Hit: Move to MRU position
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        If ttl is None, uses the default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl
        
        expiry_time = time.monotonic() + ttl

        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # New key: Check capacity and evict if necessary
            if len(self.cache) >= self.capacity:
                # Evict LRU item
                lru_node = self._pop_tail()
                if lru_node:
                    del self.cache[lru_node.key]
            
            # Create and insert new node
            new_node = Node(key, value, expiry_time)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """Removes a key-value pair from the cache."""
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        # Note: This returns the count of items currently in the map, 
        # but doesn't perform a full sweep for expired items.
        return len(self.cache)

# Example Usage (Optional, for local testing)
if __name__ == '__main__':
    # Capacity 3, Default TTL 5 seconds
    cache = TTLCache(capacity=3, default_ttl=5.0)

    print("--- Initial Puts ---")
    cache.put("a", 100) # TTL 5s
    cache.put("b", 200) # TTL 5s
    cache.put("c", 300) # TTL 5s
    print(f"Size after 3 puts: {cache.size()}")

    print("\n--- Accessing (LRU Update) ---")
    print(f"Get 'a': {cache.get('a')}") # 'a' becomes MRU
    
    print("\n--- Eviction Test ---")
    # 'b' is now LRU
    cache.put("d", 400) # Should evict 'b'
    print(f"Size after eviction: {cache.size()}")
    print(f"Get 'b' (should be None): {cache.get('b')}")
    print(f"Get 'd': {cache.get('d')}")

    print("\n--- TTL Test (Simulated Wait) ---")
    # Manually set a short TTL for testing expiration
    cache.put("short_lived", 99, ttl=0.1) 
    print(f"Size before wait: {cache.size()}")
    
    time.sleep(0.2) # Wait longer than the TTL
    
    # Accessing 'short_lived' should trigger lazy cleanup and return None
    print(f"Get 'short_lived' after TTL: {cache.get('short_lived')}")
    print(f"Size after lazy cleanup: {cache.size()}")

import pytest
import time
from unittest.mock import patch


# Define a fixed time base for predictable testing
FIXED_TIME_BASE = 1000.0

@pytest.fixture
def cache():
    """Fixture to provide a fresh cache instance for each test."""
    return TTLCache(capacity=2, default_ttl=10.0)

# --- Test Cases ---

def test_initialization_and_capacity(cache):
    """Tests basic initialization and capacity constraints."""
    assert cache.capacity == 2
    assert cache.size() == 0
    
    cache.put("k1", 1)
    cache.put("k2", 2)
    assert cache.size() == 2
    
    # Test eviction
    cache.put("k3", 3) # Should evict k1 (LRU)
    assert cache.size() == 2
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    assert cache.get("k3") is not None

def test_get_hit_and_lru_update(cache):
    """Tests that accessing an item moves it to the Most Recently Used (MRU) position."""
    cache.put("k1", 1)
    cache.put("k2", 2) # k2 is MRU, k1 is LRU
    
    # Access k1, making it MRU
    assert cache.get("k1") == 1
    
    # Now, put k3. It should evict k2 (the original LRU)
    cache.put("k3", 3)
    
    assert cache.get("k2") is None
    assert cache.get("k1") == 1
    assert cache.get("k3") == 3

def test_put_update_and_ttl_reset(cache):
    """Tests updating an existing key resets its TTL and moves it to MRU."""
    # Use a very short TTL for the initial put
    cache.put("k1", 1, ttl=0.1)
    
    # Mock time to simulate passage of time (e.g., 0.2s)
    with patch('time.monotonic', return_value=FIXED_TIME_BASE + 0.2):
        # Accessing it should fail due to TTL
        assert cache.get("k1") is None
        
        # Now, update it. This should reset the expiration time.
        cache.put("k1", 100, ttl=5.0)
        
        # Mock time again, but only slightly past the original expiration
        with patch('time.monotonic', return_value=FIXED_TIME_BASE + 0.3):
            # It should now be valid because the TTL was reset
            assert cache.get("k1") == 100

def test_lazy_cleanup_on_get_expired(cache):
    """Tests that calling get() on an expired item triggers cleanup and returns None."""
    # Set a TTL that is already in the past relative to the mocked time
    expired_time = FIXED_TIME_BASE - 1.0
    
    # Manually create a node and place it in the cache for testing cleanup logic
    node = cache.cache["k1"] # Assuming k1 exists or mocking the structure
    
    # We must use put to ensure the internal structure is correct, but force the expiry
    cache.put("k_exp", 99, ttl=0.001)
    
    # Mock time to be far in the future relative to the put time
    with patch('time.monotonic', return_value=FIXED_TIME_BASE + 1.0):
        # Accessing it should trigger cleanup and return None
        result = cache.get("k_exp")
        
        assert result is None
        # The item should be removed from the internal map
        assert "k_exp" not in cache.cache

def test_ttl_with_default_ttl(cache):
    """Tests that the default_ttl is used when no specific TTL is provided."""
    # Default TTL is 10.0 seconds
    
    # Mock time to simulate 5 seconds passing
    with patch('time.monotonic', return_value=FIXED_TIME_BASE + 5.0):
        cache.put("k_default", 1)
        assert cache.get("k_default") == 1
        
    # Mock time to simulate 15 seconds passing (past the 10s default)
    with patch('time.monotonic', return_value=FIXED_TIME_BASE + 15.0):
        # Accessing it should trigger cleanup
        assert cache.get("k_default") is None
        assert cache.size() == 0

def test_delete_functionality(cache):
    """Tests successful deletion and attempting to delete a non-existent key."""
    cache.put("k1", 1)
    cache.put("k2", 2)
    
    # Test successful delete
    assert cache.delete("k1") == True
    assert cache.get("k1") is None
    assert cache.size() == 1
    
    # Test deleting non-existent key
    assert cache.delete("k_missing") == False
    assert cache.size() == 1 # Size should remain unchanged

# Run tests using: pytest