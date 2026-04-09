import time
, Optional, Dict, Tuple

class Node:
    """
    Represents a node in the doubly linked list.
    Stores the key, value, and the time when the entry expires.
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

    Uses a doubly linked list for O(1) LRU operations and a dictionary
    for O(1) key lookups.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initializes the TTLCache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default time-to-live in seconds for entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        
        # Dictionary mapping key to its corresponding Node object
        self.cache_map: Dict[Any, Node] = {}
        
        # Dummy head and tail nodes for the doubly linked list
        self.head = Node(None, None, 0.0)  # Most recently used (MRU) side
        self.tail = Node(None, None, 0.0)  # Least recently used (LRU) side
        
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: Node) -> None:
        """Adds a node right after the head (making it MRU)."""
        # Node goes between head and head.next
        next_node = self.head.next
        
        self.head.next = node
        node.prev = self.head
        
        node.next = next_node
        next_node.prev = node

    def _move_to_head(self, node: Node) -> None:
        """Moves an existing node to the head of the list (MRU)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _cleanup_expired(self) -> None:
        """
        Performs lazy cleanup by iterating from the tail (LRU) and removing
        any expired entries.
        """
        current = self.tail.prev
        while current != self.head:
            if current.expiry_time <= time.monotonic():
                # Expired: remove it
                self._remove_node(current)
                del self.cache_map[current.key]
            else:
                # Since we iterate from LRU, if this one isn't expired, 
                # subsequent ones (closer to MRU) won't be either.
                break
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves a value from the cache. Updates its position to MRU if found.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if not found or expired.
        """
        self._cleanup_expired() # Check for expiration before lookup
        
        if key not in self.cache_map:
            return None
        
        node = self.cache_map[key]
        
        # Check expiration again (in case cleanup didn't run recently)
        if node.expiry_time <= time.monotonic():
            self.delete(key) # Clean up the expired item
            return None
        
        # Hit: Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair in the cache.

        Args:
            key: The key.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default_ttl if None.
        """
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + ttl_to_use
        
        if key in self.cache_map:
            # Update existing item
            node = self.cache_map[key]
            node.value = value
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # New item
            
            # Check capacity before insertion
            if len(self.cache_map) >= self.capacity:
                # Evict LRU item (node before tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self._remove_node(lru_node)
                    del self.cache_map[lru_node.key]
            
            # Create and insert new node
            new_node = Node(key, value, expiry_time)
            self.cache_map[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """
        Removes a key-value pair from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        if key in self.cache_map:
            node = self.cache_map[key]
            self._remove_node(node)
            del self.cache_map[key]
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        # Note: This returns the count of items in the map, which is accurate 
        # immediately after cleanup or insertion/deletion.
        return len(self.cache_map)

# Example Usage (optional, for testing environment)
if __name__ == '__main__':
    cache = TTLCache(capacity=3, default_ttl=1.0)
    cache.put("a", 10)
    cache.put("b", 20)
    print(f"Size after 2 puts: {cache.size()}") # Output: 2
    
    time.sleep(0.5)
    cache.put("c", 30)
    print(f"Size after 3 puts: {cache.size()}") # Output: 3
    
    # Test LRU eviction
    cache.put("d", 40) # Should evict 'a'
    print(f"Size after eviction: {cache.size()}") # Output: 3
    print(f"Get 'a': {cache.get('a')}") # Output: None
    print(f"Get 'b': {cache.get('b')}") # Output: 20 (b moves to MRU)
    
    # Test TTL
    print("Waiting for TTL...")
    time.sleep(1.5)
    print(f"Get 'c' after TTL: {cache.get('c')}") # Output: None (and 'c' is cleaned up)
    print(f"Size after TTL cleanup: {cache.size()}") # Output: 2 (b, d remain)

import pytest
import time
from unittest.mock import patch


# Define a fixed time reference for testing
START_TIME = 1000.0

@pytest.fixture
def cache():
    """Fixture to provide a fresh cache instance for each test."""
    # Capacity 3, Default TTL 1.0 second
    return TTLCache(capacity=3, default_ttl=1.0)

@pytest.fixture(autouse=True)
def mock_time(monkeypatch):
    """
    Mocks time.monotonic() to return a controllable, increasing time value.
    This fixture ensures that time advances predictably across tests.
    """
    mock_time_value = START_TIME
    
    def mock_monotonic():
        nonlocal mock_time_value
        return mock_time_value

    # We use a side_effect function to allow tests to manually advance time
    monkeypatch.setattr(time, 'monotonic', mock_monotonic)
    
    # Provide a helper function to advance time within the test scope
    def advance_time(seconds: float):
        nonlocal mock_time_value
        mock_time_value += seconds
        return mock_time_value

    return advance_time

# --- Test Cases ---

def test_initialization(cache):
    """Tests correct initialization parameters."""
    assert cache.capacity == 3
    assert cache.default_ttl == 1.0
    assert cache.size() == 0

def test_put_and_get_basic(cache):
    """Tests basic insertion and retrieval."""
    cache.put("key1", "value1")
    assert cache.size() == 1
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None

def test_lru_eviction(cache, mock_time):
    """Tests that the Least Recently Used item is evicted when capacity is reached."""
    # 1. Fill the cache (A, B, C)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("C", 3)
    assert cache.size() == 3

    # 2. Access B to make it MRU (Order: A, C, B)
    cache.get("B") 

    # 3. Insert D. A should be evicted (LRU)
    cache.put("D", 4) 
    
    assert cache.size() == 3
    assert cache.get("A") is None  # Evicted
    assert cache.get("B") == 2    # Still present
    assert cache.get("D") == 4    # New item present

def test_update_existing_key_and_move_to_mru(cache):
    """Tests updating a value and ensuring the item moves to the MRU position."""
    cache.put("K1", 10)
    cache.put("K2", 20)
    
    # Access K1 (K1 becomes MRU)
    cache.get("K1") 
    
    # Update K2 (K2 is now the LRU candidate)
    cache.put("K2", 200)
    
    # Insert K3. K2 should be evicted because K1 was accessed last.
    cache.put("K3", 30) 
    
    assert cache.size() == 2
    assert cache.get("K2") is None # Evicted
    assert cache.get("K1") == 10  # Still present
    assert cache.get("K3") == 30  # New item present

def test_ttl_expiration_on_get(cache, mock_time):
    """Tests that an item expires and is removed when accessed via get()."""
    # Put item with a short TTL (0.1s)
    cache.put("expiring_key", "data", ttl=0.1)
    
    # Time 1000.0: Item is fresh
    assert cache.get("expiring_key") == "data"
    
    # Advance time past the TTL
    mock_time(0.2) 
    
    # Accessing it should trigger cleanup and return None
    assert cache.get("expiring_key") is None
    assert cache.size() == 0

def test_lazy_cleanup_on_put(cache, mock_time):
    """Tests that expired items are cleaned up when a new item is inserted."""
    # Put item with a very short TTL
    cache.put("old_key", "expired", ttl=0.01)
    
    # Advance time past TTL
    mock_time(0.1) 
    
    # Cache size is 1, but it's expired.
    assert cache.size() == 1
    
    # Insert a new item. This should trigger cleanup of 'old_key'.
    cache.put("new_key", "fresh")
    
    # Size should now be 1 (only 'new_key')
    assert cache.size() == 1
    assert cache.get("old_key") is None # Confirms cleanup happened

def test_delete_functionality(cache):
    """Tests the explicit deletion mechanism."""
    cache.put("to_delete", "data")
    assert cache.size() == 1
    
    # Successful delete
    assert cache.delete("to_delete") is True
    assert cache.size() == 0
    assert cache.get("to_delete") is None
    
    # Attempt to delete non-existent key
    assert cache.delete("nonexistent") is False
    assert cache.size() == 0