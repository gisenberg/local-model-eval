import time
, Optional, Dict, Tuple

class Node:
    """
    A node for the doubly linked list.
    Stores the key, value, and the time the entry expires.
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

    Uses a combination of a hash map (dict) and a doubly linked list 
    to achieve O(1) average time complexity for get and put operations.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initializes the TTLCache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        
        # Hash map: key -> Node
        self.cache: Dict[Any, Node] = {}
        
        # Doubly Linked List setup (Head and Tail act as sentinels)
        self.head = Node(None, None, 0.0)  # Dummy head
        self.tail = Node(None, None, 0.0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node right after the head (Most Recently Used)."""
        head_next = self.head.next
        
        self.head.next = node
        node.prev = self.head
        
        node.next = head_next
        head_next.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Moves an existing node to the front (MRU position)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has passed its expiration time."""
        return time.monotonic() >= node.expiry_time

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves an item from the cache. If found and not expired, 
        it is marked as Most Recently Used (MRU).

        Args:
            key: The key of the item to retrieve.

        Returns:
            The value if found and valid, otherwise None.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            # Lazy cleanup: remove expired item
            self.delete(key)
            return None
        
        # Move to front (LRU logic)
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates an item in the cache. Handles eviction if capacity is reached.

        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional time-to-live in seconds. Uses default_ttl if None.
        """
        if key in self.cache:
            # Update existing item
            node = self.cache[key]
            node.value = value
            
            # Update expiry time
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            node.expiry_time = time.monotonic() + ttl_to_use
            
            # Mark as MRU
            self._move_to_front(node)
            return

        # New item
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + ttl_to_use
        new_node = Node(key, value, expiry_time)

        if len(self.cache) >= self.capacity:
            # Evict LRU item (the node just before the tail sentinel)
            lru_node = self.tail.prev
            if lru_node != self.head:
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
        
        # Add new node to the front (MRU)
        self._add_to_front(new_node)
        self.cache[key] = new_node

    def delete(self, key: Any) -> bool:
        """
        Removes a specific key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return len(self.cache)

# Example Usage (Optional, for local testing)
if __name__ == '__main__':
    cache = TTLCache(capacity=3, default_ttl=2)
    
    print("--- Initial Puts ---")
    cache.put("a", 10) # TTL 2s
    cache.put("b", 20) # TTL 2s
    cache.put("c", 30) # TTL 2s
    print(f"Size: {cache.size()}") # 3

    print("\n--- Accessing (MRU Update) ---")
    print(f"Get 'a': {cache.get('a')}") # Access 'a', moves 'a' to front
    
    print("\n--- Eviction Test ---")
    cache.put("d", 40) # Evicts 'b' (LRU)
    print(f"Size after eviction: {cache.size()}") # 3
    print(f"Get 'b' (should be None): {cache.get('b')}") # None
    print(f"Get 'c' (should be 30): {cache.get('c')}") # 30

    print("\n--- TTL Test ---")
    print("Waiting 2.5 seconds for TTL expiration...")
    time.sleep(2.5)
    
    print(f"Get 'a' after expiry (should be None): {cache.get('a')}") # None (Lazy cleanup)
    print(f"Size after expired get: {cache.size()}") # 2 (a is gone)

import pytest
from unittest.mock import patch, MagicMock
import time


# Define a fixed time reference for predictable testing
MOCK_TIME_START = 1000.0

@pytest.fixture
def cache():
    """Fixture to provide a fresh TTLCache instance for each test."""
    # Capacity 2, Default TTL 1.0 second
    return TTLCache(capacity=2, default_ttl=1.0)

@pytest.fixture(autouse=True)
def mock_time(monkeypatch):
    """
    Patches time.monotonic() to return a predictable, increasing value 
    during the test execution.
    """
    call_count = 0
    def mock_monotonic():
        nonlocal call_count
        # Increment time by 0.1s for each call
        return MOCK_TIME_START + (call_count * 0.1)
    
    monkeypatch.setattr(time, 'monotonic', mock_monotonic)
    
    # Yield control back to the test function
    yield
    
    # Cleanup (optional, but good practice)
    pass

def test_initialization(cache):
    """Tests if the cache initializes correctly."""
    assert cache.capacity == 2
    assert cache.default_ttl == 1.0
    assert cache.size() == 0

def test_put_and_get_basic_lru(cache):
    """Tests basic LRU behavior without TTL."""
    # Time 1000.0 (Initial)
    cache.put("k1", "v1") # MRU
    cache.put("k2", "v2") # LRU
    
    # Access k1, making it MRU
    assert cache.get("k1") == "v1"
    
    # Put k3, should evict k2 (LRU)
    cache.put("k3", "v3")
    
    assert cache.size() == 2
    assert cache.get("k2") is None  # Evicted
    assert cache.get("k1") == "v1"  # Still present
    assert cache.get("k3") == "v3"  # Newest

def test_ttl_expiration_on_get(cache):
    """Tests that an item expires and is removed upon access."""
    # Time 1000.0
    cache.put("k_exp", "exp_val", ttl=0.2) # Expires at 1000.2
    
    # Time 1000.1 (Still valid)
    assert cache.get("k_exp") == "exp_val"
    
    # Advance time past expiry (Mock time advances by 0.1s per call)
    # We need 3 calls to pass 0.2s (1000.0 -> 1000.1 -> 1000.2)
    
    # Call 1 (Time 1000.1)
    cache.get("k_exp") 
    
    # Call 2 (Time 1000.2) - Should be expired now
    assert cache.get("k_exp") is None
    
    # Check size after lazy cleanup
    assert cache.size() == 0

def test_put_with_custom_ttl(cache):
    """Tests setting a specific TTL different from the default."""
    # Time 1000.0
    cache.put("k_short", "short", ttl=0.1) # Expires at 1000.1
    cache.put("k_long", "long", ttl=5.0)  # Expires much later
    
    # Advance time to 1000.2 (past short TTL)
    # Need 2 calls to pass 0.2s
    cache.get("k_short") # Time 1000.1
    cache.get("k_short") # Time 1000.2 (Triggering check)
    
    assert cache.get("k_short") is None
    assert cache.get("k_long") == "long" # Long TTL remains valid

def test_delete_functionality(cache):
    """Tests the explicit deletion of an item."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    
    assert cache.size() == 2
    
    # Delete existing key
    assert cache.delete("k1") is True
    assert cache.size() == 1
    assert cache.get("k1") is None
    
    # Try deleting non-existent key
    assert cache.delete("k99") is False
    assert cache.size() == 1

def test_size_after_eviction_and_deletion(cache):
    """Tests size tracking across multiple operations."""
    cache.put("k1", 1)
    cache.put("k2", 2)
    cache.put("k3", 3) # Evicts k1 (if capacity was 2)
    
    # Since capacity is 2, k1 should be gone
    assert cache.size() == 2
    
    cache.delete("k2")
    assert cache.size() == 1
    
    cache.put("k4", 4) # Evicts k3
    assert cache.size() == 2
    assert cache.get("k3") is None