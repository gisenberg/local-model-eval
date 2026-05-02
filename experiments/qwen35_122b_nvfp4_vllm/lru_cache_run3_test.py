import time
from typing import Optional, Any, Dict

class TTLCache:
    """
    Least Recently Used (LRU) Cache with Time-To-Live (TTL) support.
    
    Uses a doubly-linked list for ordering and a hash map for O(1) access.
    Implements lazy cleanup of expired entries upon access.
    """

    class _Node:
        """Internal node for the doubly-linked list."""
        def __init__(self, key: Any = None, value: Any = None, expire: float = 0.0):
            self.key = key
            self.value = value
            self.expire = expire
            self.prev: 'Optional[TTLCache._Node]' = None
            self.next: 'Optional[TTLCache._Node]' = None

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items added without specific TTL.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, '_Node'] = {}
        
        # Dummy head and tail sentinels to simplify linked list operations
        self.head = self._Node()
        self.tail = self._Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _is_expired(self, node: '_Node') -> bool:
        """Check if a node has passed its expiration time."""
        return node.expire < time.monotonic()

    def _add_to_head(self, node: '_Node') -> None:
        """Add a node to the right after the head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: '_Node') -> None:
        """Remove an existing node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: '_Node') -> None:
        """Move an existing node to the head of the list."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Evict the least recently used item (node before tail)."""
        if self.cache:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value associated with the key.

        Performs lazy cleanup: if the key exists but is expired, it is removed
        and None is returned.

        Args:
            key: The key to retrieve.

        Returns:
            The value if found and valid, else None.
        """
        node = self.cache.get(key)
        if node is None:
            return None
        
        if self._is_expired(node):
            # Lazy cleanup
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        Args:
            key: The key to store.
            value: The value to associate with the key.
            ttl: Optional custom time-to-live. If None, default_ttl is used.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = current_time + effective_ttl

        node = self.cache.get(key)
        
        if node is not None:
            # Key exists: update value and expiry, move to head
            node.value = value
            node.expire = expire_at
            self._move_to_head(node)
            return

        # Key is new
        if len(self.cache) >= self.capacity:
            # Evict LRU to make space
            self._evict_lru()

        new_node = self._Node(key=key, value=value, expire=expire_at)
        self._add_to_head(new_node)
        self.cache[key] = new_node

    def delete(self, key: Any) -> bool:
        """
        Explicitly delete a key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        node = self.cache.get(key)
        if node is not None:
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the number of items currently stored in the cache.
        
        Note: Due to lazy cleanup, this may include expired items 
        that have not yet been accessed.
        """
        return len(self.cache)

import time
import pytest
from unittest.mock import patch, PropertyMock

# Assuming the class above is located in this module for the patch path
# In a real environment, ensure you import from the correct module file.
# e.g., 
@pytest.fixture
def cache_class():
    """Return a reference to the class to facilitate testing structure."""
    return TTLCache

@pytest.fixture
def setup_cache():
    """Create a fresh cache instance."""
    # Using default values for testing
    cap = 2
    default_ttl = 5.0
    # We will patch time.monotonic in the test function, so initialize here
    pass 

# --- Test Suite ---

def test_put_and_get(cache_class):
    """Test 1: Basic functionality of put and get."""
    mock_time = 0.0
    
    with patch('lru_cache_impl.time.monotonic', return_value=mock_time):
        c = cache_class(capacity=10, default_ttl=10)
        c.put("a", 1)
        assert c.get("a") == 1
        assert c.size() == 1

def test_expiration_miss(cache_class):
    """Test 2: Key becomes expired after TTL duration."""
    start_time = 0.0
    
    with patch('lru_cache_impl.time.monotonic', return_value=start_time):
        c = cache_class(capacity=10, default_ttl=2) # TTL 2 seconds
        c.put("x", 100)
        assert c.get("x") == 100

        # Advance time past TTL
        mock_monotonic = patch('lru_cache_impl.time.monotonic', return_value=start_time + 3)
        with mock_monotonic:
            assert c.get("x") is None
            assert c.size() == 0  # Lazy cleanup removes it

def test_capacity_eviction(cache_class):
    """Test 3: LRU eviction when capacity is exceeded."""
    with patch('lru_cache_impl.time.monotonic', return_value=0):
        c = cache_class(capacity=2, default_ttl=10)
        c.put("k1", "v1")
        c.put("k2", "v2")
        # Try to exceed capacity
        c.put("k3", "v3")
        
        # k1 should be evicted (LRU)
        assert c.get("k1") is None
        assert c.get("k2") == "v2"
        assert c.get("k3") == "v3"

def test_update_existing_key(cache_class):
    """Test 4: Updating an existing key resets expiry and moves to front."""
    with patch('lru_cache_impl.time.monotonic', return_value=0):
        c = cache_class(capacity=2, default_ttl=2)
        c.put("k1", "v1")
        c.put("k2", "v2")
        
        # Update k1 (moves to head)
        c.put("k1", "v_new")
        
        # Access k2 (older than updated k1)
        # Now force eviction by adding new key
        with patch('lru_cache_impl.time.monotonic', return_value=0): # Reset time
             c.put("k3", "v3")
             
             # k2 should be evicted because k1 was touched recently
             assert c.get("k2") is None
             assert c.get("k1") == "v_new"

def test_delete_key(cache_class):
    """Test 5: Explicit deletion of a key."""
    with patch('lru_cache_impl.time.monotonic', return_value=0):
        c = cache_class(capacity=5, default_ttl=10)
        c.put("target", "data")
        result = c.delete("target")
        
        assert result is True
        assert c.get("target") is None
        assert c.size() == 0
        
        # Deleting non-existent
        res = c.delete("non_existent")
        assert res is False

def test_lazy_cleanup_on_access(cache_class):
    """Test 6: Size includes expired items until accessed (Lazy Cleanup)."""
    t_start = 0.0
    
    with patch('lru_cache_impl.time.monotonic', return_value=t_start):
        c = cache_class(capacity=10, default_ttl=1)
        c.put("expired_k", "val", ttl=1)
        
        # Advance time to expire
        with patch('lru_cache_impl.time.monotonic', return_value=t_start + 5):
            # Size is technically still 1 because we haven't touched the expired item
            # Note: Depending on implementation, some might argue for checking expiry on insert.
            # But with LAZY cleanup, exp items sit in memory until accessed or evicted.
            initial_size = c.size() 
            assert initial_size == 1
            
            # Now access it -> Lazy cleanup happens
            assert c.get("expired_k") is None
            
            # Now size should reflect removal
            assert c.size() == 0