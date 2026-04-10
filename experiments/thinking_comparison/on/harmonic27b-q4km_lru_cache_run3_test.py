from typing import Any, Optional
import time

class Node:
    """Doubly-linked list node for LRU tracking."""
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """LRU cache with time-based expiration."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """Initialize cache with capacity and default TTL in seconds.
        
        Args:
            capacity: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, tuple[Node, float]] = {}  # key -> (node, expiration_time)
        self.head: Node = Node("", "")  # dummy head
        self.tail: Node = Node("", "")  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_head(self, node: Node) -> None:
        """Add node right after head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def _is_expired(self, expiration_time: float) -> bool:
        """Check if item is expired."""
        return time.monotonic() >= expiration_time
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items (lazy cleanup)."""
        expired_keys = [key for key, (node, exp_time) in self.cache.items() 
                       if self._is_expired(exp_time)]
        for key in expired_keys:
            node, _ = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for key if exists and not expired. Makes key most-recently-used.
        
        Args:
            key: The key to look up
            
        Returns:
            Value if key exists and not expired, None otherwise
        """
        if key not in self.cache:
            return None
        
        node, expiration_time = self.cache[key]
        
        if self._is_expired(expiration_time):
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to most recently used
        self._remove_node(node)
        self._add_to_head(node)
        
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert/update key-value pair. Evict LRU non-expired item if at capacity.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        expiration_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing key
            node, _ = self.cache[key]
            node.value = value
            self.cache[key] = (node, expiration_time)
            # Move to most recently used
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Insert new key - need to handle capacity
            # First, clean up any expired items to make room
            self._cleanup_expired()
            
            if len(self.cache) >= self.capacity:
                # Evict LRU item (which is now guaranteed to be non-expired after cleanup)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    lru_key = lru_node.key
                    self._remove_node(lru_node)
                    del self.cache[lru_key]
            
            new_node = Node(key, value)
            self._add_to_head(new_node)
            self.cache[key] = (new_node, expiration_time)
    
    def delete(self, key: str) -> bool:
        """Remove key from cache. Return True if it existed.
        
        Args:
            key: The key to remove
            
        Returns:
            True if key existed and was removed, False otherwise
        """
        if key not in self.cache:
            return False
        
        node, _ = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True
    
    def size(self) -> int:
        """Return count of non-expired items.
        
        Returns:
            Number of non-expired items in cache
        """
        self._cleanup_expired()
        return len(self.cache)


# Tests
import pytest
from unittest.mock import patch

def test_basic_get_put():
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None

def test_capacity_eviction():
    """Test LRU eviction when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # This should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    
    # Access key2 to make it MRU
    cache.get("key2")
    cache.put("key4", "value4")  # This should evict key3
    
    assert cache.get("key3") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key4") == "value4"

def test_ttl_expiry():
    """Test TTL expiration."""
    time_counter = [0.0]
    
    def mock_monotonic():
        return time_counter[0]
    
    with patch('time.monotonic', side_effect=mock_monotonic):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # At time 0, put item that expires at time 10
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Advance time to 5, item still valid
        time_counter[0] = 5.0
        assert cache.get("key1") == "value1"
        
        # Advance time to 15, item expired
        time_counter[0] = 15.0
        assert cache.get("key1") is None

def test_custom_ttl():
    """Test custom per-key TTL."""
    time_counter = [0.0]
    
    def mock_monotonic():
        return time_counter[0]
    
    with patch('time.monotonic', side_effect=mock_monotonic):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("key1", "value1", ttl=5.0)  # expires at time 5
        cache.put("key2", "value2", ttl=20.0)  # expires at time 20
        
        # At time 5, key1 expired but key2 still valid
        time_counter[0] = 5.0
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

def test_delete():
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    assert cache.delete("nonexistent") is False
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None

def test_size_with_mixed_items():
    """Test size() with mixed expired and valid items."""
    time_counter = [0.0]
    
    def mock_monotonic():
        return time_counter[0]
    
    with patch('time.monotonic', side_effect=mock_monotonic):
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1", ttl=2.0)  # expires at time 2
        cache.put("key2", "value2", ttl=10.0)  # expires at time 10
        cache.put("key3", "value3", ttl=2.0)  # expires at time 2
        
        assert cache.size() == 3  # All items present
        
        # Advance time to 5 (key1 and key3 expired)
        time_counter[0] = 5.0
        assert cache.size() == 1  # Only key2 remains