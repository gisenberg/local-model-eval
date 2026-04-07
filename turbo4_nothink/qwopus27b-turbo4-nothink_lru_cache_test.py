from typing import Any, Optional
import time


class _Node:
    """Internal node for the doubly-linked list."""
    def __init__(self, key: str, value: Any, expiration_time: float):
        self.key = key
        self.value = value
        self.expiration_time = expiration_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU cache with TTL (Time To Live) expiration.
    
    Supports O(1) average time complexity for get, put, delete, and size operations.
    Uses a doubly-linked list and hash map internally.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        self._size = 0
        # Dummy head and tail nodes for the doubly-linked list
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.expiration_time
    
    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node: _Node) -> None:
        """Add a node to the head (most recently used position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
    
    def _evict_lru(self) -> None:
        """
        Evict the least recently used non-expired item.
        If all items are expired, clear them all.
        """
        current = self._tail.prev
        while current != self._head:
            if self._is_expired(current):
                # Lazy cleanup: remove expired item
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                current = current.prev
            else:
                # Found non-expired LRU item, evict it
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                return
        # If we reach here, all items were expired and removed
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the given key.
        
        Returns the value if the key exists and is not expired, otherwise None.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value associated with the key, or None if not found or expired.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, updates the value and expiration time.
        If the cache is at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert or update.
            value: The value to store.
            ttl: Optional custom time-to-live in seconds. If None, uses default_ttl.
        """
        if key in self._cache:
            # Update existing item
            node = self._cache[key]
            node.value = value
            node.expiration_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # New item
            if self._size == self.capacity:
                self._evict_lru()
            
            node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key not in self._cache:
            return False
        
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.
        
        Note: Expired items are removed lazily on access, so this returns the
        current count of items in the cache.
        
        Returns:
            The number of items in the cache.
        """
        return self._size


# Tests
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_time():
    """Fixture to mock time.monotonic for deterministic time control."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 0.0
        yield mock_monotonic


def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") is None


def test_capacity_eviction(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1 (LRU)
    
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_ttl_expiry(mock_time):
    """Test that items expire after TTL."""
    cache = TTLCache(capacity=2, default_ttl=1.0)
    cache.put("key1", "value1")
    
    mock_time.return_value = 2.0  # Advance time by 2 seconds
    assert cache.get("key1") is None


def test_custom_ttl(mock_time):
    """Test custom per-key TTL."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("key1", "value1", ttl=1.0)
    cache.put("key2", "value2", ttl=10.0)
    
    mock_time.return_value = 2.0
    assert cache.get("key1") is None  # Expired
    assert cache.get("key2") == "value2"  # Not expired


def test_delete(mock_time):
    """Test delete operation."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("key1", "value1")
    
    assert cache.delete("key1") is True
    assert cache.delete("key1") is False
    assert cache.get("key1") is None


def test_size_mixed(mock_time):
    """Test size with mixed expired and valid items."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("key1", "value1", ttl=1.0)
    cache.put("key2", "value2", ttl=10.0)
    cache.put("key3", "value3", ttl=10.0)
    
    mock_time.return_value = 2.0
    # key1 is expired but not yet cleaned up
    # Access key1 to trigger cleanup
    assert cache.get("key1") is None
    # Now size should be 2 (key2 and key3)
    assert cache.size() == 2