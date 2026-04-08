from typing import Any, Optional
import time


class _Node:
    """Internal doubly-linked list node for cache entries."""
    def __init__(self, key: str, value: Any, timestamp: float, ttl: float):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Supports O(1) average time complexity for get, put, and delete operations.
    Uses a doubly-linked list + hash map for LRU tracking.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items (if not overridden).
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # Most Recently Used
        self._tail: Optional[_Node] = None  # Least Recently Used
    
    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = node.next = None
    
    def _add_to_head(self, node: _Node) -> None:
        """Add a node to the head (MRU position) of the list."""
        node.next = self._head
        node.prev = None
        if self._head:
            self._head.prev = node
        self._head = node
        if not self._tail:
            self._tail = node
    
    def _move_to_head(self, node: _Node) -> None:
        """Move a node to the head (MRU position) of the list."""
        if self._head == node:
            return
        self._remove(node)
        self._add_to_head(node)
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() - node.timestamp > node.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and is not expired.
        
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self._map:
            return None
        node = self._map[key]
        if self._is_expired(node):
            self._remove(node)
            del self._map[key]
            return None
        self._move_to_head(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds (overrides default_ttl).
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self._map:
            # Update existing entry
            node = self._map[key]
            node.value = value
            node.timestamp = current_time
            node.ttl = effective_ttl
            self._move_to_head(node)
        else:
            # First, remove expired items from tail (lazy cleanup)
            while self._tail and self._is_expired(self._tail):
                self._remove(self._tail)
                del self._map[self._tail.key]
            
            # If still at capacity, evict LRU non-expired item
            if len(self._map) >= self.capacity:
                if self._tail:
                    self._remove(self._tail)
                    del self._map[self._tail.key]
                # If no tail, all were expired and cleared, so we have room
            
            # Insert new node
            node = _Node(key, value, current_time, effective_ttl)
            self._map[key] = node
            self._add_to_head(node)
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key not in self._map:
            return False
        node = self._map[key]
        self._remove(node)
        del self._map[key]
        return True
    
    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items in the cache.
        """
        # Collect expired keys first to avoid modifying dict during iteration
        expired_keys = [key for key, node in self._map.items() if self._is_expired(node)]
        for key in expired_keys:
            node = self._map[key]
            self._remove(node)
            del self._map[key]
        return len(self._map)


# Tests
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_time():
    """Fixture to mock time.monotonic for deterministic testing."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 0.0
        yield mock_monotonic


def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(2, 10.0)
    
    # Put and get
    cache.put("a", 1)
    assert cache.get("a") == 1
    
    # Get non-existent
    assert cache.get("b") is None
    
    # Put another
    cache.put("b", 2)
    assert cache.get("b") == 2


def test_capacity_eviction(mock_time):
    """Test LRU eviction when capacity is reached."""
    cache = TTLCache(2, 10.0)
    
    # Fill cache
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a")
    
    # Add 'c', should evict 'b' (LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1  # Still there
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3  # New item


def test_ttl_expiry(mock_time):
    """Test that items expire after TTL."""
    cache = TTLCache(2, 1.0)  # 1 second TTL
    
    cache.put("a", 1)
    assert cache.get("a") == 1
    
    # Advance time beyond TTL
    mock_time.return_value = 2.0
    
    assert cache.get("a") is None
    assert cache.size() == 0


def test_custom_per_key_ttl(mock_time):
    """Test custom TTL per key overrides default."""
    cache = TTLCache(2, 1.0)  # Default 1 second
    
    # Item with custom TTL of 5 seconds
    cache.put("a", 1, ttl=5.0)
    
    # Advance to 2 seconds - should still be valid
    mock_time.return_value = 2.0
    assert cache.get("a") == 1
    
    # Reset time for next item
    mock_time.return_value = 0.0
    cache.put("b", 2)  # Default TTL
    
    # Advance to 2 seconds - should be expired
    mock_time.return_value = 2.0
    assert cache.get("b") is None


def test_delete(mock_time):
    """Test delete operation."""
    cache = TTLCache(2, 10.0)
    
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    
    # Delete non-existent
    assert cache.delete("a") is False


def test_size_mixed_expired_valid(mock_time):
    """Test size() with mixed expired and valid items."""
    cache = TTLCache(3, 1.0)  # Default 1 second TTL
    
    # Add items with different TTLs
    cache.put("a", 1)          # TTL 1s
    cache.put("b", 2, ttl=5.0) # TTL 5s
    cache.put("c", 3)          # TTL 1s
    
    # Advance time to 2 seconds
    mock_time.return_value = 2.0
    
    # 'a' and 'c' should be expired, 'b' should be valid
    assert cache.size() == 1
    
    # Verify individual states
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") is None