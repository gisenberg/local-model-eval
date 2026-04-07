import time
from typing import Any, Optional


class _Node:
    """Internal doubly-linked list node for LRU cache."""
    __slots__ = ('key', 'value', 'expiration_time', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, expiration_time: float):
        self.key = key
        self.value = value
        self.expiration_time = expiration_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Supports O(1) average time complexity for get, put, and delete operations.
    Uses a doubly-linked list to maintain LRU order and a hash map for O(1) key lookup.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items (unless overridden).
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        
        # Dummy head (MRU end) and tail (LRU end)
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        
    def _move_to_head(self, node: _Node) -> None:
        """Move node to head (MRU position)."""
        # Remove from current position
        node.prev.next = node.next
        node.next.prev = node.prev
        
        # Insert at head
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        
    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        
    def _evict_lru(self) -> None:
        """Evict the least-recently-used non-expired item, or clear all if all expired."""
        current_time = time.monotonic()
        
        # Traverse from tail (LRU end) to find first non-expired item
        node = self._tail.prev
        while node != self._head:
            if node.expiration_time >= current_time:
                # Found non-expired LRU item, evict it
                self._remove_node(node)
                del self._cache[node.key]
                return
            node = node.prev
        
        # All items are expired, clear them all
        while self._head.next != self._tail:
            node = self._head.next
            self._remove_node(node)
            del self._cache[node.key]
            
    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key.
        
        Returns the value if the key exists and is not expired, otherwise None.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        current_time = time.monotonic()
        
        # Check if expired
        if node.expiration_time < current_time:
            # Remove expired item
            self._remove_node(node)
            del self._cache[key]
            return None
            
        # Move to head (MRU)
        self._move_to_head(node)
        return node.value
        
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, updates the value and expiration time, and moves to MRU.
        If the key doesn't exist and cache is at capacity, evicts the LRU non-expired item.
        If all items are expired, clears them all first.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        expiration_time = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.expiration_time = expiration_time
            self._move_to_head(node)
            return
            
        # New key - check capacity
        if len(self._cache) >= self.capacity:
            self._evict_lru()
            
        # Insert new node at head
        node = _Node(key, value, expiration_time)
        self._cache[key] = node
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        if key not in self._cache:
            return False
            
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        return True
        
    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are counted as expired but not removed
        until accessed. This operation is O(n) where n is the number of items.
        
        Returns:
            The number of non-expired items in the cache.
        """
        current_time = time.monotonic()
        count = 0
        node = self._head.next
        
        while node != self._tail:
            if node.expiration_time >= current_time:
                count += 1
            node = node.next
            
        return count


# Test suite
import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_get_put(mock_monotonic):
    """Test basic get and put operations."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None
    
    # Update existing
    cache.put("a", 10)
    assert cache.get("a") == 10


@patch('time.monotonic')
def test_capacity_eviction_lru_order(mock_monotonic):
    """Test that LRU items are evicted when capacity is reached."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a")
    
    # Insert 'c', should evict 'b' (LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3


@patch('time.monotonic')
def test_ttl_expiry(mock_monotonic):
    """Test that expired items return None."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    
    cache.put("a", 1)
    assert cache.get("a") == 1
    
    # Advance time past TTL
    mock_monotonic.return_value = 6.0
    assert cache.get("a") is None  # Expired
    assert cache.size() == 0


@patch('time.monotonic')
def test_custom_per_key_ttl(mock_monotonic):
    """Test custom TTL per key overrides default."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1, ttl=2.0)  # Custom TTL
    cache.put("b", 2)           # Default TTL
    
    # Advance time to 3.0
    mock_monotonic.return_value = 3.0
    
    assert cache.get("a") is None  # Expired (2.0 < 3.0)
    assert cache.get("b") == 2     # Still valid (10.0 > 3.0)


@patch('time.monotonic')
def test_delete_operation(mock_monotonic):
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False  # Already deleted
    assert cache.delete("c") is False  # Never existed
    assert cache.get("b") == 2


@patch('time.monotonic')
def test_size_with_mixed_expired_valid(mock_monotonic):
    """Test size() returns count of non-expired items only."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("a", 1, ttl=2.0)  # Will expire
    cache.put("b", 2)           # Valid
    cache.put("c", 3, ttl=5.0)  # Will expire
    
    # At time 0, all valid
    assert cache.size() == 3
    
    # Advance to time 3.0: 'a' expired, 'c' still valid
    mock_monotonic.return_value = 3.0
    assert cache.size() == 2  # 'b' and 'c' valid
    
    # Advance to time 6.0: 'c' expired, only 'b' valid
    mock_monotonic.return_value = 6.0
    assert cache.size() == 1  # Only 'b' valid
    
    # Advance to time 11.0: all expired
    mock_monotonic.return_value = 11.0
    assert cache.size() == 0