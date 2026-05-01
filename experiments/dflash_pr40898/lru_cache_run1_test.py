import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed
    or during eviction.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for O(1) list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move existing node to the head (mark as recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the tail node (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value, or None if key is missing or expired.
        """
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup on access
            self._remove_node(node)
            del self.cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl if None.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Treat expired update as a new insertion
                self._remove_node(node)
                del self.cache[key]
            else:
                # Update existing valid entry
                node.value = value
                node.expires_at = time.monotonic() + effective_ttl
                self._move_to_head(node)
                return

        # Evict LRU if at capacity
        if len(self.cache) >= self.capacity:
            evicted = self._pop_tail()
            del self.cache[evicted.key]

        # Insert new entry
        new_node = _Node(key, value, time.monotonic() + effective_ttl)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.
        
        Args:
            key: The cache key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_lazy_ttl_expiration(mock_time):
    """Test that expired entries are cleaned up lazily on access."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that per-item TTL takes precedence over default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL
    assert cache.get('a') is None

@patch('time.monotonic')
def test_lru_eviction_on_capacity(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_delete_removes_entry(mock_time):
    """Test explicit deletion and size tracking."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.get('b') == 2

@patch('time.monotonic')
def test_update_existing_key_resets_ttl(mock_time):
    """Test that updating an existing key refreshes its TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 2.0
    cache.put('a', 2)  # Updates value, resets TTL to 2.0 + 5.0 = 7.0
    
    assert cache.get('a') == 2
    mock_time.return_value = 8.0  # Past refreshed TTL
    assert cache.get('a') is None