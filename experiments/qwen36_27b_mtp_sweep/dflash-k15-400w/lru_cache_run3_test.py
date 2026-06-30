import time
from typing import Any, Optional


class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) lookups and a doubly-linked list for O(1) 
    insertion/deletion and LRU ordering. Expired entries are lazily cleaned 
    up upon access.
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
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        
        # Sentinel nodes to simplify edge-case handling
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the node before the tail sentinel (LRU)."""
        res = self._tail.prev
        if res is self._head:
            return None
        self._remove_node(res)
        return res

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has passed its TTL."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key. Returns None if missing or expired.
        Moves accessed item to most recently used position.
        
        Args:
            key: Cache key to look up.
            
        Returns:
            Cached value, or None if not found/expired.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        now = time.monotonic()

        if key in self._cache:
            node = self._cache[key]
            # Lazy cleanup: treat expired entries as new insertions
            if self._is_expired(node):
                self.delete(key)
            else:
                node.value = value
                node.expiry = now + effective_ttl
                self._move_to_head(node)
                return

        # Evict LRU if at capacity
        if len(self._cache) >= self.capacity:
            tail = self._pop_tail()
            if tail:
                del self._cache[tail.key]

        # Insert new node
        new_node = _Node(key, value, now + effective_ttl)
        self._add_to_head(new_node)
        self._cache[key] = new_node

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.
        
        Args:
            key: Cache key to delete.
        """
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """
        Return the current number of items in the cache.
        Note: May include expired items until they are lazily cleaned up.
        
        Returns:
            Number of cached entries.
        """
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('ttl_cache.time.monotonic')
def test_basic_put_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that entries return None after TTL expires."""
    mock_time.side_effect = [0.0, 0.0, 11.0]  # put, put, get_after_expiry
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') is None
    assert cache.size() == 0  # Lazy cleanup removed it

@patch('ttl_cache.time.monotonic')
def test_capacity_eviction_lru(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that custom TTL overrides default TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=5.0)
    
    mock_time.return_value = 4.0
    assert cache.get('a') == 1
    
    mock_time.return_value = 6.0
    assert cache.get('a') is None

@patch('ttl_cache.time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion of keys."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1
    
    # Deleting non-existent key should not raise
    cache.delete('z')

@patch('ttl_cache.time.monotonic')
def test_size_and_lazy_cleanup_behavior(mock_time):
    """Test that size() reflects raw dict length and cleanup is lazy."""
    mock_time.side_effect = [0.0, 0.0, 0.0, 11.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    assert cache.size() == 2
    
    # Time passes, 'a' expires, but size() shouldn't trigger cleanup
    assert cache.size() == 2
    
    # Access triggers lazy cleanup
    assert cache.get('a') is None
    assert cache.size() == 1