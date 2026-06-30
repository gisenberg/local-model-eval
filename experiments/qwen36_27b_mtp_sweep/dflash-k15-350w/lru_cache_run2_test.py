import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) cache operations."""
    __slots__ = ('key', 'value', 'exp', 'prev', 'next')

    def __init__(self, key: Any, value: Any, exp: float) -> None:
        self.key = key
        self.value = value
        self.exp = exp
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) lookups and a doubly-linked list for O(1) 
    insertion/removal and LRU tracking. Expired entries are lazily cleaned 
    up on access.
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
            raise ValueError("Default TTL must be a positive number.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes to simplify edge-case handling in the linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_tail(self, node: _Node) -> None:
        """Insert node immediately before the tail sentinel (MRU position)."""
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_tail(self, node: _Node) -> None:
        """Move existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_tail(node)

    def _evict_lru(self) -> None:
        """Remove the least recently used item (adjacent to head sentinel)."""
        lru_node = self.head.next
        if lru_node is not self.tail:
            self._remove_node(lru_node)
            self.cache.pop(lru_node.key, None)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node's TTL has elapsed."""
        return time.monotonic() >= node.exp

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key missing or expired.
        Moves accessed key to MRU position.
        """
        node = self.cache.get(key)
        if node is None:
            return None
            
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None
            
        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.exp = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_tail(node)
            return

        if len(self.cache) >= self.capacity:
            self._evict_lru()

        exp = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = _Node(key, value, exp)
        self.cache[key] = new_node
        self._add_to_tail(new_node)

    def delete(self, key: Any) -> bool:
        """
        Remove key from cache.
        
        Returns:
            True if key was found and removed, False otherwise.
        """
        if key not in self.cache:
            return False
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """Return current number of entries in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic', return_value=0.0)
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired entries are lazily removed on access."""
    mock_time.side_effect = [0.0, 11.0]  # put at t=0, get at t=11
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic', return_value=0.0)
def test_lru_eviction_on_capacity(mock_time):
    """Test that LRU item is evicted when capacity is exceeded."""
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl_and_mru(mock_time):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_time.side_effect = [0.0, 5.0, 5.0]  # put, update, get
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('a', 2, ttl=20.0)  # Updates value, refreshes TTL, moves to MRU
    assert cache.get('a') == 2

@patch('time.monotonic', return_value=0.0)
def test_delete_key(mock_time):
    """Test explicit deletion and idempotency."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.delete('a') is True
    assert cache.get('a') is None
    assert cache.delete('a') is False  # Deleting non-existent key returns False

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that custom TTL takes precedence over default_ttl."""
    mock_time.side_effect = [0.0, 0.0, 5.0, 5.0]  # put1, put2, get1, get2
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    cache.put('b', 2)  # Uses default 10.0
    assert cache.get('a') is None  # Expired at t=5
    assert cache.get('b') == 2     # Still valid