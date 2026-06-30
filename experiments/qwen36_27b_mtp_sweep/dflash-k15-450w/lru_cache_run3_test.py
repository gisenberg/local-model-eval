import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed on access or during eviction.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Dummy head (MRU end) and tail (LRU end)
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: _Node) -> None:
        """Insert node immediately after the dummy head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_front(node)

    def _evict_lru(self) -> None:
        """Evict the least recently used item, lazily cleaning up expired items at the tail."""
        # Lazy cleanup: remove expired nodes from the tail first
        curr = self.tail.prev
        while curr != self.head:
            if curr.expires_at <= time.monotonic():
                self._remove_node(curr)
                self.cache.pop(curr.key, None)
                curr = self.tail.prev
            else:
                break

        # If cache is still at capacity, evict the actual LRU node
        if len(self.cache) >= self.capacity:
            lru = self.tail.prev
            if lru != self.head:
                self._remove_node(lru)
                self.cache.pop(lru.key, None)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed key to MRU position. O(1) average time.
        """
        if key not in self.cache:
            return None
        node = self.cache[key]
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            del self.cache[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair. If key exists, updates value and TTL, moves to MRU.
        If cache is full, evicts LRU. O(1) average time.
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_front(node)
            return

        if len(self.cache) >= self.capacity:
            self._evict_lru()

        node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self.cache[key] = node
        self._add_to_front(node)

    def delete(self, key: Any) -> None:
        """Remove key from cache if it exists. O(1) average time."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)

    def size(self) -> int:
        """Return the number of items currently in the cache. O(1) time."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that expired entries return None and are removed."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_capacity_eviction_lru(mock_time):
    """Test LRU eviction when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    
    mock_time.return_value = 1.0
    cache.put('b', 2)
    
    mock_time.return_value = 2.0
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion of a key."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_size_tracking(mock_time):
    """Test that size() accurately reflects cache state."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    assert cache.size() == 0
    
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    
    cache.delete('a')
    assert cache.size() == 1

@patch('time.monotonic')
def test_lazy_cleanup_during_eviction(mock_time):
    """Test that expired nodes at the tail are cleaned up before evicting valid LRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 1.0
    cache.put('b', 2)
    
    mock_time.return_value = 2.0
    cache.put('c', 3)
    
    # Advance time so 'a' and 'b' expire, but 'c' remains valid
    mock_time.return_value = 6.0
    
    # Access 'c' to make it MRU, then add 'd' to trigger eviction
    cache.get('c')
    cache.put('d', 4)
    
    # 'a' and 'b' should be lazily cleaned up, 'c' should remain
    assert cache.get('a') is None
    assert cache.get('b') is None
    assert cache.get('c') == 3
    assert cache.get('d') == 4
    assert cache.size() == 2