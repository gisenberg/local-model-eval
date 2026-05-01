import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
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

    Uses a custom doubly-linked list and a hash map to guarantee O(1) average
    time complexity for get, put, and delete operations. Implements lazy
    expiration checking (items are evicted on access rather than via background
    threads or periodic scans).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes simplify edge-case handling in linked list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head sentinel (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _remove_lru(self) -> Optional[_Node]:
        """Remove and return the least recently used node (before tail)."""
        if self._size == 0:
            return None
        lru = self.tail.prev
        self._remove_node(lru)
        self._size -= 1
        return lru

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if a node has expired based on provided monotonic time."""
        return now >= node.expiry

    def get(self, key: Any) -> Any:
        """
        Retrieve a value by key. Returns None if key is missing or expired.
        Moves accessed key to most recently used position. O(1) avg time.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        now = time.monotonic()
        
        if self._is_expired(node, now):
            self.delete(key)
            return None
            
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. O(1) avg time.
        Evicts LRU item if capacity is exceeded. Handles lazy expiration.
        """
        if ttl is None:
            ttl = self.default_ttl
            
        now = time.monotonic()
        expiry = now + ttl

        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node, now):
                self.delete(key)
            else:
                node.value = value
                node.expiry = expiry
                self._move_to_front(node)
                return

        # Evict LRU if at capacity
        if self._size >= self.capacity:
            lru = self._remove_lru()
            if lru:
                del self.cache[lru.key]

        # Insert new node
        new_node = _Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) avg time."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch

# 1. Basic put and get
@patch('time.monotonic', side_effect=[1.0, 1.0])
def test_basic_put_get(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1

# 2. TTL expiration on get (lazy cleanup)
@patch('time.monotonic', side_effect=[1.0, 11.0])
def test_ttl_expiration(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)  # expires at 11.0
    assert cache.get('a') is None  # accessed at 11.0 -> expired

# 3. LRU eviction order
@patch('time.monotonic', side_effect=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
def test_lru_eviction(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.get('a')      # 'a' becomes MRU
    cache.put('c', 3)   # 'b' is LRU -> evicted
    assert cache.get('b') is None
    assert cache.get('a') == 1
    assert cache.get('c') == 3

# 4. Custom TTL override
@patch('time.monotonic', side_effect=[1.0, 5.0])
def test_custom_ttl(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=3.0)  # expires at 4.0
    assert cache.get('a') is None  # accessed at 5.0 -> expired

# 5. Delete operation
@patch('time.monotonic', side_effect=[1.0, 1.0])
def test_delete(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0

# 6. Size tracking across operations
@patch('time.monotonic', side_effect=[1.0, 1.0, 1.0, 1.0])
def test_size_tracking(mock_time):
    cache = TTLCache(2, 10.0)
    assert cache.size() == 0
    cache.put('a', 1)
    assert cache.size() == 1
    cache.put('b', 2)
    assert cache.size() == 2
    cache.put('c', 3)  # evicts 'a', size stays at capacity
    assert cache.size() == 2