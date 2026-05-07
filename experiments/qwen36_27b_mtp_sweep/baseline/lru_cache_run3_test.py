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


class _DoublyLinkedList:
    """Doubly-linked list with O(1) insertion, removal, and tail eviction."""

    def __init__(self) -> None:
        # Sentinel nodes to simplify edge cases
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        """Insert node right after head (MRU position). O(1)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: _Node) -> None:
        """Remove node from list. O(1)"""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._size -= 1

    def remove_tail(self) -> Optional[_Node]:
        """Remove and return node before tail (LRU position). O(1)"""
        if self._size == 0:
            return None
        node = self.tail.prev
        self.remove(node)
        return node

    def size(self) -> int:
        """Return number of nodes. O(1)"""
        return self._size


class TTLCache:
    """LRU cache with TTL support using a hash map and doubly-linked list.
    
    All public operations run in O(1) average time. Expired entries are 
    cleaned up lazily upon access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize cache.
        
        Args:
            capacity: Maximum number of items allowed in the cache.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        self.dll = _DoublyLinkedList()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from both the linked list and hash map."""
        self.dll.remove(node)
        del self.cache[node.key]

    def get(self, key: Any) -> Any:
        """Retrieve value by key. Returns None if missing or expired.
        
        Performs lazy cleanup: expired items are evicted on access.
        Moves accessed item to MRU position. O(1) average.
        """
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            return None
            
        # Move to front (most recently used)
        self.dll.remove(node)
        self.dll.add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.
        
        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL in seconds. Uses default_ttl if None.
            
        O(1) average. Evicts LRU item if capacity is reached.
        """
        actual_ttl = ttl if ttl is not None else self.default_ttl
        expiration = time.monotonic() + actual_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = expiration
            self.dll.remove(node)
            self.dll.add_to_front(node)
            return

        # Evict LRU if at capacity
        if self.dll.size() >= self.capacity:
            lru_node = self.dll.remove_tail()
            if lru_node:
                del self.cache[lru_node.key]

        # Insert new node
        new_node = _Node(key, value, expiration)
        self.cache[key] = new_node
        self.dll.add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) average."""
        if key in self.cache:
            self._remove_node(self.cache[key])

    def size(self) -> int:
        """Return current number of items in the cache. O(1)"""
        return self.dll.size()

import pytest
from unittest.mock import patch

def test_basic_put_and_get():
    """Test standard insertion and retrieval."""
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration_lazy_cleanup():
    """Test that expired items are cleaned up lazily on get()."""
    # put @ 100.0, get @ 100.0 (valid), get @ 115.0 (expired)
    with patch('time.monotonic', side_effect=[100.0, 100.0, 115.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)  # expires at 110.0
        assert cache.get('a') == 1
        assert cache.get('a') is None  # Lazy cleanup triggers here
        assert cache.size() == 0

def test_custom_ttl_overrides_default():
    """Test that per-item TTL overrides the default."""
    # put @ 100.0, put @ 100.0, get @ 105.0, get @ 105.0
    with patch('time.monotonic', side_effect=[100.0, 100.0, 105.0, 105.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1, ttl=5.0)  # expires at 105.0
        cache.put('b', 2)           # expires at 110.0
        assert cache.get('a') is None  # Expired
        assert cache.get('b') == 2     # Still valid

def test_capacity_eviction_lru():
    """Test that LRU item is evicted when capacity is exceeded."""
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_delete_operation():
    """Test explicit deletion of a key."""
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.size() == 1

def test_update_refreshes_mru_order():
    """Test that updating an existing key moves it to MRU position."""
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10)  # Updates 'a', moves to front
        cache.put('c', 3)   # Should evict 'b' (now LRU)
        assert cache.get('b') is None
        assert cache.get('a') == 10
        assert cache.get('c') == 3