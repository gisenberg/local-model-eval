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
    LRU cache with TTL support using a custom doubly-linked list and hash map.
    Guarantees O(1) average time complexity for get, put, delete, and size.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds for new entries.
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
        self._size = 0

    def _add_to_front(self, node: _Node) -> None:
        """Insert node immediately after head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to the front (mark as most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _evict(self) -> None:
        """Remove the least recently used node (just before tail)."""
        lru = self.tail.prev
        self._remove_node(lru)
        del self.cache[lru.key]
        self._size -= 1

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Performs lazy cleanup if expired.
        Returns None if key is missing or expired.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None
            
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. Performs lazy cleanup if expired.
        Evicts LRU entry if capacity is exceeded.
        """
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                self._remove_node(node)
                del self.cache[key]
                self._size -= 1
            else:
                self._move_to_front(node)
                node.value = value
                node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                return

        if self._size >= self.capacity:
            self._evict()

        new_node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self._size += 1

    def delete(self, key: Any) -> bool:
        """
        Remove a key from the cache.
        Returns True if the key was present and removed, False otherwise.
        """
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch
import time


@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1


@patch('time.monotonic')
def test_expiration_on_get(mock_time):
    """Test lazy cleanup when retrieving an expired key."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_expiration_on_put(mock_time):
    """Test lazy cleanup when overwriting an expired key."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    cache.put('a', 2)  # Should remove expired entry and insert fresh one
    assert cache.get('a') == 2
    assert cache.size() == 1


@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test LRU eviction when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)  # Long TTL to avoid expiration interference
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2


@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion and size tracking."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    assert cache.delete('a') is True
    assert cache.size() == 1
    assert cache.get('a') is None
    assert cache.delete('nonexistent') is False


@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that explicit TTL parameter overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)  # Custom TTL of 2 seconds
    
    mock_time.return_value = 5.0  # Advance time past custom TTL
    assert cache.get('a') is None
    assert cache.size() == 0