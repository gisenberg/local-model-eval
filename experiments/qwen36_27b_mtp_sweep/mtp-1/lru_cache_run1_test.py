import time
from typing import Any, Optional


class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class _DoublyLinkedList:
    """Doubly-linked list with O(1) add/remove/move operations."""

    def __init__(self) -> None:
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_front(self, node: _Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove_node(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self.size -= 1

    def remove_last(self) -> Optional[_Node]:
        if self.size == 0:
            return None
        node = self.tail.prev
        self.remove_node(node)
        return node

    def move_to_front(self, node: _Node) -> None:
        self.remove_node(node)
        self.add_to_front(node)


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed
    or during eviction, avoiding background threads or periodic sweeps.
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
        self._cache: dict[Any, _Node] = {}
        self._dll = _DoublyLinkedList()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key. Returns None if key is missing or expired.
        Moves accessed key to the most recently used position.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value, or None if not found/expired.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        now = time.monotonic()
        
        # Lazy cleanup: remove if expired
        if now >= node.expires_at:
            self._remove_node(key)
            return None
            
        self._dll.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl

        if key in self._cache:
            node = self._cache[key]
            if now >= node.expires_at:
                # Lazy cleanup: treat expired existing key as new
                self._remove_node(key)
            else:
                node.value = value
                node.expires_at = now + effective_ttl
                self._dll.move_to_front(node)
                return

        # Evict LRU if at capacity
        if len(self._cache) >= self.capacity:
            evicted = self._dll.remove_last()
            if evicted:
                self._cache.pop(evicted.key, None)

        # Insert new node
        new_node = _Node(key, value, now + effective_ttl)
        self._dll.add_to_front(new_node)
        self._cache[key] = new_node

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.
        
        Args:
            key: The cache key to delete.
        """
        self._remove_node(key)

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self._cache)

    def _remove_node(self, key: Any) -> None:
        """Internal helper to remove a node from both dict and DLL."""
        if key in self._cache:
            node = self._cache.pop(key)
            self._dll.remove_node(node)

import pytest
from unittest.mock import patch


@patch('time.monotonic', side_effect=[1.0, 1.0])
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1


@patch('time.monotonic', side_effect=[1.0, 12.0])
def test_ttl_expiration(mock_time):
    """Test that entries return None after TTL expires."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)  # Expires at 11.0
    assert cache.get('a') is None  # Accessed at 12.0


@patch('time.monotonic', side_effect=[1.0, 1.0, 1.0, 1.0, 1.0])
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    
    assert cache.get('a') is None  # Evicted
    assert cache.get('b') == 2     # Still valid
    assert cache.get('c') == 3     # Most recent


@patch('time.monotonic', side_effect=[1.0, 6.0])
def test_custom_ttl_override(mock_time):
    """Test that put() accepts custom TTL overriding default."""
    cache = TTLCache(2, 100.0)
    cache.put('a', 1, ttl=5.0)  # Expires at 6.0
    assert cache.get('a') is None  # Accessed at 6.0


@patch('time.monotonic', side_effect=[1.0, 1.0])
def test_delete_operation(mock_time):
    """Test explicit deletion removes key immediately."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[1.0, 1.0, 1.0])
def test_size_tracking(mock_time):
    """Test that size() accurately reflects cache state after puts and evictions."""
    cache = TTLCache(2, 10.0)
    assert cache.size() == 0
    
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    
    cache.put('c', 3)  # Evicts 'a'
    assert cache.size() == 2