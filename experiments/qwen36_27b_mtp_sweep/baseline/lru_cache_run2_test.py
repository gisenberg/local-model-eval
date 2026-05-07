import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expire_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expire_at: float) -> None:
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class _DoublyLinkedList:
    """Sentinel-based doubly-linked list for O(1) insertions, deletions, and moves."""
    def __init__(self) -> None:
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._size -= 1

    def remove_last(self) -> Optional[_Node]:
        if self._size == 0:
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def move_to_front(self, node: _Node) -> None:
        self.remove(node)
        self.add_to_front(node)

    @property
    def size(self) -> int:
        return self._size


class TTLCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and hash map for O(1) average time complexity.
    Implements lazy cleanup: expired items are removed only when accessed.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._dll = _DoublyLinkedList()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expire_at

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from both the linked list and hash map."""
        self._dll.remove(node)
        del self._cache[node.key]

    def get(self, key: Any) -> Any:
        """
        Retrieve the value for a key if it exists and is not expired.
        Moves the accessed item to the most recently used position.
        Returns None if the key is missing or expired.
        """
        if key not in self._cache:
            return None
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            return None
        self._dll.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        If the key exists, updates value and TTL, moves to front.
        If capacity is exceeded, evicts the least recently used item.
        """
        if ttl is None:
            ttl = self.default_ttl
        expire_at = time.monotonic() + ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expire_at = expire_at
            self._dll.move_to_front(node)
            return

        if self._dll.size >= self.capacity:
            lru_node = self._dll.remove_last()
            if lru_node:
                del self._cache[lru_node.key]

        new_node = _Node(key, value, expire_at)
        self._cache[key] = new_node
        self._dll.add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            self._remove_node(self._cache[key])

    def size(self) -> int:
        """
        Return the current number of items in the cache.
        Note: Due to lazy cleanup, this may temporarily include expired items 
        until they are accessed via get/put/delete.
        """
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    mock_time.return_value = 15.0  # Advance time past TTL
    assert cache.get('a') is None  # Lazy cleanup triggers on access
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    mock_time.return_value = 5.0
    cache.put('a', 2, ttl=20.0)  # Refresh value and TTL
    
    mock_time.return_value = 10.0
    assert cache.get('a') == 2
    assert cache.size() == 1  # No eviction occurred

@patch('time.monotonic')
def test_delete_key(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_vs_default(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)           # Uses default TTL (10s)
    cache.put('b', 2, ttl=20.0) # Uses custom TTL (20s)
    
    mock_time.return_value = 5.0
    assert cache.get('a') == 1
    assert cache.get('b') == 2
    
    mock_time.return_value = 15.0
    assert cache.get('a') is None  # Expired
    assert cache.get('b') == 2     # Still valid