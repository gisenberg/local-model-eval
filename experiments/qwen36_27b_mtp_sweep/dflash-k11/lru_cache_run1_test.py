import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed or 
    when capacity eviction occurs.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes eliminate boundary checks for O(1) list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: _Node) -> None:
        """Inserts a node immediately after the head sentinel (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        """Moves an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _evict_lru(self) -> None:
        """Removes the least recently used node (just before tail sentinel)."""
        lru_node = self.tail.prev
        if lru_node != self.head:
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves a value by key. Returns None if key is missing or expired.
        Moves accessed key to most recently used position. O(1) avg time.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair. If key exists and is expired,
        it is lazily cleaned up before insertion. Evicts LRU if at capacity. O(1) avg time.
        """
        now = time.monotonic()
        actual_ttl = ttl if ttl is not None else self.default_ttl

        if key in self.cache:
            node = self.cache[key]
            if node.expires_at <= now:
                # Lazy cleanup: remove expired entry
                self._remove_node(node)
                del self.cache[key]
            else:
                # Update existing valid entry
                node.value = value
                node.expires_at = now + actual_ttl
                self._move_to_front(node)
                return

        # Evict if at capacity
        if len(self.cache) >= self.capacity:
            self._evict_lru()

        # Insert new node
        new_node = _Node(key, value, now + actual_ttl)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Removes a key from the cache if it exists. O(1) avg time."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Returns the current number of items in the cache. O(1) time."""
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
def test_ttl_expiration_on_get(mock_time):
    """Test that get() returns None and cleans up when TTL expires."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    mock_time.return_value = 11.0  # Past default TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that explicit ttl parameter overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=5.0)
    mock_time.return_value = 6.0  # Past custom TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_capacity_eviction_lru(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
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
def test_lazy_cleanup_on_put(mock_time):
    """Test that put() lazily cleans up expired keys before re-inserting."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    mock_time.return_value = 11.0  # Past TTL
    cache.put('a', 2)  # Should remove expired 'a', then insert new
    assert cache.get('a') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion removes key from cache and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0