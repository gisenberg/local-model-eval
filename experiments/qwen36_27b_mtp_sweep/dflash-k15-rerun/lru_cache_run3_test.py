import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed
    or when they block new insertions.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        
        # Dummy head (most recently used) and tail (least recently used)
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the dummy head."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the dummy tail (least recently used)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Any:
        """
        Retrieve the value associated with `key`.
        Returns None if the key is missing or has expired.
        Moves accessed item to the most recently used position.
        """
        now = time.monotonic()
        node = self._cache.get(key)
        if node is None:
            return None
        if now >= node.expiration:
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If `ttl` is None, uses `default_ttl`.
        Evicts least recently used item if capacity is exceeded.
        """
        now = time.monotonic()
        if ttl is None:
            ttl = self._default_ttl
        expiration = now + ttl

        node = self._cache.get(key)
        if node is not None:
            # Lazy cleanup: remove if expired
            if now >= node.expiration:
                self._remove_node(node)
                del self._cache[key]
                self._size -= 1
                node = None

        if node is None:
            node = _Node(key, value, expiration)
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1

            if self._size > self._capacity:
                tail = self._pop_tail()
                del self._cache[tail.key]
                self._size -= 1
        else:
            # Update existing valid node
            node.value = value
            node.expiration = expiration
            self._move_to_head(node)

    def delete(self, key: Any) -> None:
        """Remove the key from the cache if it exists."""
        node = self._cache.get(key)
        if node is not None:
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

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
    """Test lazy cleanup when accessing an expired key."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0  # Exceeds 5.0 TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_ttl_expiration_on_put(mock_time):
    """Test lazy cleanup when updating an expired key."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0  # Exceeds 5.0 TTL
    cache.put('a', 2)
    assert cache.get('a') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test LRU eviction when capacity is exceeded."""
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
def test_delete_operation(mock_time):
    """Test explicit key deletion."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test per-key TTL overriding the default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    mock_time.return_value = 3.0  # Exceeds custom 2.0 TTL
    assert cache.get('a') is None
    assert cache.size() == 0