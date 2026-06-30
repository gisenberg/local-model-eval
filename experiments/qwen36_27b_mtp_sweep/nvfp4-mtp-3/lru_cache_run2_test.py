import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertions/removals."""
    __slots__ = ('key', 'value', 'exp_time', 'prev', 'next')

    def __init__(self, key: str, value: Any, exp_time: float) -> None:
        self.key = key
        self.value = value
        self.exp_time = exp_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU cache with TTL support.
    Uses a doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        self._size = 0
        # Dummy head/tail simplify boundary operations
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the dummy head."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the dummy tail (LRU)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.exp_time

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if not found or expired.
        Moves accessed item to head (LRU update). Performs lazy cleanup.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and TTL, moves to head.
        If new, adds to head. Evicts LRU if at capacity.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.exp_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self._add_to_head(node)
        self._cache[key] = node
        self._size += 1

        if self._size > self.capacity:
            lru_node = self._pop_tail()
            del self._cache[lru_node.key]
            self._size -= 1

    def delete(self, key: str) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return the current number of valid items in the cache."""
        return self._size

import pytest
from unittest.mock import patch
import time

@patch('time.monotonic')
def test_put_and_get(mock_time: time) -> None:
    """Basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time: time) -> None:
    """Entry expires after TTL and returns None on access."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time: time) -> None:
    """LRU item is evicted when capacity is exceeded."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Triggers eviction of 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time: time) -> None:
    """Updating existing key refreshes TTL and moves to head."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 4.0
    cache.put('a', 2, ttl=10.0)  # Refresh with longer TTL
    mock_time.return_value = 6.0
    assert cache.get('a') == 2  # Should not be expired
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete(mock_time: time) -> None:
    """Explicit deletion removes key and decrements size."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_size_and_capacity(mock_time: time) -> None:
    """Size tracks correctly and respects hard capacity limit."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    assert cache.size() == 0
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    cache.put('d', 4)  # Evicts 'a', size stays at 3
    assert cache.size() == 3