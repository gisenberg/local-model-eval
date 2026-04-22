import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed
    via `get`, `put`, or `delete`, avoiding background threads or periodic scans.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Dummy head/tail nodes simplify boundary operations
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node right after the dummy head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the head (marks as most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> _Node:
        """Removes and returns the node before the dummy tail (least recently used)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node's TTL has elapsed using monotonic time."""
        return time.monotonic() >= node.expiration

    def get(self, key: Any) -> Any:
        """
        Retrieves the value for `key` if it exists and is not expired.
        Moves the accessed node to the head (LRU update).
        Returns None if the key is missing or expired.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        If `ttl` is None, uses `default_ttl`.
        Evicts the LRU item if capacity is exceeded.
        """
        if key in self._cache:
            node = self._cache[key]
            if self._is_expired(node):
                # Lazy cleanup: remove expired entry before re-inserting
                self._remove_node(node)
                del self._cache[key]
            else:
                node.value = value
                node.expiration = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
                self._move_to_head(node)
                return

        # If capacity is full, evict LRU
        if len(self._cache) >= self._capacity:
            lru_node = self._remove_tail()
            del self._cache[lru_node.key]

        actual_ttl = ttl if ttl is not None else self._default_ttl
        new_node = _Node(key, value, time.monotonic() + actual_ttl)
        self._cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """Removes the key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """Returns the current number of valid items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Basic insertion and retrieval."""
    mock_time.return_value = 100.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Key should be lazily removed when accessed after TTL expires."""
    mock_time.side_effect = [100.0, 111.0]  # put @100, get @111 (11s later)
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """LRU item should be evicted when capacity is exceeded."""
    mock_time.return_value = 100.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    """Updating an existing key should refresh TTL and move to head."""
    mock_time.side_effect = [100.0, 105.0, 110.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('a', 2)  # Updates value, refreshes expiration
    assert cache.get('a') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Explicit deletion should remove key and update size."""
    mock_time.return_value = 100.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Explicit ttl parameter should override default_ttl."""
    mock_time.side_effect = [100.0, 100.0, 105.0, 105.0]
    cache = TTLCache(2, 10.0)  # default 10s
    cache.put('short', 1, ttl=5.0)
    cache.put('long', 2)       # uses default 10s
    assert cache.get('short') is None  # expired at 105
    assert cache.get('long') == 2      # alive until 110