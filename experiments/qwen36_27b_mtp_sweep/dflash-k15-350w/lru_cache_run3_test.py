import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertions/deletions."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) lookups and a custom doubly-linked list for O(1)
    insertion/deletion and LRU tracking. Implements lazy cleanup: expired entries
    are only removed when accessed or when they block new insertions.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Sentinel nodes eliminate boundary checks in list operations
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node right after the head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Removes a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Removes and returns the LRU node (just before the tail sentinel)."""
        lru = self._tail.prev
        self._remove_node(lru)
        return lru

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node has exceeded its TTL using monotonic time."""
        return time.monotonic() >= node.expires_at

    def _remove_from_cache(self, node: _Node) -> None:
        """Removes a node from both the linked list and the hash map."""
        self._remove_node(node)
        del self._cache[node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves the value for a key if it exists and hasn't expired.
        Moves the accessed item to the MRU position. Returns None otherwise.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove_from_cache(node)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair with an optional custom TTL.
        If the cache is at capacity, the LRU item is evicted.
        """
        if key in self._cache:
            node = self._cache[key]
            if self._is_expired(node):
                self._remove_from_cache(node)
            else:
                node.value = value
                node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_head(node)
                return

        if len(self._cache) == self.capacity:
            lru = self._pop_tail()
            del self._cache[lru.key]

        actual_ttl = ttl if ttl is not None else self.default_ttl
        new_node = _Node(key, value, time.monotonic() + actual_ttl)
        self._add_to_head(new_node)
        self._cache[key] = new_node

    def delete(self, key: Any) -> None:
        """Removes a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_from_cache(node)

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

# Assuming TTLCache is imported or defined in the same namespace
# from . import TTLCache  # Adjust import path as needed

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup: expired item returns None and is removed."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that per-item TTL overrides the cache default."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL, but within default
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction_on_capacity(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_delete_removes_entry(mock_time):
    """Test explicit deletion removes key from both structures."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_update_existing_key_resets_ttl(mock_time):
    """Test that updating a key refreshes its expiration time."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    mock_time.return_value = 2.0
    cache.put('a', 2)  # Update value and reset TTL (expires at 12.0)
    assert cache.get('a') == 2
    
    mock_time.return_value = 11.0  # Within new TTL window
    assert cache.get('a') == 2
    
    mock_time.return_value = 13.0  # Past new TTL window
    assert cache.get('a') is None