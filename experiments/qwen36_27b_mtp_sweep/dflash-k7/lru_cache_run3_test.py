import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertions and deletions."""
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

    Uses a hash map for O(1) lookups and a doubly-linked list with sentinel nodes
    to maintain access order (MRU at tail, LRU at head). Implements lazy cleanup:
    expired entries are only removed when accessed via `get`.

    Args:
        capacity: Maximum number of items the cache can hold.
        default_ttl: Default time-to-live in seconds for new entries.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def _add_to_tail(self, node: _Node) -> None:
        """Insert node right before the tail sentinel (MRU position)."""
        prev = self._tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._tail
        self._tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list without deleting from cache."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_tail(self, node: _Node) -> None:
        """Move existing node to the tail (MRU position)."""
        self._remove_node(node)
        self._add_to_tail(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiry

    def _remove_from_cache(self, node: _Node) -> None:
        """Remove node from both linked list and hash map."""
        self._remove_node(node)
        del self._cache[node.key]
        self._size -= 1

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to MRU position. Lazy cleanup on access.
        """
        node = self._cache.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            self._remove_from_cache(node)
            return None
        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair. If key exists, updates value and TTL,
        moves to MRU. If capacity exceeded, evicts LRU item.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_tail(node)
            return

        if self._size >= self.capacity:
            # Evict LRU (head.next)
            lru_node = self._head.next
            self._remove_from_cache(lru_node)

        new_node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self._cache[key] = new_node
        self._add_to_tail(new_node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """Remove key from cache if it exists."""
        node = self._cache.get(key)
        if node is not None:
            self._remove_from_cache(node)

    def size(self) -> int:
        """Return current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch
import time

@patch('time.monotonic')
def test_basic_put_get(mock_monotonic: time.monotonic) -> None:
    """Test basic insertion and retrieval within TTL window."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_get_expired_key_returns_none(mock_monotonic: time.monotonic) -> None:
    """Test lazy cleanup: expired key returns None and is removed from cache."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_monotonic.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction_on_capacity_exceeded(mock_monotonic: time.monotonic) -> None:
    """Test that least recently used item is evicted when capacity is full."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_refreshes_ttl_and_mru_position(mock_monotonic: time.monotonic) -> None:
    """Test that updating an existing key refreshes its TTL and moves it to MRU."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_monotonic.return_value = 3.0
    cache.put('a', 10)  # Update 'a', refreshes TTL to 8.0, moves to MRU
    
    mock_monotonic.return_value = 6.0
    # 'b' expires at 5.0, 'a' expires at 8.0
    assert cache.get('b') is None
    assert cache.get('a') == 10
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_monotonic: time.monotonic) -> None:
    """Test that explicit ttl parameter overrides default_ttl."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_monotonic.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_delete_removes_key_and_updates_size(mock_monotonic: time.monotonic) -> None:
    """Test explicit deletion of a key."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.get('b') == 2