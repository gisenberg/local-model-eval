import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expire_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy cleanup by checking expiration on access (get/put/delete).
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes simplify edge-case pointer manipulation
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the linked list and hash map."""
        node.prev.next = node.next
        node.next.prev = node.prev
        del self.cache[node.key]
        self._size -= 1

    def _add_to_tail(self, node: _Node) -> None:
        """Add a node right before the tail sentinel (MRU position)."""
        last = self.tail.prev
        last.next = node
        node.prev = last
        node.next = self.tail
        self.tail.prev = node
        self.cache[node.key] = node
        self._size += 1

    def _move_to_tail(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove(node)
        self._add_to_tail(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expire_at

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key.
        Returns None if key is missing or has expired.
        Moves accessed key to MRU position.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self._remove(node)
            return None

        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and refreshes TTL.
        If cache is full, evicts LRU item.
        """
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                self._remove(node)
            else:
                node.value = value
                node.expire_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_tail(node)
                return

        if self._size == self.capacity:
            # Evict LRU (node immediately after head sentinel)
            self._remove(self.head.next)

        expire_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = _Node(key, value, expire_at)
        self._add_to_tail(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self.cache:
            self._remove(self.cache[key])

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch

# Assuming TTLCache is in the same module or imported
# from . import TTLCache 

@patch('time.monotonic', return_value=0.0)
def test_basic_put_and_get(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic', side_effect=[0.0, 11.0])
def test_ttl_expiration_on_get(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    # Time advances past default TTL of 10.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic', return_value=0.0)
def test_lru_eviction(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Triggers eviction of 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic', return_value=0.0)
def test_update_existing_key_refreshes_ttl_and_mru(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('a', 10)  # Updates 'a', moves to MRU
    cache.put('c', 3)   # Evicts 'b' (now LRU)
    assert cache.get('b') is None
    assert cache.get('a') == 10

@patch('time.monotonic', return_value=0.0)
def test_delete_key(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic', side_effect=[0.0, 0.0, 6.0, 6.0])
def test_custom_ttl_vs_default(mock_time):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)          # Uses default TTL (10.0)
    cache.put('b', 2, ttl=5.0) # Uses custom TTL (5.0)
    
    # Time advances to 6.0
    assert cache.get('b') is None  # Expired
    assert cache.get('a') == 1     # Still valid