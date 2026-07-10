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
    """Doubly-linked list with sentinel nodes for O(1) insert/remove/move operations."""
    __slots__ = ('head', 'tail', 'size')

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
        self.size -= 1

    def move_to_front(self, node: _Node) -> None:
        self.remove_node(node)
        self.add_to_front(node)

    def remove_last(self) -> Optional[_Node]:
        if self.size == 0:
            return None
        last = self.tail.prev
        self.remove_node(last)
        return last


class TTLCache:
    """
    An LRU cache with Time-To-Live (TTL) support.

    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: dict[Any, _Node] = {}
        self._list = _DoublyLinkedList()
        self._valid_count = 0  # Tracks non-expired entries for O(1) size()

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        return self._now() >= node.expire_at

    def _remove_node(self, node: _Node) -> None:
        self._list.remove_node(node)
        del self._map[node.key]
        self._valid_count -= 1

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to front (LRU update). O(1) avg time.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            self._remove_node(node)
            return None
        self._list.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional TTL.
        Evicts LRU item if capacity is exceeded. O(1) avg time.
        """
        if ttl is None:
            ttl = self.default_ttl
        expire_at = self._now() + ttl

        if key in self._map:
            node = self._map[key]
            if self._is_expired(node):
                self._remove_node(node)
            else:
                node.value = value
                node.expire_at = expire_at
                self._list.move_to_front(node)
                return

        # Insert new node
        node = _Node(key, value, expire_at)
        self._map[key] = node
        self._list.add_to_front(node)
        self._valid_count += 1

        if self._valid_count > self.capacity:
            evicted = self._list.remove_last()
            if evicted is not None:
                del self._map[evicted.key]
                self._valid_count -= 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) avg time."""
        node = self._map.get(key)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """Return the number of valid (non-expired) entries in the cache. O(1) time."""
        return self._valid_count

import pytest
from unittest.mock import patch

# Note: Patch path assumes this file is run alongside ttl_cache.py.
# In a package, use @patch('your_package.ttl_cache.time.monotonic', ...)

@patch('ttl_cache.time.monotonic', side_effect=[1.0, 1.0])
def test_basic_put_get(mock_time: Any) -> None:
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1

@patch('ttl_cache.time.monotonic', side_effect=[1.0, 11.0])
def test_ttl_expiration(mock_time: Any) -> None:
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    # Time advances past TTL, lazy cleanup triggers on get
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic', side_effect=[1.0] * 6)
def test_lru_eviction(mock_time: Any) -> None:
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('ttl_cache.time.monotonic', side_effect=[1.0, 1.0, 2.0, 2.0, 2.0])
def test_update_existing_key(mock_time: Any) -> None:
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    # Update 'a': refreshes TTL, moves to front, updates value
    cache.put('a', 10)
    assert cache.get('a') == 10
    assert cache.get('b') == 2

@patch('ttl_cache.time.monotonic', side_effect=[1.0, 1.0, 1.0])
def test_delete_key(mock_time: Any) -> None:
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic', side_effect=[1.0, 1.0, 1.0, 11.0])
def test_size_with_lazy_cleanup(mock_time: Any) -> None:
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    # Time advances, 'a' expires on access
    assert cache.get('a') is None
    assert cache.size() == 2