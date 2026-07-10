import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class _DoublyLinkedList:
    """Sentinel-based doubly-linked list for O(1) insertions/removals."""
    def __init__(self) -> None:
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node
        self._size += 1

    def remove_node(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._size -= 1

    def remove_last(self) -> Optional[_Node]:
        if self._size == 0:
            return None
        node = self._tail.prev
        self.remove_node(node)
        return node

    def move_to_front(self, node: _Node) -> None:
        self.remove_node(node)
        self.add_to_front(node)

    def __len__(self) -> int:
        return self._size


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are only removed when accessed.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: dict[Any, _Node] = {}
        self._dll = _DoublyLinkedList()

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        return self._now() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key.
        Returns None if key is missing or has expired.
        Moves accessed key to most-recently-used position.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            self._remove(node)
            return None
        self._dll.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If `ttl` is None, uses `default_ttl`.
        Evicts least-recently-used entry if capacity is exceeded.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = self._now() + effective_ttl

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._dll.move_to_front(node)
        else:
            if len(self._map) >= self.capacity:
                self._evict()
            node = _Node(key, value, expiry)
            self._map[key] = node
            self._dll.add_to_front(node)

    def delete(self, key: Any) -> None:
        """Remove key from cache if present."""
        node = self._map.get(key)
        if node is not None:
            self._remove(node)

    def size(self) -> int:
        """
        Return number of entries in the cache.
        Note: Includes expired entries pending lazy cleanup.
        """
        return len(self._map)

    def _remove(self, node: _Node) -> None:
        self._map.pop(node.key, None)
        self._dll.remove_node(node)

    def _evict(self) -> None:
        if len(self._dll) > 0:
            node = self._dll.remove_last()
            self._map.pop(node.key, None)

import pytest
from unittest.mock import patch

# Assuming TTLCache is in the same module or imported as:
@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiry_lazy_cleanup(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 6.0
    # Lazy cleanup: expired item remains in map until accessed
    assert cache.size() == 1
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.get('a')  # 'a' becomes most recent
    cache.put('c', 3)  # capacity full, evicts least recent ('b')
    assert cache.get('b') is None
    assert cache.get('a') == 1
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1, ttl=2.0)
    mock_time.return_value = 3.0
    assert cache.get('a') is None

@patch('time.monotonic')
def test_delete_removes_entry(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_update_existing_key_resets_ttl(mock_time):
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_time.return_value = 3.0
    cache.put('a', 2, ttl=10.0)  # updates value and resets TTL
    mock_time.return_value = 8.0
    assert cache.get('a') == 2