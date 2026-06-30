import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are evicted on-demand rather than via background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl < 0:
            raise ValueError("TTL must be non-negative.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        self._size: int = 0

        # Sentinel nodes simplify boundary checks in the doubly-linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node immediately after the head sentinel (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the MRU position."""
        self._remove(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Removes and returns the node before the tail sentinel (LRU position)."""
        node = self.tail.prev
        self._remove(node)
        return node

    def _cleanup_expired(self, now: float) -> None:
        """Lazily removes expired nodes from the tail until a valid node is found."""
        while self.head.next != self.tail:
            node = self.tail.prev
            if now >= node.expires_at:
                self._remove(node)
                del self.cache[node.key]
                self._size -= 1
            else:
                break

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves a value by key. Returns None if missing or expired.
        Moves accessed node to head if valid. O(1) avg time.
        """
        now = time.monotonic()
        if key not in self.cache:
            return None

        node = self.cache[key]
        if now >= node.expires_at:
            self._remove(node)
            del self.cache[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair. Evicts LRU if at capacity.
        Performs lazy cleanup before insertion. O(1) avg time.
        """
        now = time.monotonic()
        if ttl is None:
            ttl = self.default_ttl
        expires_at = now + ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
        else:
            self._cleanup_expired(now)

            if self._size >= self.capacity:
                lru_node = self._pop_tail()
                del self.cache[lru_node.key]
                self._size -= 1

            new_node = _Node(key, value, expires_at)
            self.cache[key] = new_node
            self._add_to_head(new_node)
            self._size += 1

    def delete(self, key: Any) -> None:
        """Removes a key from the cache if it exists. O(1) avg time."""
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]
            self._size -= 1

    def size(self) -> int:
        """Returns the current number of valid entries in the cache."""
        return self._size

import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Basic insertion and retrieval works correctly."""
    mock_time.side_effect = [0.0, 0.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1


@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Expired entries return None and are removed from cache."""
    mock_time.side_effect = [0.0, 11.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """LRU entry is evicted when capacity is exceeded."""
    mock_time.side_effect = [0.0] * 6
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2


@patch('time.monotonic')
def test_update_refreshes_ttl(mock_time):
    """Updating an existing key refreshes its TTL and moves it to MRU."""
    mock_time.side_effect = [0.0, 5.0, 14.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)      # Expires at 10
    cache.put('a', 2)      # Expires at 15
    assert cache.get('a') == 2  # At t=14, still valid
    assert cache.size() == 1


@patch('time.monotonic')
def test_delete(mock_time):
    """Explicit deletion removes key and decrements size."""
    mock_time.side_effect = [0.0, 0.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_lazy_cleanup(mock_time):
    """Expired nodes are lazily cleaned from the tail before insertion."""
    mock_time.side_effect = [0.0, 0.0, 11.0, 11.0]
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)  # Expires at 10
    cache.put('b', 2)  # Expires at 10
    cache.put('c', 3)  # Triggers cleanup: removes 'a' & 'b', then inserts 'c'
    assert cache.size() == 1
    assert cache.get('c') == 3