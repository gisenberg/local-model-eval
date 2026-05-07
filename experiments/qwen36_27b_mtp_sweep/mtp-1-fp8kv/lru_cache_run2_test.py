import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class _DoublyLinkedList:
    """Doubly-linked list with O(1) add/remove/move operations."""
    def __init__(self) -> None:
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._size -= 1

    def remove_tail(self) -> Optional[_Node]:
        if self._size == 0:
            return None
        node = self.tail.prev
        self.remove(node)
        return node

    def move_to_front(self, node: _Node) -> None:
        self.remove(node)
        self.add_to_front(node)

    def __len__(self) -> int:
        return self._size


class TTLCache:
    """
    LRU Cache with TTL support.

    Uses a hash map for O(1) lookups and a doubly-linked list for O(1)
    LRU eviction and reordering. Expired entries are lazily cleaned up
    on access or insertion.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._dll = _DoublyLinkedList()

    def _get_current_time(self) -> float:
        return time.monotonic()

    def _remove(self, key: Any) -> None:
        if key in self._cache:
            node = self._cache.pop(key)
            self._dll.remove(node)

    def _is_expired(self, node: _Node, current_time: float) -> bool:
        return node.expires_at <= current_time

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed key to most recently used position.
        """
        current_time = self._get_current_time()
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node, current_time):
            self._remove(key)
            return None

        self._dll.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional TTL.
        If capacity is exceeded, the least recently used item is evicted.
        """
        current_time = self._get_current_time()
        actual_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = current_time + actual_ttl

        if key in self._cache:
            node = self._cache[key]
            if self._is_expired(node, current_time):
                self._remove(key)
            else:
                node.value = value
                node.expires_at = expires_at
                self._dll.move_to_front(node)
                return

        if len(self._cache) >= self._capacity:
            evicted = self._dll.remove_tail()
            if evicted:
                del self._cache[evicted.key]

        new_node = _Node(key, value, expires_at)
        self._cache[key] = new_node
        self._dll.add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        self._remove(key)

    def size(self) -> int:
        """Return the number of items currently in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time: Any) -> None:
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time: Any) -> None:
    """Test that expired items are cleaned up lazily on access."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time: Any) -> None:
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl_and_lru(mock_time: Any) -> None:
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 3.0
    cache.put('a', 10)  # Updates value, refreshes TTL, moves to front
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 10  # Should not be expired (expires at 8.0)
    
    # Fill cache to trigger eviction; 'b' should be evicted, not 'a'
    cache.put('c', 3)
    cache.put('d', 4)
    assert cache.get('b') is None
    assert cache.get('a') == 10

@patch('time.monotonic')
def test_delete_key(mock_time: Any) -> None:
    """Test explicit deletion of a key."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time: Any) -> None:
    """Test that per-item TTL overrides the cache default."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0