import time
from typing import TypeVar, Generic, Optional

K = TypeVar('K')
V = TypeVar('V')


class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: K, value: V, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache(Generic[K, V]):
    """LRU Cache with TTL support using a doubly-linked list and hash map.
    
    Provides O(1) average time complexity for get, put, and delete operations.
    Expired entries are cleaned up lazily during access or capacity pressure.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._size = 0
        self._cache: dict[K, _Node] = {}

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiry

    def _remove(self, node: _Node) -> None:
        """Unlink a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove(node)
        self._add_to_head(node)

    def _cleanup_expired_from_tail(self) -> None:
        """Lazily remove expired nodes from the LRU end until a valid node is found."""
        while self._size > 0:
            node = self._tail.prev
            if self._is_expired(node):
                self._remove(node)
                self._cache.pop(node.key, None)
                self._size -= 1
            else:
                break

    def _evict_lru(self) -> None:
        """Evict the least recently used valid node."""
        node = self._tail.prev
        self._remove(node)
        self._cache.pop(node.key, None)
        self._size -= 1

    def get(self, key: K) -> Optional[V]:
        """Retrieve value by key, or None if missing/expired.
        
        Moves accessed key to MRU position and removes it if expired.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        if self._is_expired(node):
            self._remove(node)
            self._cache.pop(key, None)
            self._size -= 1
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair with optional TTL.
        
        If the key exists, updates value and refreshes TTL.
        If capacity is reached, performs lazy cleanup then evicts LRU if needed.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        if self._size == self.capacity:
            self._cleanup_expired_from_tail()
            if self._size == self.capacity:
                self._evict_lru()

        node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self._cache[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: K) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove(node)
            self._cache.pop(key, None)
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch


@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)


def test_basic_put_get(cache):
    """Test standard insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1


def test_ttl_expiration_on_get(cache):
    """Test that expired entries return None and are removed."""
    with patch('time.monotonic', side_effect=[0.0, 15.0]):
        cache.put('a', 1)
        assert cache.get('a') is None
        assert cache.size() == 0


def test_lru_eviction(cache):
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        cache.put('d', 4)  # Should evict 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.size() == 3


def test_update_refreshes_ttl_and_lru(cache):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    times = [0.0, 0.0, 5.0, 5.0, 12.0, 12.0, 12.0]
    with patch('time.monotonic', side_effect=times):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10, ttl=20.0)  # Refresh 'a', expires at 25
        cache.put('c', 3)            # Expires at 15
        cache.put('d', 4)            # Expires at 22, evicts 'b' (expired at 10)
        assert cache.get('a') == 10
        assert cache.get('b') is None
        assert cache.size() == 3


def test_delete(cache):
    """Test explicit deletion reduces size and removes key."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 1


def test_lazy_cleanup_on_put(cache):
    """Test lazy cleanup removes multiple expired entries during insertion."""
    # put a, b, c, d(cleanup+insert), get d, get a
    times = [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    with patch('time.monotonic', side_effect=times):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        cache.put('d', 4)  # Triggers cleanup of a, b, c
        assert cache.size() == 1
        assert cache.get('d') == 4
        assert cache.get('a') is None