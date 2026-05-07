import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) reordering."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    An LRU cache with Time-To-Live (TTL) support.

    Uses a hash map for O(1) lookups and a custom doubly-linked list for O(1)
    reordering and eviction. Expired entries are lazily cleaned up upon access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes for O(1) list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the node before the tail sentinel (LRU position)."""
        if self.tail.prev == self.head:
            return None
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiration

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        node = self.cache.get(key)
        if node is None:
            return None

        # Lazy cleanup: remove expired entries on access
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert/update.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiration = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        if len(self.cache) >= self.capacity:
            evicted = self._pop_tail()
            if evicted:
                del self.cache[evicted.key]

        new_node = _Node(key, value, time.monotonic() + (ttl if ttl is not None else self.default_ttl))
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to remove.
        """
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=2, default_ttl=10.0)

def test_basic_put_and_get(cache):
    """Verify basic insertion and retrieval."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.get('a') == 1
        assert cache.size() == 2

def test_ttl_expiration(cache):
    """Verify entries return None after TTL expires."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 12.0]):
        cache.put('a', 1)          # Expires at 1.0 + 10.0 = 11.0
        assert cache.get('a') == 1 # t=2.0, valid
        assert cache.get('a') is None  # t=12.0, expired & lazily cleaned
        assert cache.size() == 0

def test_lru_eviction(cache):
    """Verify least recently used item is evicted when capacity is reached."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a' (LRU)
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_custom_ttl_overrides_default(cache):
    """Verify custom TTL parameter works independently."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0, 4.0]):
        cache.put('a', 1, ttl=5.0)  # Expires at 6.0
        cache.put('b', 2, ttl=1.0)  # Expires at 3.0
        assert cache.get('b') is None  # t=3.0, expired
        assert cache.get('a') == 1     # t=4.0, still valid
        assert cache.size() == 1

def test_delete_removes_entry(cache):
    """Verify explicit deletion works and updates size."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 1

def test_update_refreshes_ttl_and_mru(cache):
    """Verify updating an existing key refreshes TTL and moves it to MRU."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10)  # Updates 'a', moves to MRU
        cache.put('c', 3)   # Evicts 'b' (now LRU)
        assert cache.get('b') is None
        assert cache.get('a') == 10
        assert cache.size() == 2