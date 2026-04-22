import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a hash map for O(1) lookups and a doubly-linked list for O(1)
    reordering/eviction. Implements lazy cleanup by checking expiration
    on access rather than using background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size: int = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node immediately after the head (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Removes a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Removes and returns the LRU node (just before the tail)."""
        if self._size == 0:
            return None
        lru = self._tail.prev
        self._remove_node(lru)
        return lru

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key.

        Returns None if the key is missing or expired. Moves accessed
        item to MRU position if valid.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl

        current_time = time.monotonic()
        expires_at = current_time + ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
        else:
            if self._size >= self.capacity:
                lru = self._pop_tail()
                if lru is not None:
                    del self._cache[lru.key]
                    self._size -= 1

            node = _Node(key, value, expires_at)
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=2, default_ttl=10.0)

def test_basic_put_and_get(cache):
    """Verify basic insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration_lazy_cleanup(cache):
    """Verify lazy cleanup removes expired items on access."""
    with patch('time.monotonic', side_effect=[0.0, 15.0]):
        cache.put('a', 1)
        # Time advances past TTL; get() triggers lazy cleanup
        assert cache.get('a') is None
        assert cache.size() == 0

def test_lru_eviction(cache):
    """Verify LRU item is evicted when capacity is exceeded."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a' (LRU)
        
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_update_refreshes_ttl_and_mru(cache):
    """Verify updating an existing key refreshes TTL and moves to MRU."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10)  # Updates 'a', moves to MRU
        cache.put('c', 3)   # Evicts 'b' (now LRU)
        
        assert cache.get('b') is None
        assert cache.get('a') == 10
        assert cache.size() == 2

def test_delete_operation(cache):
    """Verify explicit deletion removes items and updates size."""
    with patch('time.monotonic', return_value=0.0):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        
        assert cache.get('a') is None
        assert cache.size() == 1
        
        # Deleting non-existent key should be safe
        cache.delete('nonexistent')
        assert cache.size() == 1

def test_custom_ttl_override(cache):
    """Verify custom TTL overrides default TTL."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 5.0, 5.0]):
        cache.put('a', 1, ttl=2.0)  # Expires at t=2.0
        cache.put('b', 2)           # Expires at t=10.0 (default)
        
        # At t=5.0, 'a' is expired, 'b' is valid
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.size() == 1