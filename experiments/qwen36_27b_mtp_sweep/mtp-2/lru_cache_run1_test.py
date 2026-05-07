import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and a hash map to guarantee O(1) average
    time complexity for all operations. Implements lazy cleanup: expired entries
    are only removed when accessed via `get()`.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes simplify edge-case handling in the linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Unlink a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_tail(self, node: _Node) -> None:
        """Insert a node right before the tail sentinel (most recently used)."""
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def _move_to_tail(self, node: _Node) -> None:
        """Move an existing node to the tail to mark it as most recently used."""
        self._remove(node)
        self._add_to_tail(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired entry on access
            self._remove(node)
            del self.cache[key]
            return None

        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert/update.
            value: The value to cache.
            ttl: Optional TTL in seconds. Uses default_ttl if None.
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_tail(node)
            return

        if len(self.cache) >= self.capacity:
            # Evict least recently used (head.next)
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]

        new_node = _Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_tail(new_node)

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

# Assuming TTLCache is imported or defined in the same module
# from . import TTLCache 

def test_basic_put_and_get():
    """Verify basic insertion and retrieval."""
    with patch('time.monotonic', side_effect=[1.0, 1.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration_lazy_cleanup():
    """Verify expired items are removed lazily on access."""
    with patch('time.monotonic', side_effect=[1.0, 12.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        # Time advances past TTL
        assert cache.get('a') is None
        assert cache.size() == 0

def test_lru_eviction():
    """Verify LRU item is evicted when capacity is exceeded."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_update_existing_key():
    """Verify updating a key refreshes value, TTL, and MRU position."""
    with patch('time.monotonic', side_effect=[1.0, 2.0, 3.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        cache.put('a', 2, ttl=5.0)  # Update value and TTL
        
        assert cache.get('a') == 2
        assert cache.size() == 1

def test_delete_key():
    """Verify explicit deletion removes the key and updates size."""
    with patch('time.monotonic', side_effect=[1.0, 2.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        cache.delete('a')
        
        assert cache.get('a') is None
        assert cache.size() == 0

def test_custom_ttl_overrides_default():
    """Verify custom TTL takes precedence over default_ttl."""
    with patch('time.monotonic', side_effect=[1.0, 5.5, 11.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1, ttl=5.0)  # Expires at 6.0
        
        assert cache.get('a') == 1   # 5.5 < 6.0 -> valid
        assert cache.get('a') is None # 11.0 >= 6.0 -> expired
        assert cache.size() == 0