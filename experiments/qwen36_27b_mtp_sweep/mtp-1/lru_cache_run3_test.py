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
    
    Uses a hash map for O(1) lookups and a doubly-linked list for O(1)
    insertion/removal and LRU tracking. Implements lazy cleanup of expired
    entries upon access.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for O(1) list operations
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_tail(self, node: _Node) -> None:
        """Add a node right before the tail (most recently used position)."""
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def _move_to_tail(self, node: _Node) -> None:
        """Move an existing node to the tail (mark as most recently used)."""
        self._remove(node)
        self._add_to_tail(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Any:
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
            # Lazy cleanup: remove expired node on access
            self._remove(node)
            del self.cache[key]
            return None
            
        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. Defaults to default_ttl if None.
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Lazy cleanup: treat expired existing key as a new insertion
                self._remove(node)
                del self.cache[key]
            else:
                # Update value, refresh TTL, mark as recently used
                node.value = value
                node.expiry = expiry
                self._move_to_tail(node)
                return

        # Insert new node
        node = _Node(key, value, expiry)
        self.cache[key] = node
        self._add_to_tail(node)

        # Evict LRU if capacity exceeded
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]

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

def test_basic_put_get():
    """Test basic insertion and retrieval."""
    with patch('time.monotonic', side_effect=[100.0, 100.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration_lazy_cleanup():
    """Test that expired entries are cleaned up lazily on get()."""
    with patch('time.monotonic', side_effect=[100.0, 115.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') is None  # Expired after 10s
        assert cache.size() == 0

def test_lru_eviction():
    """Test that least recently used items are evicted when capacity is exceeded."""
    with patch('time.monotonic', side_effect=[100.0] * 6):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_update_existing_key_refreshes_ttl():
    """Test that updating an existing key refreshes its TTL and moves it to MRU."""
    with patch('time.monotonic', side_effect=[100.0, 100.0, 105.0, 115.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('a', 2)  # Updates value & TTL at t=100
        assert cache.get('a') == 2  # Valid at t=105
        assert cache.get('a') is None  # Expired at t=115

def test_delete_operation():
    """Test explicit deletion of a cache entry."""
    with patch('time.monotonic', side_effect=[100.0, 100.0, 100.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 0

def test_custom_ttl_overrides_default():
    """Test that custom TTL in put() overrides default_ttl."""
    with patch('time.monotonic', side_effect=[100.0, 106.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1, ttl=5.0)  # Custom 5s TTL
        assert cache.get('a') is None  # Expired after 5s, not 10s
        assert cache.size() == 0