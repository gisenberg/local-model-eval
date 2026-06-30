import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.

    Uses a hash map and a custom doubly-linked list to achieve O(1) average
    time complexity for get, put, and delete operations. Expired entries are
    lazily cleaned up upon access rather than via background threads.
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
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes simplify edge-case handling in the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The associated value if found and not expired, else None.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        # Lazy cleanup: check expiration on access
        if time.monotonic() >= node.expiration:
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.
            ttl: Optional custom TTL in seconds. Uses default_ttl if None.
        """
        now = time.monotonic()
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration = now + (ttl if ttl is not None else self._default_ttl)
            self._move_to_head(node)
        else:
            if self._size == self._capacity:
                # Evict LRU item (tail.prev)
                lru = self._tail.prev
                self._remove_node(lru)
                del self._cache[lru.key]
                self._size -= 1

            effective_ttl = ttl if ttl is not None else self._default_ttl
            node = _Node(key, value, now + effective_ttl)
            self._add_to_head(node)
            self._cache[key] = node
            self._size += 1

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to remove.
        """
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

@patch('ttl_cache.time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup when an entry expires."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_time):
    """Test that updating a key refreshes its expiration and moves it to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 2.0
    cache.put('a', 10)  # Update value & refresh TTL
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 10  # Should not be expired
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion and idempotent delete on missing keys."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    
    cache.delete('nonexistent')  # Should not raise

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that custom TTL takes precedence over default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    cache.put('b', 2)
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None  # Expired (custom TTL)
    assert cache.get('b') == 2     # Not expired (default TTL)