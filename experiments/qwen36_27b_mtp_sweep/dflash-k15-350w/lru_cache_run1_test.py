import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) pointer manipulation."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU cache with Time-To-Live (TTL) support.

    Uses a doubly-linked list and a hash map to guarantee O(1) average time
    complexity for `get`, `put`, `delete`, and `size`. Expired entries are
    lazily cleaned up upon access or insertion.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Sentinel nodes simplify edge-case handling in the linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Add a node immediately after the head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve a value by key, or None if missing/expired.

        Args:
            key: The key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        # Lazy cleanup: remove expired entry on access
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
            return

        # Evict LRU if at capacity
        if len(self._cache) >= self._capacity:
            lru_node = self._tail.prev
            self._remove_node(lru_node)
            del self._cache[lru_node.key]

        new_node = _Node(key, value, expires_at)
        self._add_to_head(new_node)
        self._cache[key] = new_node

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists.

        Args:
            key: The key to delete.
        """
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache.

        Returns:
            Number of non-expired items currently stored.
        """
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup: expired key returns None and is removed."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # TTL exceeded
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction_order(mock_time):
    """Test LRU eviction respects access order, not just insertion order."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put('a', 1)
    mock_time.return_value = 1.0
    cache.put('b', 2)
    
    mock_time.return_value = 2.0
    cache.get('a')  # Access 'a', moves it to MRU. Order: a, b
    
    mock_time.return_value = 3.0
    cache.put('c', 3)  # Should evict 'b' (LRU)
    
    assert cache.get('a') == 1
    assert cache.get('b') is None
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    """Test updating a key refreshes TTL and moves to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 3.0
    cache.put('a', 10)  # Updates value & refreshes TTL
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 10  # Should not be expired
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion removes key from cache and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_vs_default(mock_time):
    """Test custom TTL overrides default TTL correctly."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put('a', 1, ttl=2.0)  # Custom short TTL
    cache.put('b', 2)           # Default TTL (10.0)
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None  # Expired
    assert cache.get('b') == 2     # Still valid
    assert cache.size() == 1