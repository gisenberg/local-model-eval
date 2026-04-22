import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.

    Uses a custom doubly-linked list and a hash map for O(1) average time complexity
    on get, put, and delete operations. Implements lazy cleanup by checking TTL
    only upon key access rather than using background threads or timers.
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
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}

        # Dummy head and tail nodes simplify edge-case list operations
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node immediately after the dummy head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head to mark it as recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the node before the dummy tail (least recently used)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() > node.expiration

    def get(self, key: str) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired entry only when accessed
            self._remove_node(node)
            del self._cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to store.
            value: The value to associate with the key.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl if None.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
        else:
            if len(self._cache) >= self.capacity:
                # Evict least recently used item
                lru_node = self._pop_tail()
                del self._cache[lru_node.key]

            new_node = _Node(
                key,
                value,
                time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            )
            self._cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: The key to remove.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            return True
        return False

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired entries are removed only upon access."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    # Advance time past TTL
    mock_time.return_value = 6.0
    assert cache.get('a') is None  # Lazy cleanup triggers here
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_time):
    """Test that updating a key refreshes its TTL and moves it to head."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 2.0
    cache.put('a', 10, ttl=10.0)  # Update value & TTL
    assert cache.get('a') == 10
    
    mock_time.return_value = 11.0
    assert cache.get('a') is None  # Expired 10s after update

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion and size tracking."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    assert cache.delete('a') is True
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.delete('nonexistent') is False

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that per-key TTL overrides the cache default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)  # Default 100s
    cache.put('a', 1, ttl=2.0)  # Custom 2s
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None  # Expired at 2s, not 100s