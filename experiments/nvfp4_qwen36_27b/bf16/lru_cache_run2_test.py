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
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy expiration cleanup (checks on access/insertion, no background threads).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes simplify edge-case handling in the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the linked list and hash map."""
        node.prev.next = node.next
        node.next.prev = node.prev
        del self._cache[node.key]
        self._size -= 1

    def _add_to_tail(self, node: _Node) -> None:
        """Add a node to the tail of the linked list (most recently used)."""
        prev_node = self._tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self._tail
        self._tail.prev = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() > node.expiration

    def _cleanup_expired_head(self) -> None:
        """Lazily remove expired nodes from the head (LRU end) of the list."""
        while self._head.next != self._tail and self._is_expired(self._head.next):
            self._remove(self._head.next)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove(node)
            return None

        # Move to tail to mark as recently used
        self._remove(node)
        self._add_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert/update.
            value: The value to associate with the key.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiration = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._remove(node)
            self._add_to_tail(node)
            return

        # Lazy cleanup: remove expired items from LRU end to free space
        self._cleanup_expired_head()

        if self._size >= self.capacity:
            # Evict least recently used item
            lru_node = self._head.next
            self._remove(lru_node)

        effective_ttl = ttl if ttl is not None else self.default_ttl
        new_node = _Node(key, value, time.monotonic() + effective_ttl)
        self._cache[key] = new_node
        self._add_to_tail(new_node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to delete.
        """
        if key in self._cache:
            self._remove(self._cache[key])

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size

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
def test_ttl_expiration_on_get(mock_time):
    """Test that expired items return None and are removed."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.get('a')  # 'a' becomes most recently used
    cache.put('c', 3)  # Should evict 'b' (LRU)
    
    assert cache.get('b') is None
    assert cache.get('a') == 1
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that custom TTL overrides default TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL, but within default
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion and size tracking."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    
    cache.delete('nonexistent')  # Should not raise
    assert cache.size() == 1

@patch('time.monotonic')
def test_lazy_cleanup_on_put(mock_time):
    """Test that expired items are lazily cleaned up before insertion when full."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 6.0  # Both items expired
    cache.put('c', 3)  # Triggers lazy cleanup, then inserts
    
    assert cache.size() == 1
    assert cache.get('c') == 3
    assert cache.get('a') is None
    assert cache.get('b') is None