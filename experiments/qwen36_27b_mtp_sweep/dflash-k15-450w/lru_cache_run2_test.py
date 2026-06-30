import time
from typing import Any, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: K, value: V, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class _DoublyLinkedList:
    """Custom doubly-linked list with O(1) insertion, removal, and move-to-front."""

    def __init__(self) -> None:
        # Sentinel nodes to simplify edge-case pointer manipulation
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        """Insert node immediately after the head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: _Node) -> None:
        """Remove a specific node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._size -= 1

    def remove_last(self) -> Optional[_Node]:
        """Remove and return the node before the tail (least recently used)."""
        if self._size == 0:
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front of the list."""
        self.remove(node)
        self.add_to_front(node)

    def __len__(self) -> int:
        return self._size


class TTLCache:
    """LRU cache with Time-To-Live (TTL) support.

    Uses a custom doubly-linked list and a hash map to achieve O(1) average
    time complexity for get, put, and delete operations. Expired entries
    are lazily cleaned up upon access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[K, _Node] = {}
        self._dll = _DoublyLinkedList()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from both the linked list and hash map."""
        self._dll.remove(node)
        del self._cache[node.key]

    def get(self, key: K) -> Optional[V]:
        """Retrieve a value by key, returning None if missing or expired.

        Moves accessed items to the front (most recently used).
        Lazily cleans up expired entries.

        Args:
            key: The key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            return None

        self._dll.move_to_front(node)
        return node.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair in the cache.

        If the key exists, updates the value and refreshes TTL.
        If the cache is full, evicts the least recently used item.

        Args:
            key: The key to insert/update.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.monotonic() + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._dll.move_to_front(node)
        else:
            if len(self._dll) >= self.capacity:
                lru_node = self._dll.remove_last()
                if lru_node is not None:
                    del self._cache[lru_node.key]

            new_node = _Node(key, value, expires_at)
            self._cache[key] = new_node
            self._dll.add_to_front(new_node)

    def delete(self, key: K) -> None:
        """Remove a key from the cache if it exists.

        Args:
            key: The key to delete.
        """
        if key in self._cache:
            self._remove_node(self._cache[key])

    def size(self) -> int:
        """Return the current number of items stored in the cache.

        Note: Due to lazy cleanup, this may include expired entries
        until they are accessed via get().

        Returns:
            The number of entries currently in the cache.
        """
        return len(self._dll)

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    """Fixture providing a fresh cache with capacity=3 and default_ttl=10.0."""
    return TTLCache(capacity=3, default_ttl=10.0)

@patch('ttl_cache.time.monotonic')
def test_basic_put_and_get(mock_time, cache):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time, cache):
    """Test that expired entries are cleaned up lazily on get()."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    
    mock_time.return_value = 11.0  # Advance time past default TTL
    assert cache.get('a') is None  # Should trigger lazy cleanup
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_lru_eviction(mock_time, cache):
    """Test that LRU item is evicted when capacity is exceeded."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    assert cache.size() == 3
    
    cache.put('d', 4)  # Should evict 'a' (LRU)
    assert cache.get('a') is None
    assert cache.get('d') == 4
    assert cache.size() == 3

@patch('ttl_cache.time.monotonic')
def test_update_existing_key(mock_time, cache):
    """Test that updating a key refreshes TTL and moves it to MRU position."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 5.0
    cache.put('a', 10)  # Update 'a', refresh TTL, move to front
    
    assert cache.get('a') == 10
    # 'b' is now LRU. Adding two more should evict 'b'
    cache.put('c', 3)
    cache.put('d', 4)
    assert cache.get('b') is None
    assert cache.get('a') == 10

@patch('ttl_cache.time.monotonic')
def test_delete_key(mock_time, cache):
    """Test explicit deletion of a key."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.get('b') == 2
    
    # Deleting non-existent key should not raise
    cache.delete('z')

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_vs_default(mock_time, cache):
    """Test that custom TTL overrides default TTL correctly."""
    mock_time.return_value = 0.0
    cache.put('short', 1, ttl=5.0)
    cache.put('long', 2)  # Uses default 10.0
    
    mock_time.return_value = 6.0
    assert cache.get('short') is None  # Expired
    assert cache.get('long') == 2      # Still valid