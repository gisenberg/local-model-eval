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
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed or evicted.
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
            raise ValueError("Default TTL must be a positive number.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)  # MRU end
        self._tail = _Node(None, None, 0.0)  # LRU end
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the head (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the node before the tail (LRU position)."""
        if self._tail.prev == self._head:
            return None
        lru_node = self._tail.prev
        self._remove_node(lru_node)
        return lru_node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up.

        Returns:
            The value if found and not expired, otherwise None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert or update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. Uses default_ttl if None.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl

        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Expired: treat as a new insertion
                self._remove_node(node)
                del self.cache[key]
                self._size -= 1
            else:
                # Valid: update value, refresh TTL, move to MRU
                node.value = value
                node.expires_at = time.monotonic() + effective_ttl
                self._move_to_head(node)
                return

        # Evict LRU if at capacity
        if self._size >= self.capacity:
            evicted = self._pop_tail()
            if evicted:
                del self.cache[evicted.key]
                self._size -= 1

        # Insert new node
        new_node = _Node(key, value, time.monotonic() + effective_ttl)
        self.cache[key] = new_node
        self._add_to_head(new_node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.

        Args:
            key: The key to remove.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            The number of items currently stored (lazy cleanup may temporarily 
            include unaccessed expired entries until they are evicted or accessed).
        """
        return self._size

import pytest
from unittest.mock import patch

# Assuming TTLCache is in the same module or imported appropriately
# from . import TTLCache 

@patch('time.monotonic')
def test_basic_put_and_get(mock_monotonic):
    """Test standard insertion and retrieval."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1


@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_monotonic):
    """Test that expired items are lazily removed on access."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_monotonic.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None      # Lazy cleanup triggers here
    assert cache.size() == 0


@patch('time.monotonic')
def test_lru_eviction(mock_monotonic):
    """Test that LRU item is evicted when capacity is reached."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2


@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_monotonic):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_monotonic.return_value = 2.0
    cache.put('a', 10)  # Update value, TTL resets to 7.0
    
    mock_monotonic.return_value = 6.0
    assert cache.get('a') == 10  # Should still be valid
    assert cache.size() == 1


@patch('time.monotonic')
def test_delete_key(mock_monotonic):
    """Test explicit deletion of a key."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1


@patch('time.monotonic')
def test_custom_ttl_vs_default(mock_monotonic):
    """Test that custom TTL overrides default TTL."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)  # Custom short TTL
    cache.put('b', 2)           # Default TTL
    
    mock_monotonic.return_value = 3.0
    assert cache.get('a') is None  # Expired
    assert cache.get('b') == 2     # Still valid
    assert cache.size() == 1