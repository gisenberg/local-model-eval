import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertions/removals."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a hash map + doubly-linked list to achieve O(1) average time complexity
    for get, put, and delete operations. Implements lazy expiration cleanup
    to avoid background threads or periodic scans.
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
            raise ValueError("Default TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: dict[Any, _Node] = {}
        self._size = 0

        # Dummy head/tail nodes simplify boundary checks
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _now(self) -> float:
        """Return current monotonic time."""
        return time.monotonic()

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head."""
        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has passed its TTL."""
        return self._now() >= node.expiry

    def _cleanup_expired_from_tail(self) -> None:
        """Lazily remove expired entries from the tail until a valid entry is found."""
        curr = self._tail.prev
        while curr != self._head:
            if self._is_expired(curr):
                self._remove_node(curr)
                del self._map[curr.key]
                self._size -= 1
                curr = curr.prev
            else:
                break

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key.

        Returns None if the key is missing or expired.
        Moves the accessed item to the most recently used position.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._map[key]
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
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        if ttl is None:
            ttl = self._default_ttl
        expiry = self._now() + ttl

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
        else:
            if self._size == self._capacity:
                # Lazy cleanup: remove expired from tail first
                self._cleanup_expired_from_tail()
                if self._size == self._capacity:
                    # Still full, evict true LRU
                    lru = self._tail.prev
                    self._remove_node(lru)
                    del self._map[lru.key]
                    self._size -= 1

            node = _Node(key, value, expiry)
            self._add_to_head(node)
            self._map[key] = node
            self._size += 1

    def delete(self, key: Any) -> bool:
        """
        Remove a key from the cache.

        Returns True if the key was present, False otherwise.
        """
        if key not in self._map:
            return False
        node = self._map[key]
        self._remove_node(node)
        del self._map[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """Return the current number of valid items in the cache."""
        return self._size

import pytest
from unittest.mock import patch
import time

# Import TTLCache from the module above
# 
def test_basic_put_get():
    """Test standard insertion and retrieval."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.get('a') == 1
        assert cache.size() == 2

def test_ttl_expiration_on_get():
    """Test that expired entries return None and are removed."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 11.0]):
        cache.put('a', 1)
        assert cache.get('a') is None
        assert cache.size() == 0

def test_lru_eviction():
    """Test that least recently used items are evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Evicts 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_custom_ttl_override():
    """Test that per-key TTL overrides the default."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 0.0, 5.0, 5.0]):
        cache.put('a', 1, ttl=2.0)
        cache.put('b', 2)
        assert cache.get('a') is None  # Expired at t=2
        assert cache.get('b') == 2    # Valid until t=10
        assert cache.size() == 1

def test_delete_operation():
    """Test explicit key deletion."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.delete('a') is True
        assert cache.delete('z') is False
        assert cache.size() == 1
        assert cache.get('a') is None

def test_lazy_cleanup_on_put():
    """Test that expired items are lazily cleaned when cache is full."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    with patch('time.monotonic', side_effect=[0.0, 0.0, 11.0, 11.0]):
        cache.put('a', 1, ttl=5.0)
        cache.put('b', 2, ttl=5.0)
        # Cache is full with expired items. put('c') triggers lazy cleanup.
        cache.put('c', 3)
        assert cache.size() == 1
        assert cache.get('c') == 3