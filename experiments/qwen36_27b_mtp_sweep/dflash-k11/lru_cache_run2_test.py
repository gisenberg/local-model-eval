import time
from typing import Any, Optional


class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiration: float) -> None:
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU Cache with TTL support using a custom doubly-linked list and hash map.
    
    Operations run in O(1) average time. Expired entries are lazily cleaned up
    on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("TTL must be a positive float")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}

        # Dummy head (MRU end) and tail (LRU end) simplify list operations
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiration

    def _remove_node(self, node: _Node) -> None:
        """Unlink a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the dummy head (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the LRU node (right before dummy tail)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value by key. Returns None if key is missing or expired."""
        node = self._cache.get(key)
        if node is None:
            return None
            
        # Lazy cleanup: remove if expired
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair. Refreshes TTL on update."""
        if ttl is None:
            ttl = self._default_ttl

        node = self._cache.get(key)
        if node is not None:
            # Lazy cleanup on update: treat expired keys as new entries
            if self._is_expired(node):
                self._remove_node(node)
                del self._cache[key]
                node = None

        if node is not None:
            # Update existing valid entry
            node.value = value
            node.expiration = time.monotonic() + ttl
            self._move_to_head(node)
        else:
            # Insert new entry
            if len(self._cache) >= self._capacity:
                lru = self._pop_tail()
                del self._cache[lru.key]

            new_node = _Node(key, value, time.monotonic() + ttl)
            self._cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """Remove key from cache. Returns True if key existed, False otherwise."""
        node = self._cache.get(key)
        if node is None:
            return False
        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """Return current number of valid items in the cache."""
        return len(self._cache)

import pytest
from unittest.mock import patch


def test_basic_put_get():
    """Verify basic insertion and retrieval."""
    with patch('time.monotonic', side_effect=[100.0, 100.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1


def test_ttl_expiration_lazy_cleanup():
    """Verify lazy cleanup removes expired entries on access."""
    # put@100 (expires@110), get@105 (valid), get@115 (expired)
    with patch('time.monotonic', side_effect=[100.0, 105.0, 115.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.get('a') is None  # Lazy cleanup triggers here
        assert cache.size() == 0


def test_lru_eviction():
    """Verify LRU eviction when capacity is exceeded."""
    # Capacity 2. Insert a, b, c. 'a' should be evicted.
    with patch('time.monotonic', side_effect=[100.0] * 6):
        cache = TTLCache(2, 100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2


def test_update_existing_key_refreshes_ttl():
    """Verify updating an existing key refreshes its expiration time."""
    # put@100 (expires@110), update@105 (expires@115), get@114 (valid)
    with patch('time.monotonic', side_effect=[100.0, 105.0, 114.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('a', 2)  # Updates value and resets TTL
        assert cache.get('a') == 2


def test_delete_key():
    """Verify explicit deletion and size tracking."""
    with patch('time.monotonic', side_effect=[100.0, 100.0]):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.delete('a') is True
        assert cache.get('a') is None
        assert cache.delete('a') is False
        assert cache.size() == 0


def test_custom_ttl_override_and_mixed_eviction():
    """Verify custom TTL override and correct size after mixed operations."""
    # put('a', ttl=5)@100 (expires@105), put('b')@100 (expires@110)
    # get('a')@105 (expired), get('b')@105 (valid)
    with patch('time.monotonic', side_effect=[100.0, 100.0, 105.0, 105.0]):
        cache = TTLCache(3, 10.0)
        cache.put('a', 1, ttl=5.0)
        cache.put('b', 2)
        assert cache.size() == 2
        
        assert cache.get('a') is None  # Custom TTL expired
        assert cache.get('b') == 2     # Default TTL still valid
        assert cache.size() == 1