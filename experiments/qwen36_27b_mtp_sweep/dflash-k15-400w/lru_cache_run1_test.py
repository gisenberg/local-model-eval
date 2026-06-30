import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'exp', 'prev', 'next')

    def __init__(self, key: Any, value: Any, exp: float) -> None:
        self.key = key
        self.value = value
        self.exp = exp
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a hash map for O(1) lookups and a sentinel-based doubly-linked list 
    for O(1) reordering/eviction. Implements lazy cleanup by checking expiration 
    on access rather than using background threads.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes: head (MRU end) <-> tail (LRU end)
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head sentinel (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node's TTL has elapsed."""
        return time.monotonic() >= node.exp

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to MRU position. O(1) average time.
        """
        node = self.cache.get(key)
        if node is None:
            return None
        
        if self._is_expired(node):
            self._remove(node)
            del self.cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. If capacity is exceeded,
        evicts the LRU item. O(1) average time.
        """
        if ttl is None:
            ttl = self.default_ttl
        exp = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                self._remove(node)
                del self.cache[key]
            else:
                node.value = value
                node.exp = exp
                self._move_to_head(node)
                return

        # Evict LRU if at capacity
        if len(self.cache) >= self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

        new_node = _Node(key, value, exp)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) average time."""
        node = self.cache.get(key)
        if node is not None:
            self._remove(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Return the number of items currently in the cache. O(1) time.
        Note: Due to lazy cleanup, this may include entries that are 
        technically expired but haven't been accessed yet.
        """
        return len(self.cache)

import pytest
from unittest.mock import patch

class TestTTLCache:
    @patch('time.monotonic', side_effect=[0.0, 0.0])
    def test_basic_put_and_get(self, mock_time):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

    @patch('time.monotonic', side_effect=[0.0, 6.0])
    def test_ttl_expiration_on_get(self, mock_time):
        cache = TTLCache(2, 5.0)
        cache.put('a', 1)
        # Time advances past TTL (5.0)
        assert cache.get('a') is None
        assert cache.size() == 0

    @patch('time.monotonic', side_effect=[0.0] * 6)
    def test_lru_eviction(self, mock_time):
        cache = TTLCache(2, 100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3

    @patch('time.monotonic', side_effect=[0.0, 3.0, 6.0])
    def test_update_existing_key(self, mock_time):
        cache = TTLCache(2, 5.0)
        cache.put('a', 1)
        cache.put('a', 10)  # Updates value & refreshes TTL
        # Time is now 6.0 (past original TTL, but within refreshed TTL)
        assert cache.get('a') == 10

    @patch('time.monotonic', side_effect=[0.0, 0.0, 0.0])
    def test_delete_key(self, mock_time):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 1

    @patch('time.monotonic', side_effect=[0.0, 3.0])
    def test_custom_ttl_overrides_default(self, mock_time):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1, ttl=2.0)
        # Time advances past custom TTL (2.0)
        assert cache.get('a') is None