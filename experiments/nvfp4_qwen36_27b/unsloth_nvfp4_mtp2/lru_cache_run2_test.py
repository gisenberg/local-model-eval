import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) pointer manipulation."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.

    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes simplify edge-case pointer updates
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Detach a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node immediately after the head sentinel (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Reposition an existing node to the MRU end."""
        self._remove(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if current monotonic time has passed the node's expiry."""
        return time.monotonic() >= node.expiry

    def _cleanup_expired(self, node: _Node) -> bool:
        """Remove node if expired. Returns True if removal occurred."""
        if self._is_expired(node):
            self._remove(node)
            del self.cache[node.key]
            return True
        return False

    def _evict_lru(self) -> None:
        """Remove the least recently used item (node before tail sentinel)."""
        lru = self.tail.prev
        if lru != self.head:
            self._remove(lru)
            del self.cache[lru.key]

    def get(self, key: Any) -> Any:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to head to update LRU order. O(1) avg.
        """
        node = self.cache.get(key)
        if node is None:
            return None
        if self._cleanup_expired(node):
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. If key exists, updates value & TTL.
        If cache is full, evicts LRU item. O(1) avg.
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            if self._cleanup_expired(node):
                # Expired entry removed; fall through to insert as new
                pass
            else:
                node.value = value
                node.expiry = expiry
                self._move_to_head(node)
                return

        if len(self.cache) >= self.capacity:
            self._evict_lru()

        new_node = _Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """
        Remove key from cache. Returns True if key existed, False otherwise.
        O(1) avg.
        """
        node = self.cache.get(key)
        if node is None:
            return False
        self._remove(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """Return current number of valid items in cache. O(1)."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)

def test_basic_put_get(cache):
    """Verify standard insertion and retrieval."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.get('a') == 1
        assert cache.get('b') == 2

def test_lru_eviction(cache):
    """Verify LRU eviction when capacity is reached."""
    with patch('time.monotonic', side_effect=[0.0] * 8):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        cache.put('d', 4)  # Should evict 'a'
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.get('d') == 4

def test_ttl_expiration_on_get(cache):
    """Verify lazy cleanup triggers on get when TTL expires."""
    with patch('time.monotonic', side_effect=[0.0, 15.0]):
        cache.put('x', 100)
        assert cache.get('x') is None
        assert cache.size() == 0

def test_ttl_expiration_on_put(cache):
    """Verify expired key is replaced cleanly on put."""
    with patch('time.monotonic', side_effect=[0.0, 15.0, 15.0]):
        cache.put('y', 200)
        cache.put('y', 300)  # Old expires, new inserted
        assert cache.get('y') == 300
        assert cache.size() == 1

def test_custom_ttl_vs_default(cache):
    """Verify custom TTL overrides default TTL correctly."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 5.0, 15.0]):
        cache.put('short', 1, ttl=5.0)
        cache.put('long', 2)  # Uses default 10.0
        assert cache.get('short') is None  # Expires at 5.0
        assert cache.get('long') is None   # Expires at 15.0

def test_delete_and_size(cache):
    """Verify delete removes entries and size tracks accurately."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        assert cache.size() == 2
        assert cache.delete('a') is True
        assert cache.size() == 1
        assert cache.delete('nonexistent') is False
        assert cache.get('a') is None