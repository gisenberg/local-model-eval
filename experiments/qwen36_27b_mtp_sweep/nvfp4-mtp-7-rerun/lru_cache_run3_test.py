import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) manipulation."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU Cache with TTL support using a custom doubly-linked list and hash map.
    
    Provides O(1) average time complexity for get, put, delete, and size operations.
    Uses lazy cleanup: expired items are removed only when accessed or when space is needed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize cache with maximum capacity and default TTL in seconds."""
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("TTL must be non-negative")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for simplified list manipulation
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after head sentinel (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move existing node to head to mark as recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove least recently used node (node before tail sentinel)."""
        lru = self.tail.prev
        if lru is not self.head:
            self._remove_node(lru)
            del self.cache[lru.key]

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if node has passed its TTL."""
        return now >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value by key. Returns None if missing or expired.
        
        Performs lazy cleanup: removes expired entry if accessed.
        """
        now = time.monotonic()
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node, now):
            self._remove_node(node)
            del self.cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key-value pair.
        
        Args:
            key: Cache key
            value: Cache value
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self.cache:
            # Update existing: refresh value, TTL, and move to head
            node = self.cache[key]
            node.value = value
            node.expiry = now + effective_ttl
            self._move_to_head(node)
        else:
            # Evict LRU if at capacity
            if len(self.cache) >= self.capacity:
                self._evict_lru()
                
            # Insert new node
            node = _Node(key, value, now + effective_ttl)
            self._add_to_head(node)
            self.cache[key] = node

    def delete(self, key: Any) -> None:
        """Remove key from cache if it exists."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch
from typing import List

# Import TTLCache from your module
# 
@pytest.fixture
def cache():
    """Fixture providing a fresh cache with capacity=2 and default_ttl=10s."""
    return TTLCache(capacity=2, default_ttl=10.0)


def test_basic_put_and_get(cache: TTLCache) -> None:
    """Test standard insertion and retrieval."""
    with patch('time.monotonic', side_effect=[0.0, 0.0]):
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1


def test_ttl_expiration(cache: TTLCache) -> None:
    """Test that get returns None after TTL expires."""
    with patch('time.monotonic', side_effect=[0.0, 11.0]):
        cache.put('a', 1)  # expires at 10.0
        assert cache.get('a') is None  # accessed at 11.0
        assert cache.size() == 0


def test_lru_eviction(cache: TTLCache) -> None:
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 0.0]):
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # triggers eviction of 'a'
        
        assert cache.get('a') is None  # evicted
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2


def test_update_refreshes_ttl(cache: TTLCache) -> None:
    """Test that updating a key refreshes its TTL and moves it to head."""
    with patch('time.monotonic', side_effect=[0.0, 5.0, 14.0]):
        cache.put('a', 1)       # expiry: 10.0
        cache.put('a', 2)       # refreshes expiry to 15.0
        assert cache.get('a') == 2  # 14.0 < 15.0, still valid
        assert cache.size() == 1


def test_delete_removes_entry(cache: TTLCache) -> None:
    """Test explicit deletion."""
    with patch('time.monotonic', side_effect=[0.0, 0.0]):
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 0


def test_custom_ttl_override(cache: TTLCache) -> None:
    """Test that explicit ttl parameter overrides default_ttl."""
    with patch('time.monotonic', side_effect=[0.0, 5.0]):
        cache.put('a', 1, ttl=2.0)  # expires at 2.0
        assert cache.get('a') is None  # accessed at 5.0, expired
        assert cache.size() == 0