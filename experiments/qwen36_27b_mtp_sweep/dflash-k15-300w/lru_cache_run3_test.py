import time
from typing import Any, Optional
import pytest
from unittest.mock import patch


class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'exp_time', 'prev', 'next')

    def __init__(self, key: Any, value: Any, exp_time: float) -> None:
        self.key = key
        self.value = value
        self.exp_time = exp_time
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list and a hash map to guarantee O(1) average
    time complexity for get, put, and delete operations. Implements lazy cleanup:
    expired entries are only removed when accessed or evicted, avoiding background threads.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes simplify edge-case handling in the doubly-linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head to mark it as most recently used."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the tail node (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL using monotonic time."""
        return time.monotonic() >= node.exp_time

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve the value for a key if it exists and is not expired.
        Moves the accessed key to the most recently used position.
        Returns None if the key is missing or expired.
        """
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired entry on access
            self._remove_node(node)
            del self.cache[key]
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        If the cache is at capacity, the least recently used item is evicted.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if key in self.cache:
            # Update existing entry
            node = self.cache[key]
            node.value = value
            node.exp_time = time.monotonic() + effective_ttl
            self._move_to_head(node)
        else:
            # Evict LRU if at capacity
            if len(self.cache) >= self.capacity:
                lru_node = self._pop_tail()
                del self.cache[lru_node.key]
                
            # Insert new entry
            new_node = _Node(key, value, time.monotonic() + effective_ttl)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)

    def size(self) -> int:
        """Return the current number of active items in the cache."""
        return len(self.cache)


# =============================================================================
# TESTS
# =============================================================================

@patch('time.monotonic', side_effect=[0.0, 0.0])
def test_basic_put_get(mock_time: Any) -> None:
    """Test basic insertion and retrieval."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    assert cache.get('a') == 1


@patch('time.monotonic', side_effect=[0.0, 6.0])
def test_ttl_expiration(mock_time: Any) -> None:
    """Test lazy cleanup when TTL expires on access."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    # Time advances past TTL (5.0)
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[0.0] * 6)
def test_lru_eviction(mock_time: Any) -> None:
    """Test that LRU item is evicted when capacity is reached."""
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3


@patch('time.monotonic', side_effect=[0.0, 0.0, 2.0, 2.0])
def test_custom_ttl_override(mock_time: Any) -> None:
    """Test that custom TTL in put() overrides default_ttl."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    cache.put('b', 2)
    # 'a' expires at t=2.0, 'b' at t=10.0
    assert cache.get('a') is None
    assert cache.get('b') == 2


@patch('time.monotonic', side_effect=[0.0, 0.0])
def test_delete_operation(mock_time: Any) -> None:
    """Test explicit deletion of a key."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic', side_effect=[0.0] * 7)
def test_update_refreshes_ttl_and_lru_order(mock_time: Any) -> None:
    """Test that updating an existing key refreshes TTL and moves it to MRU."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('a', 10)  # Updates 'a', moves to head, refreshes TTL
    cache.put('c', 3)   # Evicts 'b' (now LRU)
    assert cache.get('b') is None
    assert cache.get('a') == 10
    assert cache.get('c') == 3