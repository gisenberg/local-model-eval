import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) pointer manipulation."""
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
    
    Uses a hash map for O(1) key lookup and a doubly-linked list with sentinel 
    nodes to maintain access order. Employs lazy cleanup: expired entries are 
    removed only when accessed or evicted, avoiding background threads/timers.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes simplify edge-case handling (empty list, single node, etc.)
        self.head = _Node(None, None, 0.0)  # MRU side
        self.tail = _Node(None, None, 0.0)  # LRU side
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node immediately after the head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the MRU position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Removes and returns the LRU node (immediately before tail)."""
        if self.tail.prev == self.head:
            return None
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if missing or expired.
        Moves accessed item to MRU position. O(1) average time.
        """
        node = self.cache.get(key)
        if node is None:
            return None

        # Lazy cleanup: check expiration on access
        if time.monotonic() > node.expiry:
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. O(1) average time.
        
        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl if None.
        """
        node = self.cache.get(key)
        if node is not None:
            if time.monotonic() > node.expiry:
                # Expired entry: treat as new insertion
                self._remove_node(node)
                del self.cache[key]
            else:
                # Valid entry: update value, refresh TTL, move to MRU
                node.value = value
                node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                self._move_to_head(node)
                return

        # Evict LRU if at capacity
        if len(self.cache) >= self.capacity:
            lru = self._pop_tail()
            if lru is not None:
                del self.cache[lru.key]

        # Insert new node
        effective_ttl = ttl if ttl is not None else self.default_ttl
        new_node = _Node(key, value, time.monotonic() + effective_ttl)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists. O(1) average time."""
        node = self.cache.get(key)
        if node is not None:
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of valid items in the cache. O(1) time."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that entries are lazily cleaned up after TTL expires."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that LRU item is evicted when capacity is exceeded."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put('a', 1)
    mock_time.return_value = 1.0
    cache.put('b', 2)
    mock_time.return_value = 2.0
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 4.0
    cache.put('a', 10)  # Update value, refreshes expiry to 4 + 5 = 9
    
    mock_time.return_value = 8.0  # Would be expired if TTL wasn't refreshed
    assert cache.get('a') == 10
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion removes item without affecting others."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that custom ttl parameter overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)  # Custom TTL of 2 seconds
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0