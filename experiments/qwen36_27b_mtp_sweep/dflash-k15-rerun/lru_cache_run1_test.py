import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) lookups and a doubly-linked list for O(1) 
    insertion/deletion/movement. Implements lazy cleanup: expired entries 
    are removed on access (get) or when evicted due to capacity limits.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Sentinel nodes for the doubly-linked list
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node immediately after the head sentinel (MRU position)."""
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

    def _evict_lru(self) -> None:
        """Removes the LRU node (immediately before the tail sentinel)."""
        lru_node = self.tail.prev
        if lru_node is self.head:
            return  # Cache is empty
        self._remove_node(lru_node)
        del self.cache[lru_node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves the value associated with `key`.
        
        Returns None if the key is missing or has expired.
        Moves accessed items to the MRU position. Performs lazy cleanup.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        # Lazy cleanup: check expiration on access
        if time.monotonic() >= node.expires_at:
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        
        If `key` exists, updates value and expiration, moves to MRU.
        If `key` is new, evicts LRU entries until capacity is free, then inserts at MRU.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.monotonic() + effective_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
        else:
            # Evict until capacity allows insertion
            while len(self.cache) >= self.capacity:
                self._evict_lru()
                
            node = _Node(key, value, expires_at)
            self._add_to_head(node)
            self.cache[key] = node

    def delete(self, key: Any) -> None:
        """Removes `key` from the cache if it exists."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('ttl_cache.time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time):
    """Test that expired entries are cleaned up on get()."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 106.0  # Advance time past TTL
    assert cache.get('a') is None  # Should return None and clean up
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_lru_eviction_on_capacity(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_time):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 105.0
    cache.put('a', 10, ttl=20.0)  # Update 'a' with longer TTL
    
    mock_time.return_value = 110.0
    assert cache.get('a') == 10  # Should not be expired
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion of a key."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_size_tracking(mock_time):
    """Test that size() accurately reflects cache state."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    assert cache.size() == 0
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2
    
    cache.delete('a')
    assert cache.size() == 1
    
    cache.put('c', 3)
    cache.put('d', 4)
    cache.put('e', 5)  # Evicts 'b'
    assert cache.size() == 3