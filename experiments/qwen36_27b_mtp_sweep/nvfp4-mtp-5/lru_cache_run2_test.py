import time
from typing import Any, Optional


class _Node:
    """Internal doubly-linked list node."""
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
    
    Uses a hash map + custom doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed on access or when making room.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return the node immediately before the tail sentinel (least recently used)."""
        if self._tail.prev is self._head:
            return None
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has exceeded its TTL."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to most recently used position.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If capacity is reached, evicts the least recently used item.
        Lazy cleanup removes expired items from the tail when making room.
        """
        if ttl is None:
            ttl = self._default_ttl
        expiry = time.monotonic() + ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        # Lazy cleanup: remove expired nodes from tail to make room
        while self._size >= self._capacity:
            node = self._tail.prev
            if node is self._head:
                break
            if self._is_expired(node):
                self._remove_node(node)
                del self._cache[node.key]
                self._size -= 1
            else:
                break

        # If still at capacity, evict the actual LRU item
        if self._size >= self._capacity:
            evicted = self._pop_tail()
            if evicted:
                del self._cache[evicted.key]
                self._size -= 1

        new_node = _Node(key, value, expiry)
        self._add_to_head(new_node)
        self._cache[key] = new_node
        self._size += 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return the number of valid items currently in the cache."""
        return self._size

import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1


@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test that expired entries return None and are lazily cleaned."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test LRU eviction when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a' (LRU)
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3


@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that explicit ttl parameter overrides default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL
    assert cache.get('a') is None


@patch('time.monotonic')
def test_delete_operation(mock_time):
    """Test explicit deletion removes key from cache and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_size_tracking_with_expiration_and_eviction(mock_time):
    """Test size updates correctly across expiration, eviction, and insertion."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.size() == 2

    mock_time.return_value = 6.0
    assert cache.get('a') is None  # Expires on access
    assert cache.size() == 1

    cache.put('c', 3)  # Adds 'c', size becomes 2
    assert cache.size() == 2
    assert cache.get('b') == 2
    assert cache.get('c') == 3