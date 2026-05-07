import time
from typing import Any, Optional, Dict

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class _DoublyLinkedList:
    """Doubly-linked list with sentinel nodes for O(1) insert/remove/move operations."""

    def __init__(self) -> None:
        self.head = _Node('', None, 0.0)
        self.tail = _Node('', None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the head sentinel."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def remove_node(self, node: _Node) -> None:
        """Unlink node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def remove_tail(self) -> Optional[_Node]:
        """Remove and return the node before the tail sentinel. Returns None if empty."""
        if self.head.next == self.tail:
            return None
        node = self.tail.prev
        self.remove_node(node)
        return node

    def move_to_head(self, node: _Node) -> None:
        """Move an existing node to the most recently used position."""
        self.remove_node(node)
        self.add_to_head(node)


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, _Node] = {}
        self.dll = _DoublyLinkedList()

    def _now(self) -> float:
        return time.monotonic()

    def _remove_node(self, node: _Node) -> None:
        """Remove node from both the linked list and hash map."""
        self.dll.remove_node(node)
        del self.cache[node.key]

    def _is_expired(self, node: _Node) -> bool:
        return self._now() >= node.expires_at

    def get(self, key: str) -> Any:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Moves accessed item to the most recently used position.
        """
        node = self.cache.get(key)
        if node is None:
            return None
        
        if self._is_expired(node):
            self._remove_node(node)
            return None
            
        self.dll.move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        - If key exists: updates value and refreshes TTL.
        - If key is missing: inserts new entry, evicting LRU item if at capacity.
        """
        node = self.cache.get(key)
        if node is not None:
            if self._is_expired(node):
                self._remove_node(node)
            else:
                node.value = value
                node.expires_at = self._now() + (ttl if ttl is not None else self.default_ttl)
                self.dll.move_to_head(node)
                return

        # Evict LRU if at capacity
        if len(self.cache) >= self.capacity:
            evicted = self.dll.remove_tail()
            if evicted:
                del self.cache[evicted.key]

        # Insert new node
        new_node = _Node(key, value, self._now() + (ttl if ttl is not None else self.default_ttl))
        self.dll.add_to_head(new_node)
        self.cache[key] = new_node

    def delete(self, key: str) -> None:
        """Remove key from cache if it exists."""
        node = self.cache.get(key)
        if node is not None:
            self._remove_node(node)

    def size(self) -> int:
        """Return current number of items in cache (includes potentially expired items until accessed)."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test that get() returns None and cleans up after TTL expires."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_time):
    """Test that updating an existing key refreshes its expiration time."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 3.0
    cache.put('a', 2)  # Updates value and resets TTL
    
    mock_time.return_value = 6.0  # Would be expired if TTL wasn't refreshed
    assert cache.get('a') == 2
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion removes key from cache."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time):
    """Test that passing ttl= overrides the default_ttl."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_time.return_value = 3.0  # Past custom TTL, but within default TTL
    assert cache.get('a') is None
    assert cache.size() == 0