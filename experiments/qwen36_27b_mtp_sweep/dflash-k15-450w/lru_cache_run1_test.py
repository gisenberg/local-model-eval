import time
from typing import Any, Optional, Dict

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
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Expired entries are removed lazily during access or eviction.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: Dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: _Node) -> None:
        """Inserts a node immediately after the head sentinel."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Moves an existing node to the most recently used position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node has exceeded its TTL."""
        return time.monotonic() >= node.expires_at

    def _remove_if_expired(self, node: _Node) -> bool:
        """Removes a node if expired. Returns True if removed."""
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[node.key]
            self._size -= 1
            return True
        return False

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves the value for the given key.
        
        Returns None if the key is missing or expired.
        Moves the accessed key to the most recently used position.
        """
        node = self._cache.get(key)
        if node is None:
            return None

        if self._remove_if_expired(node):
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        
        If the cache is full, evicts the least recently used valid entry.
        Expired entries are cleaned up lazily to maintain O(1) amortized time.
        """
        if key in self._cache:
            node = self._cache[key]
            if self._remove_if_expired(node):
                # Expired entry removed; fall through to new insertion logic
                pass
            else:
                node.value = value
                node.expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
                self._move_to_head(node)
                return

        # Evict if at capacity. Skip expired nodes at the tail to avoid 
        # unnecessarily evicting valid entries. Amortized O(1).
        if self._size >= self._capacity:
            while self._size > 0:
                tail_node = self._tail.prev
                if self._remove_if_expired(tail_node):
                    continue
                # Evict valid LRU node
                self._remove_node(tail_node)
                del self._cache[tail_node.key]
                self._size -= 1
                break

        # Insert new node
        effective_ttl = ttl if ttl is not None else self._default_ttl
        new_node = _Node(key, value, time.monotonic() + effective_ttl)
        self._cache[key] = new_node
        self._add_to_head(new_node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """Removes the key from the cache if it exists."""
        node = self._cache.get(key)
        if node is not None:
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Returns the current number of items in the cache."""
        return self._size

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_time: Any) -> None:
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time: Any) -> None:
    """Test that expired keys return None and are lazily removed."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time: Any) -> None:
    """Test per-item TTL takes precedence over default."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)  # Long default
    cache.put('a', 1, ttl=2.0)  # Short custom
    
    mock_time.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction_order(mock_time: Any) -> None:
    """Test that least recently used valid entry is evicted."""
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
def test_lazy_cleanup_during_put(mock_time: Any) -> None:
    """Test that put() cleans up expired entries before inserting."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 6.0  # Both expired
    cache.put('c', 3)  # Should clean 'a' and 'b', then insert 'c'
    
    assert cache.get('a') is None
    assert cache.get('b') is None
    assert cache.get('c') == 3
    assert cache.size() == 1

@patch('time.monotonic')
def test_delete_operation(mock_time: Any) -> None:
    """Test explicit deletion removes key and updates size."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.size() == 1
    
    # Deleting non-existent key should be safe
    cache.delete('z')
    assert cache.size() == 1