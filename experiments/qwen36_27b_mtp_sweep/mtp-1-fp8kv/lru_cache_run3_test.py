import time
from typing import TypeVar, Generic, Optional, Dict

K = TypeVar('K')
V = TypeVar('V')

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: K, value: V, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache(Generic[K, V]):
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a custom doubly-linked list + hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed or evicted.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[K, _Node] = {}

        # Dummy head/tail simplify insertion & deletion without edge-case checks
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node: _Node) -> None:
        """Inserts a node immediately after the dummy head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        """Moves an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _pop_tail(self) -> _Node:
        """Removes and returns the node before the dummy tail (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Checks if a node's TTL has elapsed using monotonic time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: K) -> Optional[V]:
        """
        Retrieves the value for `key` if it exists and hasn't expired.
        Moves the accessed item to the front (most recently used).
        Returns None if the key is missing or expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        - If key exists: updates value, refreshes TTL, moves to front.
        - If cache is full: evicts the least recently used item.
        - Uses `ttl` if provided, otherwise falls back to `default_ttl`.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.monotonic()

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expires_at = current_time + effective_ttl
            self._move_to_front(node)
        else:
            if len(self.cache) == self.capacity:
                evicted = self._pop_tail()
                del self.cache[evicted.key]

            node = _Node(key, value, current_time + effective_ttl)
            self._add_to_front(node)
            self.cache[key] = node

    def delete(self, key: K) -> None:
        """Removes `key` from the cache if it exists."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Returns the number of items currently in the cache.
        Note: Due to lazy cleanup, this may include expired entries 
        that haven't been accessed or evicted yet.
        """
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_and_get(mock_monotonic):
    """Test standard insertion and retrieval."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_monotonic):
    """Test that expired items are removed lazily on access."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    
    mock_monotonic.return_value = 6.0  # Past TTL
    assert cache.get('a') is None      # Lazy cleanup triggers here
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_monotonic):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_update_existing_key(mock_monotonic):
    """Test that updating a key refreshes TTL and moves it to front."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_monotonic.return_value = 1.0
    cache.put('a', 10)  # Updates 'a', new expiry = 6.0, moves to front
    
    mock_monotonic.return_value = 5.5
    assert cache.get('b') is None  # 'b' expires at 5.0
    assert cache.get('a') == 10    # 'a' expires at 6.0, still valid

@patch('time.monotonic')
def test_delete_key(mock_monotonic):
    """Test explicit deletion of a key."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1

@patch('time.monotonic')
def test_custom_ttl_override(mock_monotonic):
    """Test that per-item TTL overrides the default."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=2.0)
    
    mock_monotonic.return_value = 3.0
    assert cache.get('a') is None
    assert cache.size() == 0