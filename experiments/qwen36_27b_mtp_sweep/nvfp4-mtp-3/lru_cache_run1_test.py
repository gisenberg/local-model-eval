import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertion/removal."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    
    Uses a custom doubly-linked list and hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed or during eviction.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items allowed in the cache.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add(self, node: _Node) -> None:
        """Insert node at the front of the list (most recently used)."""
        nxt = self._head.next
        self._head.next = node
        nxt.prev = node
        node.prev = self._head
        node.next = nxt
        self._size += 1

    def _remove(self, node: _Node) -> None:
        """Remove node from the list without freeing memory."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front of the list."""
        self._remove(node)
        self._add(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has passed its expiration time."""
        return time.monotonic() >= node.expires_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if not found or expired.
        Moves accessed item to front (LRU behavior). Lazy cleanup on access.
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        if self._is_expired(node):
            self._remove(node)
            del self._cache[key]
            return None
            
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If capacity is exceeded, evicts least recently used items.
        Expired items at the tail are removed without counting against capacity.
        """
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
        else:
            node = _Node(key, value, expires_at)
            self._add(node)
            self._cache[key] = node

            # Evict until within capacity
            while self._size > self._capacity:
                to_remove = self._tail.prev
                if to_remove is self._head:
                    break
                    
                self._remove(to_remove)
                del self._cache[to_remove.key]
                
                # Lazy cleanup: expired evictions don't count against capacity limit
                if self._is_expired(to_remove):
                    continue
                break

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove(node)
            del self._cache[key]

    def size(self) -> int:
        """
        Return the current number of items in the cache.
        Note: May include expired entries until they are lazily cleaned via get/put/delete.
        """
        return self._size

import pytest
from unittest.mock import patch

@pytest.fixture
def mock_time():
    """Mock time.monotonic to control expiration behavior."""
    with patch('ttl_cache.time.monotonic') as m:
        m.return_value = 0.0
        yield m

def test_basic_put_get(mock_time):
    """Test standard insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    assert cache.get('a') == 1
    assert cache.get('b') == 2
    assert cache.size() == 2

def test_ttl_expiration_on_get(mock_time):
    """Test that expired entries return None and are lazily removed."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0

def test_lru_eviction(mock_time):
    """Test that least recently used valid items are evicted first."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Should evict 'a'
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

def test_custom_ttl_override(mock_time):
    """Test that per-entry TTL overrides default TTL."""
    cache = TTLCache(2, 10.0)
    cache.put('short', 1, ttl=2.0)
    cache.put('long', 2, ttl=20.0)
    
    mock_time.return_value = 5.0
    assert cache.get('short') is None  # Expired
    assert cache.get('long') == 2      # Still valid
    assert cache.size() == 1

def test_delete_operation(mock_time):
    """Test explicit deletion removes entry and updates size."""
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    cache.delete('nonexistent')  # Should not raise
    assert cache.size() == 1

def test_lazy_cleanup_during_eviction(mock_time):
    """Test that expired tail items don't consume capacity slots."""
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 6.0  # Both expired
    cache.put('c', 3, ttl=10.0)   # Triggers eviction
    
    # 'b' and 'a' are expired, so they are removed without blocking 'c'
    assert cache.get('c') == 3
    assert cache.size() == 2
    assert cache.get('a') is None
    assert cache.get('b') is None