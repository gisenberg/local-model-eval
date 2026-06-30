import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """
    LRU Cache with TTL support.
    
    Uses a doubly-linked list and a hash map for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed upon access or insertion
    rather than via background threads or periodic scans.
    """
    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items in the cache.
        :param default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive.")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._size: int = 0

        # Dummy head and tail for the doubly-linked list
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_tail(self, node: _Node) -> None:
        """Add node to the tail (most recently used)."""
        prev = self._tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._tail
        self._tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_tail(self, node: _Node) -> None:
        """Move existing node to the tail (mark as recently used)."""
        self._remove_node(node)
        self._add_to_tail(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() > node.expiry

    def _evict_lru(self) -> None:
        """Evict the least recently used item (head.next)."""
        if self._size > 0:
            lru_node = self._head.next
            self._remove_node(lru_node)
            del self._cache[lru_node.key]
            self._size -= 1

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key is missing or expired.
        Updates LRU order on successful access.
        """
        if key not in self._cache:
            return None
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and resets TTL.
        If key is new, adds it. Evicts LRU if at capacity.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_tail(node)
            return

        if self._size >= self.capacity:
            self._evict_lru()

        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        node = _Node(key, value, expiry)
        self._cache[key] = node
        self._add_to_tail(node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return self._size


# =============================================================================
# Pytest Tests
# =============================================================================
import pytest
from unittest.mock import patch

@pytest.fixture
def mock_monotonic():
    """Fixture to mock time.monotonic for deterministic TTL testing."""
    with patch('time.monotonic') as mock:
        mock.return_value = 0.0
        yield mock

def test_basic_put_get(mock_monotonic):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

def test_ttl_expiration_on_get(mock_monotonic):
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_monotonic.return_value = 6.0  # Exceeds 5.0 TTL
    assert cache.get('a') is None
    assert cache.size() == 0

def test_lru_eviction(mock_monotonic):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # Evicts 'a' (LRU)
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

def test_update_refreshes_ttl(mock_monotonic):
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    mock_monotonic.return_value = 4.0
    cache.put('a', 2)  # Refreshes TTL to 4.0 + 5.0 = 9.0
    mock_monotonic.return_value = 8.0  # Still valid
    assert cache.get('a') == 2
    mock_monotonic.return_value = 10.0  # Expired
    assert cache.get('a') is None

def test_delete(mock_monotonic):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.get('b') == 2

def test_custom_ttl(mock_monotonic):
    cache = TTLCache(2, 10.0)
    cache.put('a', 1, ttl=2.0)
    mock_monotonic.return_value = 3.0  # Exceeds custom 2.0 TTL
    assert cache.get('a') is None
    assert cache.size() == 0