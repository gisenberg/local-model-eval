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


class _DoublyLinkedList:
    """Doubly-linked list with sentinel head/tail for O(1) operations."""

    def __init__(self) -> None:
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def add_to_front(self, node: _Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._size -= 1

    def remove_last(self) -> Optional[_Node]:
        if self._size == 0:
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def move_to_front(self, node: _Node) -> None:
        self.remove(node)
        self.add_to_front(node)

    def __len__(self) -> int:
        return self._size


class TTLCache:
    """LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed on access or during eviction.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._dll = _DoublyLinkedList()

    def _current_time(self) -> float:
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        return self._current_time() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        self._dll.remove(node)
        del self._cache[node.key]

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value by key. Returns None if missing or expired."""
        if key not in self._cache:
            return None
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            return None
        self._dll.move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key-value pair. Uses default_ttl if ttl is None."""
        if ttl is None:
            ttl = self.default_ttl
        expires_at = self._current_time() + ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._dll.move_to_front(node)
            return

        # Evict LRU items until capacity is available
        while len(self._dll) >= self.capacity:
            lru_node = self._dll.remove_last()
            if lru_node is None:
                break
            # Lazy cleanup: discard expired nodes without counting against capacity
            if self._is_expired(lru_node):
                del self._cache[lru_node.key]
                continue
            # Evict valid LRU node to make space
            del self._cache[lru_node.key]
            break

        new_node = _Node(key, value, expires_at)
        self._cache[key] = new_node
        self._dll.add_to_front(new_node)

    def delete(self, key: Any) -> None:
        """Remove key from cache if it exists."""
        if key in self._cache:
            self._remove_node(self._cache[key])

    def size(self) -> int:
        """Return current number of stored items.
        
        Note: Due to lazy cleanup, this may temporarily include expired entries
        until they are accessed or evicted.
        """
        return len(self._cache)

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)

@patch('ttl_cache.time.monotonic')
def test_basic_put_get(mock_time, cache):
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration(mock_time, cache):
    """Test that entries return None after TTL expires."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    mock_time.return_value = 11.0  # Exceeds default_ttl=10.0
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_lru_eviction(mock_time, cache):
    """Test that least recently used item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    cache.put('d', 4)  # Should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.get('d') == 4
    assert cache.size() == 3

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_overrides_default(mock_time, cache):
    """Test that custom TTL parameter overrides default_ttl."""
    mock_time.return_value = 0.0
    cache.put('a', 1, ttl=5.0)
    mock_time.return_value = 6.0
    assert cache.get('a') is None

@patch('ttl_cache.time.monotonic')
def test_delete_removes_entry(mock_time, cache):
    """Test explicit deletion of a cache entry."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_lazy_cleanup_during_eviction(mock_time, cache):
    """Test that expired items are cleaned up lazily during eviction/access."""
    mock_time.return_value = 0.0
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    
    mock_time.return_value = 11.0  # All entries expired
    cache.put('d', 4)  # Eviction triggers lazy cleanup of expired 'c'
    
    # 'a' and 'b' remain in dict until accessed (lazy cleanup)
    assert cache.get('a') is None  # Triggers cleanup
    assert cache.get('b') is None  # Triggers cleanup
    assert cache.get('c') is None
    assert cache.get('d') == 4
    assert cache.size() == 1  # Only 'd' remains after lazy cleanup