import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node for O(1) insertions and removals."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) key lookups and a doubly-linked list to maintain
    access order. Expired entries are lazily cleaned up on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, _Node] = {}
        
        # Dummy head and tail nodes simplify edge-case handling
        self.head = _Node("", None, 0.0)
        self.tail = _Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after the dummy head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the list without freeing it."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> _Node:
        """Remove and return the node before the dummy tail (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node's TTL has elapsed."""
        return time.monotonic() >= node.expires_at

    def _cleanup_if_expired(self, key: str) -> Optional[_Node]:
        """Lazy cleanup: remove expired node if present, else return it."""
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                self._remove_node(node)
                del self.cache[key]
                return None
            return node
        return None

    def get(self, key: str) -> Any:
        """
        Retrieve value by key. Returns None if missing or expired.
        Moves accessed item to most-recently-used position.
        """
        node = self._cleanup_if_expired(key)
        if node is None:
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If capacity is exceeded, the least recently used item is evicted.
        """
        if ttl is None:
            ttl = self.default_ttl
        expires_at = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Treat expired update as a fresh insertion
                self._remove_node(node)
                del self.cache[key]
            else:
                node.value = value
                node.expires_at = expires_at
                self._move_to_head(node)
                return

        # New insertion
        node = _Node(key, value, expires_at)
        self._add_to_head(node)
        self.cache[key] = node

        if len(self.cache) > self.capacity:
            evicted = self._remove_tail()
            del self.cache[evicted.key]

    def delete(self, key: str) -> None:
        """Remove a key from the cache if it exists."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch
from typing import Any

# Assuming TTLCache is imported from the module above
# 
@patch('time.monotonic')
def test_basic_put_get(mock_time: Any) -> None:
    """Test standard insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_lazy_cleanup(mock_time: Any) -> None:
    """Test that expired items are cleaned up lazily on get()."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    assert cache.size() == 1  # Still in cache until accessed
    
    mock_time.return_value = 6.0
    assert cache.get('a') is None  # Lazy cleanup triggers here
    assert cache.size() == 0

@patch('time.monotonic')
def test_lru_eviction(mock_time: Any) -> None:
    """Test that LRU item is evicted when capacity is exceeded."""
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
def test_update_existing_key_refreshes_ttl(mock_time: Any) -> None:
    """Test that updating an existing key refreshes its TTL and moves it to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 5.0
    cache.put('a', 10)  # Update 'a', TTL resets to 5.0 + 10.0 = 15.0
    
    mock_time.return_value = 12.0
    assert cache.get('a') == 10  # Still valid
    assert cache.get('b') is None  # Expired at 10.0

@patch('time.monotonic')
def test_delete_key(mock_time: Any) -> None:
    """Test explicit deletion removes item from cache and list."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 1
    assert cache.get('b') == 2

@patch('time.monotonic')
def test_custom_ttl_overrides_default(mock_time: Any) -> None:
    """Test that per-item TTL overrides the default TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)  # Default is 100s
    cache.put('a', 1, ttl=5.0)  # Custom TTL
    
    mock_time.return_value = 6.0
    assert cache.get('a') is None  # Expired despite long default TTL