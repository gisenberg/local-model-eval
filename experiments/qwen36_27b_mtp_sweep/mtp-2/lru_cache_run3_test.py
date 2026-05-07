import time
from typing import Any, Optional

class _Node:
    """Internal doubly-linked list node."""
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
    
    Uses a hash map for O(1) lookups and a custom doubly-linked list 
    to maintain access order. Expired entries are lazily cleaned up 
    on access or during eviction.
    """
    
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}
        
        # Sentinel nodes for O(1) list operations
        self.head = _Node(None, None, 0.0)  # MRU end
        self.tail = _Node(None, None, 0.0)  # LRU end
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: _Node) -> None:
        """Insert node immediately after head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: _Node) -> None:
        """Move existing node to MRU position."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[_Node]:
        """Remove and return node at LRU position."""
        if self.tail.prev == self.head:
            return None
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: _Node) -> bool:
        """Check if node has exceeded its TTL."""
        return time.monotonic() >= node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key.
        
        Returns None if key is missing or expired. 
        Performs lazy cleanup on expired entries.
        """
        if key not in self.cache:
            return None
            
        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        
        If key exists, updates value and refreshes TTL.
        If capacity is exceeded, evicts LRU entry (cleaning expired ones first).
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.monotonic() + ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        # Evict if at capacity
        while len(self.cache) >= self.capacity:
            node = self._pop_tail()
            if node is None:
                break
            if node.key in self.cache:
                del self.cache[node.key]
            # If evicted node was valid, we freed a slot. 
            # If expired, we also freed a slot. Break after one eviction to maintain O(1) avg.
            break

        new_node = _Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: Any) -> None:
        """Remove key from cache if present."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)

    def size(self) -> int:
        """Return current number of items in cache."""
        return len(self.cache)

import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_put_get(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 10.0)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup when accessing an expired key."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    
    mock_time.return_value = 6.0  # Advance time past TTL
    assert cache.get('a') is None
    assert cache.size() == 0  # Lazy cleanup removed it

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
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key_refreshes_ttl(mock_time):
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 3.0
    cache.put('a', 10)  # Update 'a', refreshes TTL to 3.0 + 5.0 = 8.0
    
    mock_time.return_value = 6.0
    assert cache.get('a') == 10  # Should still be valid
    assert cache.get('b') is None  # 'b' expired at 5.0

@patch('time.monotonic')
def test_delete_key(mock_time):
    """Test explicit deletion and size tracking."""
    mock_time.return_value = 0.0
    cache = TTLCache(3, 10.0)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    
    cache.delete('b')
    assert cache.get('b') is None
    assert cache.size() == 2
    assert cache.get('a') == 1
    assert cache.get('c') == 3

@patch('time.monotonic')
def test_custom_ttl_and_lazy_cleanup_during_eviction(mock_time):
    """Test custom TTL and lazy cleanup prioritizing expired nodes during eviction."""
    mock_time.return_value = 0.0
    cache = TTLCache(2, 100.0)
    cache.put('a', 1, ttl=2.0)  # Expires at 2.0
    cache.put('b', 2, ttl=100.0)
    
    mock_time.return_value = 3.0
    cache.put('c', 3)  # Capacity full. 'a' is LRU & expired. Should evict 'a'.
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2