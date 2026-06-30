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


class _DoublyLinkedList:
    """Doubly-linked list with sentinel head/tail for O(1) operations."""

    def __init__(self) -> None:
        self.head = _Node(None, None, 0.0)
        self.tail = _Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_to_head(self, node: _Node) -> None:
        """Insert node immediately after head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove(self, node: _Node) -> None:
        """Unlink node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def pop_tail(self) -> _Node:
        """Remove and return the node before tail (LRU)."""
        if self.tail.prev == self.head:
            raise IndexError("Cannot pop from empty list")
        node = self.tail.prev
        self.remove(node)
        return node

    def move_to_head(self, node: _Node) -> None:
        """Move existing node to the most-recently-used position."""
        self.remove(node)
        self.add_to_head(node)


class TTLCache:
    """
    Least Recently Used (LRU) cache with Time-To-Live (TTL) support.
    
    Uses a hash map + doubly-linked list for O(1) average time complexity.
    Implements lazy cleanup: expired entries are removed only when accessed
    or evicted, avoiding background threads or periodic scans.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for entries without explicit TTL.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: dict[Any, _Node] = {}
        self._list = _DoublyLinkedList()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has passed its expiration time."""
        return time.monotonic() >= node.expiry

    def _remove_node(self, node: _Node) -> None:
        """Remove node from both the list and hash map."""
        self._list.remove(node)
        del self._cache[node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns None if the key is missing or has expired.
        Updates LRU order if found and valid. Lazy cleanup occurs here.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            return None
            
        self._list.move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: Cache key.
            value: Cache value.
            ttl: Optional custom TTL in seconds. Falls back to default_ttl if None.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
            self._list.move_to_head(node)
        else:
            # Evict LRU if at capacity
            if len(self._cache) >= self._capacity:
                tail = self._list.pop_tail()
                del self._cache[tail.key]
                # If evicted tail was already expired, we effectively gained a free slot.
                # No capacity adjustment needed since we already removed it from the dict.
                
            expiry = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
            new_node = _Node(key, value, expiry)
            self._list.add_to_head(new_node)
            self._cache[key] = new_node

    def delete(self, key: Any) -> None:
        """Remove a key from the cache if it exists."""
        if key in self._cache:
            self._remove_node(self._cache[key])

    def size(self) -> int:
        """Return the current number of items in the cache (including unaccessed expired entries)."""
        return len(self._cache)

import pytest
from unittest.mock import patch
import time

# Import the cache class (assuming it's in the same module or imported)
# 
def test_basic_put_get():
    """Test basic insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1
        assert cache.size() == 1

def test_ttl_expiration_on_get():
    """Test that expired entries return None and are lazily cleaned up."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(2, 5.0)
        cache.put('a', 1)
        
        mock_time.return_value = 6.0  # Past TTL
        assert cache.get('a') is None
        assert cache.size() == 0  # Lazy cleanup removed it from dict

def test_lru_eviction():
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2

def test_update_refreshes_ttl_and_lru():
    """Test that updating a key refreshes its TTL and moves it to MRU."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(2, 5.0)
        cache.put('a', 1)
        
        mock_time.return_value = 4.0
        cache.put('a', 2)  # Refreshes TTL
        
        mock_time.return_value = 8.0  # 4s after refresh, still valid
        assert cache.get('a') == 2
        
        mock_time.return_value = 10.0  # 6s after refresh, expired
        assert cache.get('a') is None

def test_delete():
    """Test explicit deletion of a key."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.delete('a')
        
        assert cache.get('a') is None
        assert cache.size() == 1

def test_custom_ttl():
    """Test that custom TTL overrides default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(2, 10.0)  # Default TTL is 10s
        cache.put('a', 1, ttl=2.0)  # Custom TTL is 2s
        
        mock_time.return_value = 3.0  # Past custom TTL
        assert cache.get('a') is None