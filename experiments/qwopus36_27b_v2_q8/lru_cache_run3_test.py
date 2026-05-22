import time
from typing import Any, Dict, Optional

class Node:
    """Doubly-linked list node to maintain LRU order."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support using a hash map and doubly-linked list.
    Implements lazy cleanup (expires on access/eviction) with O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTLCache.
        
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds for cache entries.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive float.")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: Dict[Any, Node] = {}
        self._size: int = 0
        
        # Sentinel nodes to simplify edge cases (empty list, adding/removing at ends)
        self._head: Node = Node(None, None, 0.0)  # MRU end (Most Recently Used)
        self._tail: Node = Node(None, None, 0.0)  # LRU end (Least Recently Used)
        
        # Link sentinels
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly-linked list in O(1)."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Adds a node to the MRU end (after head) in O(1)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: Node) -> None:
        """Moves an existing node to the MRU end in O(1)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has expired based on the current monotonic time."""
        return node.expires_at <= time.monotonic()

    def get(self, key: Any) -> Any:
        """
        Retrieve the value for the key if it exists and hasn't expired.
        Moves the accessed node to the MRU end.
        
        :param key: The key to look up.
        :return: The value associated with the key, or None if not found or expired.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        # Lazy cleanup: check if expired on access
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        # Access updates LRU order
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        :param key: The key to insert or update.
        :param value: The value to store.
        :param ttl: Custom TTL in seconds for this entry. Falls back to default_ttl if None.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
            self._move_to_head(node)
            return

        # Evict LRU if at capacity
        if self._size >= self._capacity:
            lru_node = self._tail.prev
            self._remove_node(lru_node)
            del self._cache[lru_node.key]
            self._size -= 1

        # Insert new node
        ttl_time = ttl if ttl is not None else self._default_ttl
        node = Node(key, value, time.monotonic() + ttl_time)
        self._cache[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache if it exists.
        
        :param key: The key to delete.
        """
        if key in self._cache:
            node = self._cache[key]
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1

    def size(self) -> int:
        """
        Returns the current number of active entries in the cache.
        
        :return: The number of items in the cache.
        """
        return self._size

import pytest
from unittest.mock import patch

@patch('time.monotonic', return_value=100.0)
def test_basic_put_and_get(mock_time):
    """Test basic insertion and retrieval of a value."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    """Test that the Least Recently Used item is evicted when capacity is exceeded."""
    cache = TTLCache(capacity=2, default_ttl=100.0)
    mock_time.return_value = 100.0
    
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)  # 'a' should be evicted
    
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.size() == 2

@patch('time.monotonic')
def test_ttl_expiration_on_get(mock_time):
    """Test lazy cleanup: expired items are removed and return None upon access."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    mock_time.return_value = 100.0
    cache.put('a', 1)
    
    # Advance time past the 10-second TTL
    mock_time.return_value = 111.0
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_custom_ttl_override(mock_time):
    """Test that a custom TTL provided in put() overrides the default_ttl."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    mock_time.return_value = 100.0
    
    # Insert with custom 5-second TTL
    cache.put('a', 1, ttl=5.0)
    
    # Advance 6 seconds (should be expired, despite default 10s TTL)
    mock_time.return_value = 106.0
    
    assert cache.get('a') is None

@patch('time.monotonic')
def test_delete_method(mock_time):
    """Test explicit deletion of an item before it expires."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    mock_time.return_value = 100.0
    
    cache.put('a', 1)
    assert cache.size() == 1
    
    cache.delete('a')
    
    assert cache.get('a') is None
    assert cache.size() == 0

@patch('time.monotonic')
def test_get_updates_lru_order(mock_time):
    """Test that getting an item moves it to MRU, saving it from eviction."""
    cache = TTLCache(capacity=2, default_ttl=100.0)
    mock_time.return_value = 100.0
    
    cache.put('a', 1)
    cache.put('b', 2)
    cache.get('a')  # 'a' is now MRU, 'b' becomes LRU
    
    cache.put('c', 3)  # 'b' should be evicted, not 'a'
    
    assert cache.get('a') == 1
    assert cache.get('b') is None
    assert cache.get('c') == 3