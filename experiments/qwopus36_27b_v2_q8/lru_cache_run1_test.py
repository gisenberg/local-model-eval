import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly-linked list."""
    __slots__ = 'key', 'value', 'ttl', 'expiry_time', 'prev', 'next'

    def __init__(self, key: Any, value: Any, ttl: float, expiry_time: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expiry_time = expiry_time
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    
    Uses a doubly-linked list and a hash map for O(1) average time complexity
    for get, put, and delete operations. Expired items are cleaned up lazily
    upon access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL cache.

        Args:
            capacity: The maximum number of items the cache can hold.
            default_ttl: The default Time-To-Live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.map: dict[Any, _Node] = {}
        
        # Sentinel nodes to simplify list manipulation
        self.head = _Node(None, None, 0, 0)
        self.tail = _Node(None, None, 0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_tail(self, node: _Node) -> None:
        """Adds a node immediately before the tail sentinel (Most Recently Used)."""
        prev_node = self.tail.prev
        prev_node.next = node
        self.tail.prev = node
        node.prev = prev_node
        node.next = self.tail

    def _remove_node(self, node: _Node) -> None:
        """Removes a node from its current position in the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_tail(self, node: _Node) -> None:
        """Moves an existing node to the end of the linked list (MRU)."""
        self._remove_node(node)
        self._add_to_tail(node)

    def _pop_head(self) -> _Node:
        """Removes and returns the node immediately after the head sentinel (LRU)."""
        node = self.head.next
        self._remove_node(node)
        return node

    def get(self, key: Any) -> Any:
        """
        Retrieves the value associated with the key, or None if not found/expired.

        Args:
            key: The key to look up.

        Returns:
            The associated value, or None.
        """
        if key not in self.map:
            return None
            
        node = self.map[key]
        
        # Lazy cleanup: Evict expired items upon access
        if time.monotonic() >= node.expiry_time:
            self._remove_node(node)
            del self.map[key]
            return None
            
        self._move_to_tail(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates the value for the given key.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.
            ttl: The Time-To-Live for this specific item. Uses default_ttl if None.
        """
        if ttl is None:
            ttl = self.default_ttl
            
        expiry_time = time.monotonic() + ttl

        if key in self.map:
            # Update existing key
            node = self.map[key]
            node.value = value
            node.expiry_time = expiry_time
            node.ttl = ttl
            self._move_to_tail(node)
        else:
            # Insert new key
            if len(self.map) >= self.capacity:
                lru_node = self._pop_head()
                del self.map[lru_node.key]
                
            node = _Node(key, value, ttl, expiry_time)
            self.map[key] = node
            self._add_to_tail(node)

    def delete(self, key: Any) -> bool:
        """
        Removes the key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        if key in self.map:
            node = self.map[key]
            self._remove_node(node)
            del self.map[key]
            return True
        return False

    def size(self) -> int:
        """
        Returns the current number of items in the cache (including expired ones 
        that haven't been lazily cleaned up yet).

        Returns:
            The number of items.
        """
        return len(self.map)

import pytest
from unittest.mock import patch

@patch('ttl_cache.time.monotonic')
def test_basic_put_and_get(mock_time: MagicMock):
    """Test 1: Basic insertion and retrieval."""
    mock_time.return_value = 100.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_ttl_expiration(mock_time: MagicMock):
    """Test 2: Lazy cleanup removes expired items on get()."""
    mock_time.return_value = 100.0
    
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1) # Expires at 105.0
    
    assert cache.get('a') == 1
    assert cache.size() == 1 # Still in memory until lazily cleaned
    
    mock_time.return_value = 105.0 # Time reaches expiry
    assert cache.get('a') is None # Lazy cleanup triggers here
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic', return_value=100.0)
def test_lru_eviction(mock_time: MagicMock):
    """Test 3: Putting a new item evicts the Least Recently Used item."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3) # Should evict 'a'
    
    assert cache.get('a') is None # Evicted
    assert cache.get('b') == 2
    assert cache.get('c') == 3

@patch('ttl_cache.time.monotonic')
def test_update_key_refreshes_ttl_and_lru(mock_time: MagicMock):
    """Test 4: Updating a key moves it to MRU and resets its TTL."""
    mock_time.return_value = 100.0
    
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put('a', 1)
    cache.put('b', 2)
    
    mock_time.return_value = 103.0 # 'a' is about to expire, but we refresh it
    cache.put('a', 10) # Updates value, refreshes TTL to 108.0, moves 'a' to MRU
    
    cache.put('c', 3) # Evicts 'b' (the new LRU)
    
    assert cache.get('a') == 10
    assert cache.get('b') is None

@patch('ttl_cache.time.monotonic', return_value=100.0)
def test_delete_key(mock_time: MagicMock):
    """Test 5: Deleting a key removes it and returns True."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1)
    
    assert cache.delete('a') is True
    assert cache.get('a') is None
    assert cache.delete('a') is False # Deleting again returns False
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_custom_ttl_override(mock_time: MagicMock):
    """Test 6: Passing a custom ttl argument overrides the default_ttl."""
    mock_time.return_value = 100.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put('a', 1, ttl=5.0) # Expires at 105.0
    cache.put('b', 2, ttl=20.0) # Expires at 120.0
    
    mock_time.return_value = 106.0
    
    assert cache.get('a') is None # 'a' expired
    assert cache.get('b') == 2    # 'b' is still alive