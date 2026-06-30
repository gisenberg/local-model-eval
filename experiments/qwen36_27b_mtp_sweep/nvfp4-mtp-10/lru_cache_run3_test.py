import time
from typing import Any, Optional

class Node:
    """Node for the doubly linked list."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = None
        self.next = None


class TTLCache:
    """
    LRU Cache with TTL support.
    
    Uses a doubly linked list and a hash map to achieve O(1) average time complexity
    for get, put, and delete operations. Implements lazy cleanup for expired items.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, Node] = {}
        
        # Dummy nodes for the doubly linked list to simplify edge cases
        self.head = Node(0, 0, 0)
        self.tail = Node(0, 0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value by key.
        
        Returns None if the key is not found or if the item has expired.
        Updates the LRU status (moves to head) if the item is valid.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value associated with the key, or None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        current_time = time.monotonic()

        # Lazy cleanup: Check expiration on access
        if current_time > node.expiry:
            self._remove(node)
            del self.cache[key]
            return None

        # Move to head (most recently used)
        self._remove(node)
        self._add_to_head(node)

        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, updates the value and refreshes the TTL/LRU status.
        If the key is new, adds it to the cache. Evicts the LRU item if capacity is exceeded.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        actual_ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.monotonic()
        expiry = current_time + actual_ttl

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            # Move to head
            self._remove(node)
            self._add_to_head(node)
        else:
            # Insert new node
            node = Node(key, value, expiry)
            self.cache[key] = node
            self._add_to_head(node)

            # Check capacity and evict LRU if necessary
            if len(self.cache) > self.capacity:
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

    def delete(self, key: Any) -> None:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to delete.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]

    def size(self) -> int:
        """
        Return the current number of items in the cache.
        
        Returns:
            Integer count of items.
        """
        return len(self.cache)


# --- Tests ---

import pytest
from unittest.mock import patch

class TestTTLCache:
    """Test suite for TTLCache."""

    @patch('time.monotonic')
    def test_basic_put_get(self, mock_time):
        """Test basic insertion and retrieval."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        assert cache.get('a') == 1

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        """Test that items are removed when TTL expires."""
        # Time 0: Put with TTL 5. Time 6: Get (should be expired).
        mock_time.side_effect = [0.0, 6.0]
        cache = TTLCache(capacity=2, default_ttl=5.0)
        cache.put('a', 1)
        assert cache.get('a') is None

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        """Test LRU eviction when capacity is reached."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        cache.put('a', 1)
        cache.put('b', 2)
        # Access 'a' to make it recent
        cache.get('a')
        # Add 'c', should evict 'b' (LRU)
        cache.put('c', 3)
        assert cache.get('b') is None
        assert cache.get('a') == 1
        assert cache.get('c') == 3

    @patch('time.monotonic')
    def test_update_existing_key(self, mock_time):
        """Test updating an existing key refreshes TTL and LRU position."""
        # Time 0: Put 'a' (TTL 5, expires 5). 
        # Time 2: Update 'a' (TTL 5, expires 7).
        # Time 6: Get 'a' (should be valid).
        mock_time.side_effect = [0.0, 2.0, 6.0]
        cache = TTLCache(capacity=2, default_ttl=5.0)
        cache.put('a', 1)
        cache.put('a', 2, ttl=5.0)
        assert cache.get('a') == 2

    @patch('time.monotonic')
    def test_delete_key(self, mock_time):
        """Test explicit deletion of a key."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)
        cache.delete('a')
        assert cache.get('a') is None
        assert cache.size() == 0

    @patch('time.monotonic')
    def test_size_and_default_ttl(self, mock_time):
        """Test size method and default TTL usage."""
        mock_time.side_effect = [0.0, 11.0]
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put('a', 1) # Uses default TTL
        assert cache.size() == 1
        # Time 11: Expired
        assert cache.get('a') is None
        assert cache.size() == 0