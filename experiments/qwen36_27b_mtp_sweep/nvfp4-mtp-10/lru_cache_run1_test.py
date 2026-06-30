import time
from typing import Any, Optional

class _Node:
    """Internal node for the doubly linked list."""
    __slots__ = ('key', 'value', 'expiry_time', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev = None
        self.next = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Uses a doubly linked list and a hash map for O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        if default_ttl < 0:
            raise ValueError("TTL must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, _Node] = {}

        # Dummy nodes for the doubly linked list
        # Head represents Most Recently Used (MRU)
        # Tail represents Least Recently Used (LRU)
        self.head = _Node(0, 0, 0)
        self.tail = _Node(0, 0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current time."""
        return time.monotonic() > node.expiry_time

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value by key.
        Updates LRU status if found and not expired.
        Performs lazy cleanup if expired.

        Args:
            key: The key to look up.

        Returns:
            The value if found and valid, None otherwise.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy cleanup: check expiration
        if self._is_expired(node):
            self._remove(node)
            del self.cache[key]
            return None

        # Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, updates value and TTL, moves to head.
        If key is new, inserts at head. Evicts LRU if at capacity.

        Args:
            key: The key.
            value: The value.
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + effective_ttl

        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Evict LRU (node before tail)
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

            new_node = _Node(key, value, expiry_time)
            self._add_to_head(new_node)
            self.cache[key] = new_node

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
        Return the number of items currently in the cache.
        Note: May include expired items until they are accessed or evicted.

        Returns:
            Integer count of items.
        """
        return len(self.cache)

import pytest
from unittest.mock import patch
import time

# Assuming TTLCache is defined in the same scope or imported
# 
class TestTTLCache:
    """Tests for TTLCache using time.monotonic mocking."""

    def test_basic_put_get(self):
        """Test 1: Basic insertion and retrieval."""
        cache = TTLCache(capacity=2, default_ttl=10)
        with patch('time.monotonic', side_effect=[0, 0]):
            cache.put(1, 'a', ttl=10)
            result = cache.get(1)
        
        assert result == 'a'
        assert cache.size() == 1

    def test_ttl_expiration(self):
        """Test 2: Item expires after TTL duration."""
        cache = TTLCache(capacity=2, default_ttl=10)
        # Time 0: Insert with TTL 5
        # Time 6: Access (should be expired)
        with patch('time.monotonic', side_effect=[0, 6]):
            cache.put(1, 'a', ttl=5)
            result = cache.get(1)
        
        assert result is None
        assert cache.size() == 0

    def test_capacity_lru_eviction(self):
        """Test 3: LRU eviction when capacity is reached."""
        cache = TTLCache(capacity=2, default_ttl=100)
        # Time 0 for all operations
        with patch('time.monotonic', side_effect=[0, 0, 0, 0, 0]):
            cache.put(1, 'a', ttl=100)
            cache.put(2, 'b', ttl=100)
            cache.put(3, 'c', ttl=100) # Evicts 1
            
            # 1 should be evicted
            assert cache.get(1) is None
            # 2 and 3 should be valid
            assert cache.get(2) == 'b'
            assert cache.get(3) == 'c'

    def test_update_refreshes_ttl(self):
        """Test 4: Updating a key refreshes its TTL."""
        cache = TTLCache(capacity=2, default_ttl=10)
        # Time 0: Insert (Expiry 10)
        # Time 5: Update (Expiry 15)
        # Time 12: Access (Valid)
        # Time 16: Access (Expired)
        with patch('time.monotonic', side_effect=[0, 5, 12, 16]):
            cache.put(1, 'a', ttl=10)
            cache.put(1, 'b', ttl=10)
            assert cache.get(1) == 'b'
            assert cache.get(1) is None

    def test_delete_key(self):
        """Test 5: Explicit deletion of a key."""
        cache = TTLCache(capacity=2, default_ttl=10)
        with patch('time.monotonic', side_effect=[0]):
            cache.put(1, 'a', ttl=10)
            cache.delete(1)
            
            assert cache.get(1) is None
            assert cache.size() == 0

    def test_default_ttl(self):
        """Test 6: Using default TTL when none is provided."""
        cache = TTLCache(capacity=2, default_ttl=5)
        # Time 0: Insert (Expiry 5)
        # Time 6: Access (Expired)
        with patch('time.monotonic', side_effect=[0, 6]):
            cache.put(1, 'a') # No ttl arg
            result = cache.get(1)
        
        assert result is None