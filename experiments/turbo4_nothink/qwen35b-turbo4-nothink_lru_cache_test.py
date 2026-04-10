import time
from typing import Any, Dict, Optional, List

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'ttl', 'expire_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, ttl: float, expire_at: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expire_at = expire_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    
    Supports O(1) get, put, delete, and size operations using a 
    doubly-linked list and a hash map.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds for new items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("TTL cannot be negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: Dict[str, _Node] = {}
        
        # Dummy head and tail for easier list manipulation
        self._head = _Node("", "", 0, 0)
        self._tail = _Node("", "", 0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        """Add a node immediately after the head (most recently used)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """Evict the least recently used item (before the tail)."""
        if self._size == 0:
            return
        
        # The item before tail is the LRU
        lru_node = self._tail.prev
        
        if lru_node == self._head:
            return # Only dummy nodes exist

        self._remove_node(lru_node)
        del self._cache[lru_node.key]
        self._size -= 1

    def _cleanup_expired(self) -> None:
        """
        Scan and remove all expired items.
        Called when the cache is full and we need to make room, 
        or when checking size.
        """
        current_time = time.monotonic()
        # We must iterate carefully. Since we are modifying the list,
        # we collect keys first to avoid iterator issues, or traverse safely.
        # Given the requirement for O(1) average, full scans are O(N) worst case,
        # but they only happen when capacity is reached or size is queried.
        
        # To ensure we don't break the list traversal while deleting,
        # we traverse from head to tail.
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if current.expire_at <= current_time:
                # Remove from list
                self._remove_node(current)
                # Remove from dict
                del self._cache[current.key]
                self._size -= 1
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for a key.
        
        :param key: The key to retrieve.
        :return: The value if found and not expired, else None.
        """
        if key not in self._cache:
            return None

        node = self._cache[key]
        current_time = time.monotonic()

        if node.expire_at <= current_time:
            # Expired: Remove from cache and list
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None

        # Accessing makes it most recently used
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        :param key: The key to insert/update.
        :param value: The value to store.
        :param ttl: Optional custom TTL. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = current_time + effective_ttl

        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.ttl = effective_ttl
            node.expire_at = expire_at
            
            # Move to front (MRU)
            self._remove_node(node)
            self._add_to_front(node)
            return

        # New item
        if self._size >= self.capacity:
            # Check for expired items first to potentially free space
            self._cleanup_expired()
            
            # If still full, evict LRU
            if self._size >= self.capacity:
                self._evict_lru()

        new_node = _Node(key, value, effective_ttl, expire_at)
        self._cache[key] = new_node
        self._add_to_front(new_node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        :param key: The key to remove.
        :return: True if the key existed and was removed, False otherwise.
        """
        if key not in self._cache:
            return False
        
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        Performs lazy cleanup of expired items.
        """
        current_time = time.monotonic()
        
        # We need to clean up expired items to return an accurate count.
        # We traverse and remove expired nodes.
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if current.expire_at <= current_time:
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
            current = next_node
            
        return self._size

import pytest
from unittest.mock import patch
from typing import Any, Optional


# Helper to create a mock time function that increments
def create_mock_time(initial: float = 0.0):
    current_time = [initial]
    def mock_monotonic():
        return current_time[0]
    def advance(seconds: float):
        current_time[0] += seconds
    return mock_monotonic, advance

class TestTTLCache:
    @pytest.fixture
    def cache(self):
        return TTLCache(capacity=3, default_ttl=10.0)

    @pytest.fixture
    def mock_time(self):
        return create_mock_time(initial=0.0)

    def test_basic_get_put(self, cache, mock_time):
        """Test basic insertion and retrieval."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            assert cache.get("key2") is None

    def test_capacity_eviction_lru(self, cache, mock_time):
        """Test that LRU items are evicted when capacity is reached."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            # Fill cache to capacity
            cache.put("a", 1)
            cache.put("b", 2)
            cache.put("c", 3)
            
            # Access 'a' to make it MRU
            cache.get("a")
            
            # Add 'd'. 'b' should be evicted (LRU)
            cache.put("d", 4)
            
            assert cache.get("a") == 1
            assert cache.get("b") is None  # Evicted
            assert cache.get("c") == 3
            assert cache.get("d") == 4

    def test_ttl_expiry(self, cache, mock_time):
        """Test that items expire after their TTL."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            cache.put("expiring", "value", ttl=5.0)
            
            # Should exist
            assert cache.get("expiring") == "value"
            
            # Advance time past TTL
            advance(6.0)
            
            # Should be expired
            assert cache.get("expiring") is None

    def test_custom_per_key_ttl(self, cache, mock_time):
        """Test that custom TTL overrides default TTL."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            # Default TTL is 10.0
            cache.put("short", "val", ttl=2.0)
            cache.put("long", "val", ttl=20.0)
            
            advance(3.0)
            
            # "short" should be gone
            assert cache.get("short") is None
            # "long" should still exist
            assert cache.get("long") == "val"

    def test_delete(self, cache, mock_time):
        """Test deletion of keys."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            cache.put("x", 100)
            assert cache.delete("x") is True
            assert cache.delete("x") is False  # Already deleted
            assert cache.get("x") is None

    def test_size_mixed_expired_valid(self, cache, mock_time):
        """Test size() returns count of non-expired items."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            cache.put("valid", 1, ttl=10.0)
            cache.put("expired", 2, ttl=1.0)
            
            # Size should be 2
            assert cache.size() == 2
            
            advance(2.0)
            
            # "expired" is now gone, "valid" remains
            assert cache.size() == 1
            
            # Accessing "valid" should not change size
            cache.get("valid")
            assert cache.size() == 1

    def test_evict_expired_when_full(self, cache, mock_time):
        """Test that expired items are cleared before evicting LRU when full."""
        mock_monotonic, advance = mock_time
        
        with patch('ttl_cache.time.monotonic', mock_monotonic):
            # Fill cache
            cache.put("a", 1, ttl=10.0)
            cache.put("b", 2, ttl=10.0)
            cache.put("c", 3, ttl=10.0)
            
            # Make 'a' and 'b' expire
            advance(11.0)
            
            # Now add 'd'. Since 'a' and 'b' are expired, they should be removed
            # and 'd' added without evicting 'c' (which is valid).
            cache.put("d", 4)
            
            assert cache.get("a") is None
            assert cache.get("b") is None
            assert cache.get("c") == 3
            assert cache.get("d") == 4
            assert cache.size() == 2