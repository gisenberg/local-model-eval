Here's a complete implementation of a TTLCache with a doubly-linked list and hash map, along with the requested pytest tests:

```python
from typing import Any, Optional
import time


class _Node:
    """Doubly-linked list node for LRU tracking."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU cache with time-based expiration.

    Uses a doubly-linked list for O(1) LRU ordering and a hash map
    for O(1) key access. Expired items are lazily removed.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with maximum capacity and default TTL.

        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # Most recently used
        self._tail: Optional[_Node] = None  # Least recently used

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

        node.prev = None
        node.next = None

    def _add_to_head(self, node: _Node) -> None:
        """Add node to the front (most recently used position)."""
        node.prev = None
        node.next = self._head

        if self._head:
            self._head.prev = node
        else:
            self._tail = node

        self._head = node

    def _move_to_head(self, node: _Node) -> None:
        """Move existing node to the front."""
        if node is self._head:
            return

        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if node has expired."""
        return node.expires_at <= now

    def _evict_lru(self, now: float) -> None:
        """
        Evict least-recently-used non-expired item.
        If all items are expired, clear them all.
        """
        # First, remove all expired items from the tail
        while self._tail and self._is_expired(self._tail, now):
            expired = self._tail
            self._remove_node(expired)
            del self._cache[expired.key]

        # If cache is empty after cleanup, nothing more to do
        if not self._tail:
            return

        # Evict the LRU non-expired item
        lru = self._tail
        self._remove_node(lru)
        del self._cache[lru.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value for key if it exists and hasn't expired.

        Accessing a key makes it most-recently-used.

        Args:
            key: Cache key

        Returns:
            Value if present and valid, else None
        """
        node = self._cache.get(key)
        if not node:
            return None

        now = time.monotonic()
        if self._is_expired(node, now):
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self._cache[key]
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        If at capacity, evicts LRU non-expired item. If all items
        are expired, clears them all first.

        Args:
            key: Cache key
            value: Value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        now = time.monotonic()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        # If key exists, update value and TTL, move to head
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
            return

        # Check capacity
        if len(self._cache) >= self.capacity:
            self._evict_lru(now)

        # Create new node and add to head
        node = _Node(key, value, expires_at)
        self._cache[key] = node
        self._add_to_head(node)

    def _evict_lru(self, now: float) -> None:
        """
        Evict least-recently-used item, handling expired items.
        """
        # Remove expired items from tail first
        while self._tail and self._is_expired(self._tail, now):
            expired = self._tail
            self._remove_node(expired)
            del self._cache[expired.key]

        # If all items were expired, cache is now empty
        if not self._tail:
            return

        # Evict LRU non-expired item
        lru = self._tail
        self._remove_node(lru)
        del self._cache[lru.key]

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed, False otherwise
        """
        node = self._cache.get(key)
        if not node:
            return False

        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """
        Return count of non-expired items.

        Performs lazy cleanup of expired items.

        Returns:
            Number of valid items in cache
        """
        now = time.monotonic()

        # Remove expired items from tail
        while self._tail and self._is_expired(self._tail, now):
            expired = self._tail
            self._remove_node(expired)
            del self._cache[expired.key]

        return len(self._cache)
```

Now the pytest tests:

```python
import pytest
from unittest.mock import patch
from typing import Any, Optional
import time
from ttl_cache import TTLCache


class TestTTLCache:
    """Test suite for TTLCache."""

    def test_basic_get_put(self):
        """Test basic put/get operations."""
        cache = TTLCache(capacity=3, default_ttl=10)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("missing") is None

        # Update existing key
        cache.put("a", 10)
        assert cache.get("a") == 10

    def test_capacity_eviction_lru_order(self):
        """Test LRU eviction when at capacity."""
        cache = TTLCache(capacity=3, default_ttl=100)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access 'a' to make it most recently used
        cache.get("a")

        # Add new item, should evict 'b' (LRU)
        cache.put("d", 4)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

        # Access 'c' then add new item, should evict 'a'
        cache.get("c")
        cache.put("e", 5)

        assert cache.get("a") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert cache.get("e") == 5

    def test_ttl_expiry(self):
        """Test TTL-based expiration."""
        cache = TTLCache(capacity=5, default_ttl=10)

        cache.put("a", 1)
        cache.put("b", 2)

        # Mock time to advance 11 seconds
        with patch('time.monotonic', side_effect=[0, 0, 11, 11]):
            # First get returns value (not expired yet)
            assert cache.get("a") == 1

            # Second get should return None (expired)
            assert cache.get("a") is None

            # 'b' should also be expired
            assert cache.get("b") is None

    def test_custom_per_key_ttl(self):
        """Test custom TTL override for individual keys."""
        cache = TTLCache(capacity=5, default_ttl=100)

        # Custom TTL of 5 seconds
        cache.put("short", 1, ttl=5)
        # Default TTL of 100 seconds
        cache.put("long", 2)

        # After 6 seconds, 'short' should be expired
        with patch('time.monotonic', side_effect=[0, 0, 6, 6]):
            assert cache.get("short") is None
            assert cache.get("long") == 2

        # After 101 seconds, 'long' should be expired
        with patch('time.monotonic', side_effect=[0, 0, 101, 101]):
            assert cache.get("long") is None

    def test_delete(self):
        """Test delete operation."""
        cache = TTLCache(capacity=5, default_ttl=100)

        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.delete("a") is True
        assert cache.delete("a") is False  # Already deleted
        assert cache.get("a") is None
        assert cache.get("b") == 2

        # Delete non-existent key
        assert cache.delete("missing") is False

    def test_size_with_mixed_expired_valid(self):
        """Test size with mixed expired and valid items."""
        cache = TTLCache(capacity=5, default_ttl=100)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Simulate time passing
        with patch('time.monotonic', side_effect=[0, 0, 0, 50, 50, 50]):
            # Access 'a' to keep it valid
            cache.get("a")

            # Simulate 'b' and 'c' expiring
            # (We'll mock time to be 150 for size check)
            with patch('time.monotonic', return_value=150):
                assert cache.size() == 1  # Only 'a' is valid

            # Verify expired items were cleaned up
            assert cache.get("b") is None
            assert cache.get("c") is None
            assert cache.get("a") == 1

    def test_all_expired_clear_on_put(self):
        """Test that all-expired cache is cleared before new put."""
        cache = TTLCache(capacity=3, default_ttl=10)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Simulate all items expiring
        with patch('time.monotonic', side_effect=[0, 0, 0, 11, 11, 11, 11]):
            # All items should be expired
            assert cache.get("a") is None
            assert cache.get("b") is None
            assert cache.get("c") is None

            # Now put new item - should clear expired and add
            cache.put("d", 4)

            assert cache.size() == 1
            assert cache.get("d") == 4
```

These tests use `unittest.mock.patch` to mock `time.monotonic` for deterministic time control. The implementation uses a doubly-linked list with a hash map for O(1) operations, and handles all the edge cases including expired items, capacity eviction, and lazy cleanup.