Here's a complete implementation of a TTLCache with LRU eviction and time-based expiration:

```python
from typing import Any, Optional
import time


class _Node:
    """Doubly-linked list node for LRU tracking."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = None
        self.next = None


class TTLCache:
    """
    LRU cache with time-based expiration.

    Uses a doubly-linked list for O(1) LRU tracking and a hash map
    for O(1) key lookup. Expired items are lazily removed on access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with capacity and default TTL in seconds.

        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live in seconds for items
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

    def _add_to_front(self, node: _Node) -> None:
        """Add a node to the front (most recently used position)."""
        node.prev = None
        node.next = self._head

        if self._head:
            self._head.prev = node
        else:
            self._tail = node

        self._head = node

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front."""
        if node is self._head:
            return

        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if a node has expired."""
        return node.expiry <= now

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        current = self._tail
        now = time.monotonic()

        # Find the least recently used non-expired item
        while current and self._is_expired(current, now):
            current = current.prev

        if current:
            self._remove_node(current)
            del self._cache[current.key]

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        now = time.monotonic()
        expired_keys = [
            key for key, node in self._cache.items()
            if self._is_expired(node, now)
        ]

        for key in expired_keys:
            node = self._cache.pop(key)
            self._remove_node(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache if it exists and hasn't expired.

        Args:
            key: The key to look up

        Returns:
            The stored value, or None if key doesn't exist or has expired
        """
        node = self._cache.get(key)
        if not node:
            return None

        now = time.monotonic()
        if self._is_expired(node, now):
            # Remove expired item
            self._remove_node(node)
            del self._cache[key]
            return None

        # Move to front (most recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to store
            value: The value to store
            ttl: Optional custom TTL in seconds (overrides default)
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        # If key exists, update it
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Check capacity
        if len(self._cache) >= self.capacity:
            # Clean up expired items first
            self._cleanup_expired()

            # If still at capacity, evict LRU
            if len(self._cache) >= self.capacity:
                self._evict_lru()

        # Create new node and add to front
        node = _Node(key, value, expiry)
        self._cache[key] = node
        self._add_to_front(node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: The key to remove

        Returns:
            True if the key existed, False otherwise
        """
        node = self._cache.pop(key, None)
        if node:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """
        Get the number of non-expired items in the cache.

        Returns:
            Count of valid (non-expired) items
        """
        self._cleanup_expired()
        return len(self._cache)
```

Now here are the 6 pytest tests:

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def test_basic_get_put():
    """Test basic put and get operations."""
    cache = TTLCache(capacity=3, default_ttl=10)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("missing") is None

    # Update existing key
    cache.put("a", 100)
    assert cache.get("a") == 100


def test_capacity_eviction_lru_order():
    """Test LRU eviction when cache is at capacity."""
    cache = TTLCache(capacity=3, default_ttl=100)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    # Access "a" to make it most recently used
    assert cache.get("a") == 1

    # Add new item, should evict "b" (least recently used)
    cache.put("d", 4)

    assert cache.get("b") is None  # Evicted
    assert cache.get("a") == 1     # Still present
    assert cache.get("c") == 3     # Still present
    assert cache.get("d") == 4     # New item


def test_ttl_expiry():
    """Test that items expire after their TTL."""
    cache = TTLCache(capacity=3, default_ttl=5)

    cache.put("a", 1)

    # Mock time to simulate expiration
    with patch("time.monotonic", return_value=0):
        cache.put("b", 2)

    # Advance time beyond TTL
    with patch("time.monotonic", return_value=6):
        assert cache.get("a") is None  # Expired
        assert cache.get("b") == 2     # Still valid

    # Test that expired items are cleaned up
    with patch("time.monotonic", return_value=10):
        assert cache.size() == 1  # Only "b" remains


def test_custom_per_key_ttl():
    """Test custom TTL override for individual keys."""
    cache = TTLCache(capacity=3, default_ttl=10)

    with patch("time.monotonic", return_value=0):
        cache.put("short", 1, ttl=2)
        cache.put("long", 2, ttl=20)

    # Short TTL item expires
    with patch("time.monotonic", return_value=3):
        assert cache.get("short") is None
        assert cache.get("long") == 2

    # Long TTL item still valid
    with patch("time.monotonic", return_value=15):
        assert cache.get("long") == 2


def test_delete():
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("a") is False  # Already deleted
    assert cache.get("a") is None
    assert cache.get("b") == 2

    # Delete expired item
    with patch("time.monotonic", return_value=0):
        cache.put("c", 3)

    with patch("time.monotonic", return_value=11):
        assert cache.delete("c") is True  # Exists but expired


def test_size_with_mixed_expired_valid():
    """Test size with mixed expired and valid items."""
    cache = TTLCache(capacity=5, default_ttl=10)

    with patch("time.monotonic", return_value=0):
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

    # Advance time so some items expire
    with patch("time.monotonic", return_value=5):
        cache.put("d", 4)  # New item with TTL until 15

    with patch("time.monotonic", return_value=12):
        # "a", "b", "c" have expired (TTL was 10)
        # "d" is still valid (TTL until 15)
        assert cache.size() == 1
        assert cache.get("d") == 4
        assert cache.get("a") is None
```

These tests use `unittest.mock.patch` to control `time.monotonic()` for deterministic testing without using `time.sleep()`. The implementation uses a doubly-linked list with a hash map for O(1) operations, includes type hints throughout, and has docstrings on all methods.