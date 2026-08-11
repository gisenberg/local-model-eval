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
        Initialize the cache.

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

    def _add_to_head(self, node: _Node) -> None:
        """Add a node to the front of the linked list (most recently used)."""
        node.prev = None
        node.next = self._head

        if self._head:
            self._head.prev = node
        else:
            self._tail = node

        self._head = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the front of the linked list."""
        if node is self._head:
            return

        self._remove_node(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if a node has expired."""
        return node.expires_at <= now

    def _evict_lru(self, now: float) -> None:
        """
        Evict the least recently used non-expired item.
        If all items are expired, clear them all.
        """
        # First, check if all items are expired
        all_expired = True
        current = self._head
        while current:
            if not self._is_expired(current, now):
                all_expired = False
                break
            current = current.next

        if all_expired:
            # Clear all expired items
            self._cache.clear()
            self._head = None
            self._tail = None
            return

        # Find the least recently used non-expired item
        current = self._tail
        while current:
            if not self._is_expired(current, now):
                # Evict this item
                self._cache.pop(current.key, None)
                self._remove_node(current)
                return
            current = current.prev

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Returns the value if the key exists and hasn't expired,
        otherwise returns None. Accessing a key makes it most-recently-used.
        """
        node = self._cache.get(key)
        if not node:
            return None

        now = time.monotonic()
        if self._is_expired(node, now):
            # Remove expired item
            self._cache.pop(key, None)
            self._remove_node(node)
            return None

        # Move to head (most recently used)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        If the cache is at capacity, evicts the least recently used
        non-expired item. Custom TTL overrides the default.
        """
        now = time.monotonic()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        # Check if key already exists
        existing = self._cache.get(key)
        if existing:
            existing.value = value
            existing.expires_at = expires_at
            self._move_to_head(existing)
            return

        # Check capacity
        if len(self._cache) >= self.capacity:
            self._evict_lru(now)

        # Create new node and add to cache
        node = _Node(key, value, expires_at)
        self._cache[key] = node
        self._add_to_head(node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Returns True if the key existed, False otherwise.
        """
        node = self._cache.pop(key, None)
        if node:
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.

        Performs lazy cleanup: expired items are removed during this call.
        """
        now = time.monotonic()
        expired_keys = []

        # Find expired items
        for key, node in self._cache.items():
            if self._is_expired(node, now):
                expired_keys.append(key)

        # Remove expired items
        for key in expired_keys:
            node = self._cache.pop(key, None)
            if node:
                self._remove_node(node)

        return len(self._cache)
```

Now here are the 6 pytest tests:

```python
import pytest
from unittest.mock import patch
from typing import Any
from ttl_cache import TTLCache


def test_basic_get_put():
    """Test basic get/put operations."""
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


def test_capacity_eviction_lru_order():
    """Test LRU eviction when at capacity."""
    cache = TTLCache(capacity=3, default_ttl=100)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    # Access 'a' to make it most recently used
    cache.get("a")

    # Add new item, should evict 'b' (least recently used)
    cache.put("d", 4)

    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_ttl_expiry():
    """Test TTL-based expiration."""
    cache = TTLCache(capacity=3, default_ttl=5)

    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)
        cache.put("b", 2)

        # Before expiry
        mock_time.return_value = 104.0
        assert cache.get("a") == 1

        # After expiry
        mock_time.return_value = 106.0
        assert cache.get("a") is None
        assert cache.get("b") is None

        # Cache should be empty after lazy cleanup
        assert cache.size() == 0


def test_custom_per_key_ttl():
    """Test custom TTL per key."""
    cache = TTLCache(capacity=3, default_ttl=10)

    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("short", 1, ttl=2)
        cache.put("long", 2, ttl=20)

        # Short TTL expires
        mock_time.return_value = 103.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

        # Long TTL still valid
        mock_time.return_value = 115.0
        assert cache.get("long") == 2

        # Long TTL expires
        mock_time.return_value = 121.0
        assert cache.get("long") is None


def test_delete():
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("a") is False  # Already deleted
    assert cache.delete("missing") is False

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.size() == 1


def test_size_with_mixed_expired_valid():
    """Test size with mixed expired and valid items."""
    cache = TTLCache(capacity=5, default_ttl=10)

    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)  # Will expire at 110
        cache.put("b", 2)  # Will expire at 110
        cache.put("c", 3, ttl=2)  # Will expire at 102
        cache.put("d", 4, ttl=20)  # Will expire at 120

        # At time 105: 'c' is expired, others valid
        mock_time.return_value = 105.0
        assert cache.size() == 3

        # At time 111: 'a' and 'b' expired, 'd' still valid
        mock_time.return_value = 111.0
        assert cache.size() == 1

        # At time 121: all expired
        mock_time.return_value = 121.0
        assert cache.size() == 0
```

These tests cover all the required functionality:
1. Basic get/put operations
2. Capacity eviction following LRU order
3. TTL expiry with mocked time
4. Custom per-key TTL
5. Delete operation
6. Size with mixed expired/valid items

The implementation uses a doubly-linked list with a hash map for O(1) operations, uses `time.monotonic()` for time tracking, and includes type hints and docstrings throughout.