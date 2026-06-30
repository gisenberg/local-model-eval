```python
import time
from typing import Any, Optional

class _Node:
    """Doubly-linked list node for LRU tracking."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """LRU cache with time-based expiration.

    Uses a doubly-linked list + hash map for O(1) average time operations.
    Expired items are lazily cleaned up on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")

        self._capacity = capacity
        self._default_ttl = default_ttl

        # Hash map: key -> Node
        self._map: dict[str, _Node] = {}

        # Doubly-linked list with sentinel head and tail
        # Head -> most recently used, Tail -> least recently used
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _now(self) -> float:
        """Return current monotonic time."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return self._now() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_expired_from_tail(self) -> None:
        """Evict all expired items starting from the tail (least recently used)."""
        current = self._tail.prev
        while current is not None and current is not self._head:
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
                current = current.prev
            else:
                break

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # First, evict any expired items from the tail
        self._evict_expired_from_tail()

        # If still at capacity, evict the LRU non-expired item
        if len(self._map) >= self._capacity:
            lru_node = self._tail.prev
            if lru_node is not None and lru_node is not self._head:
                self._remove_node(lru_node)
                del self._map[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """Get the value for a key if it exists and is not expired.

        Accessing a key makes it the most recently used.

        Args:
            key: The key to look up.

        Returns:
            The value if the key exists and is not expired, else None.
        """
        node = self._map.get(key)
        if node is None:
            return None

        if self._is_expired(node):
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to head (most recently used)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair in the cache.

        If the cache is at capacity, the least recently used non-expired item
        is evicted. If all items are expired, they are cleared first.

        Args:
            key: The key to insert or update.
            value: The value to associate with the key.
            ttl: Optional custom time-to-live in seconds. If None, uses default_ttl.
        """
        if ttl is None:
            ttl = self._default_ttl

        expires_at = self._now() + ttl

        # If key already exists, update it
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
            return

        # Evict if at capacity
        self._evict_lru()

        # Insert new node
        new_node = _Node(key, value, expires_at)
        self._map[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Args:
            key: The key to remove.

        Returns:
            True if the key existed (even if expired), False otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False

        self._remove_node(node)
        del self._map[key]
        return True

    def size(self) -> int:
        """Return the count of non-expired items in the cache.

        Expired items are lazily cleaned up during this operation.

        Returns:
            The number of non-expired items.
        """
        # Clean up expired items
        current = self._tail.prev
        while current is not None and current is not self._head:
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
                current = current.prev
            else:
                current = current.prev

        return len(self._map)
```

```python
import pytest
from unittest.mock import patch
from typing import Any

from lru_cache import TTLCache

@pytest.fixture
def mock_time(monkeypatch):
    """Fixture to mock time.monotonic for deterministic time control."""
    current_time = [0.0]

    def mock_monotonic():
        return current_time[0]

    monkeypatch.setattr('time.monotonic', mock_monotonic)
    return current_time

def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") is None

    # Update existing key
    cache.put("key1", "updated_value1")
    assert cache.get("key1") == "updated_value1"

def test_capacity_eviction_lru_order(mock_time):
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Access key1 to make it most recently used
    cache.get("key1")

    # Add key3, should evict key2 (least recently used)
    cache.put("key3", "value3")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None  # Evicted
    assert cache.get("key3") == "value3"

def test_ttl_expiry(mock_time):
    """Test that items expire after their TTL."""
    cache = TTLCache(capacity=3, default_ttl=5.0)

    cache.put("key1", "value1")

    # Advance time past TTL
    mock_time[0] = 6.0

    assert cache.get("key1") is None  # Expired

def test_custom_per_key_ttl(mock_time):
    """Test that custom per-key TTL overrides default TTL."""
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("key1", "value1", ttl=2.0)
    cache.put("key2", "value2")  # Uses default TTL of 10.0

    # Advance time past key1's TTL but not key2's
    mock_time[0] = 3.0

    assert cache.get("key1") is None  # Expired
    assert cache.get("key2") == "value2"  # Still valid

def test_delete(mock_time):
    """Test delete operation."""
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    assert cache.delete("key1") is True
    assert cache.delete("key1") is False  # Already deleted
    assert cache.delete("nonexistent") is False

    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"

def test_size_with_mixed_expired_valid_items(mock_time):
    """Test size() returns count of non-expired items with lazy cleanup."""
    cache = TTLCache(capacity=5, default_ttl=5.0)

    cache.put("key1", "value1", ttl=2.0)
    cache.put("key2", "value2", ttl=10.0)
    cache.put("key3", "value3", ttl=3.0)

    # Advance time past key1 and key3 TTLs
    mock_time[0] = 4.0

    # size() should lazily clean up expired items
    assert cache.size() == 1  # Only key2 is valid

    # Verify the expired items are actually removed
    assert cache.get("key1") is None
    assert cache.get("key3") is None
    assert cache.get("key2") == "value2"
```