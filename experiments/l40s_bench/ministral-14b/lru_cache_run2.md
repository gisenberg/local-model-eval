# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — LRU Cache with TTL — Run 2

## Output

# LRU Cache with Time-Based Expiration

Here's a complete implementation of the TTLCache class with all the specified requirements:

```python
import time
from typing import Any, Optional, Dict, Tuple

class Node:
    """Node for doubly-linked list in TTLCache"""
    def __init__(self, key: str, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expiry_time = time.monotonic() + ttl
        self.prev = None
        self.next = None

class TTLCache:
    """LRU cache with time-based expiration (TTL)"""

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache with given capacity and default TTL.

        Args:
            capacity: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head = Node("", None, 0)  # dummy head
        self.tail = Node("", None, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        current = self.tail.prev
        while current != self.head:
            if time.monotonic() > current.expiry_time:
                self._remove_node(current)
                del self.cache[current.key]
                current = self.tail.prev
            else:
                break

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        self._cleanup_expired()

        if len(self.cache) >= self.capacity:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if it exists and isn't expired.

        Args:
            key: Key to look up

        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if time.monotonic() > node.expiry_time:
            self._remove_node(node)
            del self.cache[key]
            return None

        # Move to front (most recently used)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair with optional TTL.

        Args:
            key: Key to insert/update
            value: Value to store
            ttl: Time-to-live in seconds (optional, uses default if None)
        """
        ttl = ttl if ttl is not None else self.default_ttl

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl = ttl
            node.expiry_time = time.monotonic() + ttl
            self._remove_node(node)
            self._add_to_front(node)
        else:
            self._evict_lru()
            new_node = Node(key, value, ttl)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: Key to remove

        Returns:
            True if key existed and was removed, False otherwise
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """
        Get count of non-expired items in cache.

        Returns:
            Number of non-expired items
        """
        self._cleanup_expired()
        return len(self.cache)
```

## Test Cases

Here are the pytest tests covering all requirements:

```python
import pytest
from unittest.mock import patch
from your_module import TTLCache  # Replace with actual module name

def test_basic_get_put():
    """Test basic get and put operations"""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None

def test_capacity_eviction():
    """Test LRU eviction when capacity is reached"""
    cache = TTLCache(capacity=2, default_ttl=100)

    # Fill cache to capacity
    cache.put("a", 1)
    cache.put("b", 2)

    # Access 'a' to make it most recent
    cache.get("a")

    # Add new item, should evict 'b' (least recently used)
    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test TTL-based expiration"""
    with patch('time.monotonic', return_value=10):
        cache = TTLCache(capacity=2, default_ttl=5)

    # Put items at time=10
    cache.put("a", 1)

    # Fast forward time to 15 (5 seconds later)
    with patch('time.monotonic', return_value=15):
        assert cache.get("a") is None  # Should be expired

def test_custom_per_key_ttl():
    """Test custom TTL per key"""
    with patch('time.monotonic', return_value=10):
        cache = TTLCache(capacity=2, default_ttl=10)

    # Item with default TTL (10s)
    cache.put("a", 1)

    # Item with custom TTL (2s)
    cache.put("b", 2, ttl=2)

    # Fast forward time to 12 (2s later)
    with patch('time.monotonic', return_value=12):
        assert cache.get("a") == 1  # Still valid (default TTL)
        assert cache.get("b") is None  # Expired (custom TTL)

def test_delete():
    """Test key deletion"""
    cache = TTLCache(capacity=2, default_ttl=100)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("c") is False
    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_size_with_expired_items():
    """Test size() with mixed expired and valid items"""
    with patch('time.monotonic', return_value=10):
        cache = TTLCache(capacity=2, default_ttl=5)

    # Put items at time=10
    cache.put("a", 1)
    cache.put("b", 2)

    # Fast forward time to 15 (5 seconds later)
    with patch('time.monotonic', return_value=15):
        # Add new item, should evict expired 'a' first
        cache.put("c", 3)

        # Size should be 2 (b and c)
        assert cache.size() == 2

        # Fast forward to 20 (10 seconds later)
        with patch('time.monotonic', return_value=20):
            # Both remaining items should be expired
            assert cache.size() == 0
```

## Implementation Notes

1. **Data Structures**:
   - Uses a doubly-linked list for O(1) insertions/deletions
   - Uses a dictionary for O(1) lookups
   - Each node stores key, value, TTL, and expiry time

2. **Time Handling**:
   - Uses `time.monotonic()` for reliable time tracking
   - Expiry time is calculated as current time + TTL

3. **Lazy Cleanup**:
   - Expired items are only removed when accessed or when size is checked
   - This maintains O(1) average time complexity for all operations

4. **Eviction Policy**:
   - When capacity is reached, evicts the least recently used non-expired item
   - If all items are expired, clears them all first

5. **Type Hints**:
   - All methods include proper type hints as specified

The implementation meets all the requirements including O(1) average time complexity for all operations and proper handling of both LRU and TTL expiration.
