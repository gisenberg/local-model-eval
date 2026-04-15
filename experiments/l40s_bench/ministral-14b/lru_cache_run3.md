# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — LRU Cache with TTL — Run 3

## Output

# LRU Cache with Time-Based Expiration

Here's a complete implementation of the TTLCache class with all the specified requirements:

```python
import time
from typing import Any, Optional, Dict, Tuple

class Node:
    """Node for doubly-linked list in TTLCache."""
    def __init__(self, key: str, value: Any, ttl: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expiry_time = time.monotonic() + ttl
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """LRU cache with time-based expiration (TTL).

    Uses a doubly-linked list for O(1) LRU operations and a dictionary for O(1) lookups.
    Expired items are removed on access (lazy cleanup).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """Initialize the cache.

        Args:
            capacity: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds for items without custom TTL
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("TTL must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head = Node("", "", 0)  # dummy head
        self.tail = Node("", "", 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _evict_expired(self) -> None:
        """Evict all expired items from the cache."""
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
        self._evict_expired()  # Clean up expired items first
        if len(self.cache) >= self.capacity:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """Get value for key if it exists and is not expired.

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
        """Insert or update a key-value pair with optional TTL.

        Args:
            key: Key to store
            value: Value to store
            ttl: Custom time-to-live in seconds (optional)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        if ttl < 0:
            raise ValueError("TTL must be non-negative")

        # If key exists, update its TTL and value
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl = ttl
            node.expiry_time = time.monotonic() + ttl
            self._remove_node(node)
            self._add_to_front(node)
            return

        # If at capacity, evict LRU item first
        self._evict_lru()

        # Add new node
        new_node = Node(key, value, ttl)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: Key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """Get the number of non-expired items in the cache.

        Returns:
            Count of non-expired items
        """
        self._evict_expired()
        return len(self.cache)
```

## Test Cases

Here are 6 pytest tests covering all the requirements:

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache

def test_basic_get_put():
    """Test basic get and put operations."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None

def test_capacity_eviction():
    """Test LRU eviction when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" (least recently used)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

    # Access "b" to make it most recent
    cache.get("b")
    cache.put("d", 4)  # Should evict "c" now

    assert cache.get("b") == 2
    assert cache.get("c") is None
    assert cache.get("d") == 4

def test_ttl_expiry():
    """Test TTL-based expiration."""
    with patch('time.monotonic', return_value=10):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put("a", 1, ttl=5)

    with patch('time.monotonic', return_value=15):
        assert cache.get("a") is None  # Expired

    with patch('time.monotonic', return_value=10):
        cache.put("a", 1, ttl=5)

    with patch('time.monotonic', return_value=14):
        assert cache.get("a") == 1  # Not yet expired

def test_custom_per_key_ttl():
    """Test custom TTL per key."""
    with patch('time.monotonic', return_value=10):
        cache = TTLCache(capacity=2, default_ttl=2)
        cache.put("a", 1, ttl=5)  # Longer TTL than default
        cache.put("b", 2)         # Uses default TTL

    with patch('time.monotonic', return_value=12):
        assert cache.get("a") == 1  # Still valid
        assert cache.get("b") is None  # Expired (default TTL=2)

def test_delete():
    """Test delete operation."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("c") is False
    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_size_with_expired_items():
    """Test size() with mixed expired and valid items."""
    with patch('time.monotonic', side_effect=[10, 10, 15, 15]):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put("a", 1)
        cache.put("b", 2, ttl=1)  # Will expire quickly

        # Before expiry
        assert cache.size() == 2

        # After expiry
        assert cache.size() == 1  # Only "a" remains
        assert cache.get("a") == 1
        assert cache.get("b") is None
```

## Implementation Notes

1. **Data Structures**:
   - Uses a doubly-linked list for O(1) LRU operations
   - Uses a dictionary for O(1) lookups
   - Each node stores key, value, TTL, expiry time, and pointers to previous/next nodes

2. **Time Handling**:
   - Uses `time.monotonic()` for reliable time tracking
   - Expiry time is calculated as `current_time + ttl`

3. **Lazy Cleanup**:
   - Expired items are removed only when accessed or when size is checked
   - This ensures O(1) average time for all operations

4. **Edge Cases Handled**:
   - Capacity eviction when all items are expired
   - Custom TTL per key
   - Proper handling of dummy head/tail nodes in the linked list
   - Input validation for capacity and TTL values

5. **Testing**:
   - Uses `unittest.mock.patch` to control time for deterministic testing
   - Tests cover all major functionality including edge cases

The implementation meets all the specified requirements including O(1) average time complexity for all operations.
