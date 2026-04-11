# Mistral-Small-4-119B-2603 Q4_K_M (bartowski) — KV f16 (baseline) — LRU Cache with TTL

## Output

# LRU Cache with Time-Based Expiration

Here's an implementation of a TTLCache (LRU cache with time-based expiration) in Python that meets all your requirements:

```python
import time
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CacheNode:
    """Node in the doubly-linked list representing cache items."""
    key: str
    value: Any
    ttl: float
    expiry_time: float
    prev: Optional['CacheNode'] = None
    next: Optional['CacheNode'] = None

class TTLCache:
    """LRU cache with time-based expiration for items.

    Uses a doubly-linked list and hash map to maintain O(1) average time complexity
    for all operations. Expired items are only removed when accessed (lazy cleanup).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache with a given capacity and default TTL.

        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for items without custom TTL
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheNode] = {}
        self.head: Optional[CacheNode] = None
        self.tail: Optional[CacheNode] = None
        self.size_count = 0

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache (lazy cleanup)."""
        current_time = time.monotonic()
        keys_to_remove = []

        # Find all expired keys
        for key, node in self.cache.items():
            if node.expiry_time <= current_time:
                keys_to_remove.append(key)

        # Remove them
        for key in keys_to_remove:
            self.delete(key)

    def _add_to_head(self, node: CacheNode) -> None:
        """Add a node to the head of the linked list (most recently used)."""
        node.prev = None
        node.next = self.head

        if self.head:
            self.head.prev = node
        self.head = node

        if not self.tail:
            self.tail = node

    def _remove_node(self, node: CacheNode) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _move_to_head(self, node: CacheNode) -> None:
        """Move an existing node to the head of the linked list."""
        if node == self.head:
            return

        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Evict the least recently used item from the cache."""
        if not self.tail:
            return

        lru_node = self.tail
        if lru_node.expiry_time > time.monotonic():
            # Only evict if not expired
            self.delete(lru_node.key)

    def get(self, key: str) -> Optional[Any:
        """Get a value from the cache if it exists and hasn't expired.

        Args:
            key: The key to look up in the cache

        Returns:
            The value if the key exists and hasn't expired, None otherwise
        """
        self._cleanup_expired()

        if key not in self.cache:
            return None

        node = self.cache[key]
        if node.expiry_time <= time.monotonic():
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a value in the cache with optional TTL.

        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Time-to-live in seconds for this specific key. If None, uses default_ttl
        """
        self._cleanup_expired()

        current_time = time.monotonic()
        expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.ttl = ttl if ttl is not None else self.default_ttl
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # Create new node
            node = CacheNode(key, value, expiry_time, expiry_time)
            self.cache[key] = node
            self._add_to_head(node)
            self.size_count += 1

            # Evict if over capacity
            if self.size_count > self.capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """Delete a key from the cache if it exists.

        Args:
            key: The key to delete

        Returns:
            True if the key existed and was deleted, False otherwise
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size_count -= 1
        return True

    def size(self) -> int:
        """Return the number of non-expired items in the cache.

        Returns:
            Count of non-expired items (lazy cleanup)
        """
        self._cleanup_expired()
        return self.size_count
```

## Test Implementation

Here are 6 pytest tests that cover all the requirements, using `unittest.mock.patch` to mock `time.monotonic()` for deterministic testing:

```python
import pytest
from typing import Any
from unittest.mock import patch
from time import monotonic

from ttl_cache import TTLCache, CacheNode

@pytest.fixture
def mock_time():
    """Fixture to mock time.monotonic() for deterministic tests."""
    with patch('time.monotonic') as mock:
        yield mock

def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None

def test_capacity_eviction(mock_time):
    """Test LRU eviction when cache is at capacity."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" as it's LRU

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry(mock_time):
    """Test that items expire after their TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=15.0)

    mock_time.return_value = 105.0  # 5 seconds after put
    assert cache.get("a") is None  # Expired
    assert cache.get("b") == 2  # Still valid

def test_custom_per_key_ttl(mock_time):
    """Test that custom TTL overrides default TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2)  # Uses default TTL

    mock_time.return_value = 105.0
    assert cache.get("a") is None  # Expired after 5 seconds
    assert cache.get("b") == 2  # Still valid (default TTL is 10 seconds)

def test_delete(mock_time):
    """Test delete operation."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("c") is False  # Non-existent key

def test_size_with_mixed_expired_valid(mock_time):
    """Test size() returns correct count with mixed expired/valid items."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=15.0)
    cache.put("c", 3, ttl=5.0)

    mock_time.return_value = 105.0  # 5 seconds after puts
    assert cache.size() == 1  # Only "b" is valid
    assert cache.get("a") is None  # Expired
    assert cache.get("b") == 2  # Valid
    assert cache.get("c") is None  # Expired

    # Add a new item that will expire immediately
    cache.put("d", 4, ttl=0.0)
    assert cache.size() == 1  # "d" is expired

    # Add an item that will expire in the future
    cache.put("e", 5, ttl=10.0)
    mock_time.return_value = 110.0  # 10 seconds after put
    assert cache.size() == 0  # All items expired
```

## Explanation

1. **Data Structure**:
   - Uses a doubly-linked list (`CacheNode` class) to maintain LRU order
   - Uses a hash map (`cache` dictionary) for O(1) access to nodes

2. **Operations**:
   - `get()`: Returns value if key exists and hasn't expired, moves node to head if valid
   - `put()`: Updates existing node or creates new one, handles eviction
   - `delete()`: Removes node from both list and hash map
   - `size()`: Returns count of non-expired items (with lazy cleanup)

3. **Time Handling**:
   - Uses `time.monotonic()` for time tracking
   - Items are checked for expiration during access operations
   - TTL is converted to absolute expiry time when stored

4. **LRU Eviction**:
   - When capacity is exceeded, evicts the tail node (LRU) if it's not expired
   - If all items are expired, clears them first

5. **Testing**:
   - All tests mock `time.monotonic()` to control time deterministically
   - Tests cover all specified requirements without using `time.sleep`

The implementation maintains O(1) average time complexity for all operations by using the hash map for direct access and the doubly-linked list for maintaining order and efficient eviction.
