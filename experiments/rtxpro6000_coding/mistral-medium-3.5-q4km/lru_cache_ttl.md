Here's a complete implementation of an LRU cache with time-based expiration in Python, along with pytest tests:

```python
import time
from typing import Any, Optional, Dict

class Node:
    """Doubly-linked list node for LRU cache."""
    __slots__ = ['key', 'value', 'expire_time', 'prev', 'next']

    def __init__(self, key: str, value: Any, expire_time: float):
        self.key = key
        self.value = value
        self.expire_time = expire_time
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    """LRU cache with time-based expiration."""

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head: Optional[Node] = None  # Most recently used
        self.tail: Optional[Node] = None  # Least recently used

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add a node to the head of the doubly-linked list."""
        node.prev = None
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node

    def _move_to_head(self, node: Node) -> None:
        """Move a node to the head of the list (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> Optional[Node]:
        """Remove and return the tail node (least recently used)."""
        if not self.tail:
            return None

        node = self.tail
        self._remove_node(node)
        return node

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache (lazy cleanup)."""
        current = self.head
        while current:
            next_node = current.next
            if current.expire_time <= time.monotonic():
                del self.cache[current.key]
                self._remove_node(current)
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for a key if it exists and hasn't expired.

        Args:
            key: The key to look up

        Returns:
            The value if found and not expired, None otherwise
        """
        node = self.cache.get(key)
        if not node:
            return None

        if node.expire_time <= time.monotonic():
            del self.cache[key]
            self._remove_node(node)
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert/update
            value: The value to associate with the key
            ttl: Optional time-to-live in seconds (overrides default_ttl)
        """
        # First cleanup any expired items
        self._cleanup_expired()

        # If key exists, update it
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        # If at capacity, evict the least recently used non-expired item
        if len(self.cache) >= self.capacity:
            # Find the first non-expired node from the tail
            node_to_remove = self.tail
            while node_to_remove and node_to_remove.expire_time <= time.monotonic():
                node_to_remove = node_to_remove.prev

            if node_to_remove:
                del self.cache[node_to_remove.key]
                self._remove_node(node_to_remove)
            else:
                # All items are expired, clear them all
                self.cache.clear()
                self.head = None
                self.tail = None

        # Add new node
        expire_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expire_time)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete

        Returns:
            True if the key existed and was deleted, False otherwise
        """
        node = self.cache.get(key)
        if not node:
            return False

        del self.cache[key]
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Get the number of non-expired items in the cache.

        Returns:
            Count of non-expired items
        """
        self._cleanup_expired()
        return len(self.cache)
```

And here are the pytest tests:

```python
import pytest
from unittest.mock import patch
from your_module import TTLCache  # Replace with actual import path

class TestTTLCache:
    @patch('time.monotonic')
    def test_basic_get_put(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 1.0]  # Times for init, put, get
        cache = TTLCache(capacity=2, default_ttl=10.0)

        # Test
        cache.put("a", 1)
        result = cache.get("a")

        # Assert
        assert result == 1
        assert cache.size() == 1

    @patch('time.monotonic')
    def test_capacity_eviction(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 0.0, 0.0, 1.0]  # Times for init and puts
        cache = TTLCache(capacity=2, default_ttl=10.0)

        # Test
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a" (LRU)

        # Assert
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2

    @patch('time.monotonic')
    def test_ttl_expiry(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 11.0]  # Times for init, put, get
        cache = TTLCache(capacity=2, default_ttl=10.0)

        # Test
        cache.put("a", 1)
        result = cache.get("a")  # Should be expired

        # Assert
        assert result is None
        assert cache.size() == 0

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 5.0]  # Times for init, put, get
        cache = TTLCache(capacity=2, default_ttl=10.0)

        # Test
        cache.put("a", 1, ttl=3.0)  # Custom TTL of 3 seconds
        result = cache.get("a")  # Should be expired (5.0 > 0.0 + 3.0)

        # Assert
        assert result is None
        assert cache.size() == 0

    @patch('time.monotonic')
    def test_delete(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 0.0]  # Times for init and puts
        cache = TTLCache(capacity=2, default_ttl=10.0)

        # Test
        cache.put("a", 1)
        result = cache.delete("a")
        deleted_again = cache.delete("a")

        # Assert
        assert result is True
        assert deleted_again is False
        assert cache.size() == 0

    @patch('time.monotonic')
    def test_size_with_mixed_items(self, mock_monotonic):
        # Setup
        mock_monotonic.side_effect = [0.0, 0.0, 0.0, 0.0, 15.0]  # Times for init, puts, size
        cache = TTLCache(capacity=3, default_ttl=10.0)

        # Test
        cache.put("a", 1, ttl=5.0)  # Expires at 5.0
        cache.put("b", 2, ttl=20.0)  # Expires at 20.0
        cache.put("c", 3)  # Expires at 10.0 (default)
        size = cache.size()  # At time 15.0, only "b" should be valid

        # Assert
        assert size == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") is None
```

Key implementation notes:
1. Uses a doubly-linked list (Node class) and a dictionary (cache) for O(1) operations
2. Time tracking uses time.monotonic() as required
3. Lazy cleanup of expired items happens during get(), put(), and size() operations
4. When at capacity, it finds the first non-expired LRU item to evict
5. If all items are expired, it clears them all before adding new items
6. Custom TTL overrides the default TTL for individual items

The tests cover all the required scenarios:
- Basic get/put operations
- Capacity eviction in LRU order
- TTL expiration
- Custom per-key TTL
- Delete operation
- Size with mixed expired/valid items

All tests use unittest.mock.patch to control time.monotonic() for deterministic testing without using time.sleep.