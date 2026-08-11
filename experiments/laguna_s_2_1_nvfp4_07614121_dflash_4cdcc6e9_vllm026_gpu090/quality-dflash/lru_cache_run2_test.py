import time
from typing import Optional, Any, Dict, Tuple

class Node:
    def __init__(self, key: Any = None, value: Any = None, expire_time: float = 0):
        self.key = key
        self.value = value
        self.expire_time = expire_time
        self.prev = None
        self.next = None

class TTLCache:
    """
    LRU Cache with TTL support.
    Uses doubly-linked list + hash map for O(1) operations.
    """

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl < 0:
            raise ValueError("Default TTL must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node()  # Dummy head
        self.tail = Node()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove node from linked list."""
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add node right after head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move existing node to front."""
        self._remove_node(node)
        self._add_to_front(node)

    def _pop_tail(self) -> Node:
        """Remove and return the least recently used node."""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node

    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired."""
        return node.expire_time > 0 and time.monotonic() >= node.expire_time

    def _cleanup_expired(self) -> None:
        """Lazy cleanup of expired nodes."""
        current = self.head.next
        while current != self.tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                if current.key in self.cache:
                    del self.cache[current.key]
            current = next_node

    def get(self, key: Any) -> Any:
        """
        Get value by key. Returns -1 if not found or expired.
        Moves accessed item to front (most recently used).
        """
        if key not in self.cache:
            return -1

        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return -1

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put key-value pair into cache with optional TTL.
        If key exists, update value and move to front.
        If capacity exceeded, remove least recently used item.
        """
        # Lazy cleanup before any operation
        self._cleanup_expired()

        current_time = time.monotonic()
        expire_time = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_time = expire_time
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                lru_node = self._pop_tail()
                if lru_node.key in self.cache:
                    del self.cache[lru_node.key]

            new_node = Node(key, value, expire_time)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Delete key from cache. Returns True if deleted, False if not found.
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True

    def size(self) -> int:
        """Return current number of items in cache."""
        return len(self.cache)

# tests/test_ttl_cache.py
import pytest
from unittest.mock import patch
from typing import List
import time

# Import the TTLCache class

class TestTTLCache:

    @patch('time.monotonic')
    def test_basic_get_put(self, mock_time):
        """Test basic get and put operations."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10)

        cache.put(1, 'value1')
        assert cache.get(1) == 'value1'
        assert cache.size() == 1

        cache.put(2, 'value2')
        assert cache.get(2) == 'value2'
        assert cache.size() == 2

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_time):
        """Test LRU eviction when capacity is exceeded."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10)

        cache.put(1, 'value1')
        cache.put(2, 'value2')
        cache.get(1)  # Access key 1 to make it more recent
        cache.put(3, 'value3')  # Should evict key 2

        assert cache.get(1) == 'value1'
        assert cache.get(2) == -1  # Evicted
        assert cache.get(3) == 'value3'

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_time):
        """Test that items expire after TTL."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=5)

        cache.put(1, 'value1')
        assert cache.get(1) == 'value1'

        # Advance time beyond TTL
        mock_time.return_value = 1006.0
        assert cache.get(1) == -1  # Expired

    @patch('time.monotonic')
    def test_custom_ttl_per_item(self, mock_time):
        """Test custom TTL for individual items."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10)

        cache.put(1, 'value1', ttl=3)  # Custom TTL of 3 seconds
        cache.put(2, 'value2')  # Uses default TTL of 10 seconds

        # Advance time to 3 seconds - item 1 should expire
        mock_time.return_value = 1003.0
        assert cache.get(1) == -1  # Expired
        assert cache.get(2) == 'value2'  # Still valid

    @patch('time.monotonic')
    def test_delete_operation(self, mock_time):
        """Test delete operation."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10)

        cache.put(1, 'value1')
        cache.put(2, 'value2')

        assert cache.delete(1) is True
        assert cache.delete(1) is False  # Already deleted
        assert cache.get(1) == -1
        assert cache.size() == 1

    @patch('time.monotonic')
    def test_lazy_cleanup_on_access(self, mock_time):
        """Test that expired items are cleaned up lazily."""
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=5)

        cache.put(1, 'value1')
        cache.put(2, 'value2')
        cache.put(3, 'value3')

        # Advance time to expire all items
        mock_time.return_value = 1006.0

        # Accessing should trigger cleanup
        cache.get(1)
        assert cache.size() == 0  # All expired items cleaned up