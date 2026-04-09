import time
from typing import Any, Optional, Dict


class Node:
    """Doubly linked list node storing cache entry data."""
    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live expiration.
    Uses a hash map + doubly-linked list for O(1) operations.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items to store.
        :param default_ttl: Default lifetime of items in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head = Node("", None, 0)  # Most Recently Used
        self.tail = Node("", None, 0)  # Least Recently Used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add a node to the head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item. Returns None if key doesn't exist or is expired.
        Moves accessed item to MRU position.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if self._is_expired(node):
            self.delete(key)
            return None

        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item. Evicts LRU item if at capacity.
        Custom ttl overrides default_ttl.
        """
        expiry_duration = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + expiry_duration

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._remove_node(node)
            self._add_to_head(node)
            return

        if len(self.cache) >= self.capacity:
            self._evict()

        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def _evict(self) -> None:
        """Evict the LRU item. If all items expired, clear all."""
        lru = self.tail.prev
        if lru == self.head:
            return

        if self._is_expired(lru):
            items = list(self.cache.keys())
            for k in items:
                self.delete(k)
        else:
            self.delete(lru.key)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if key existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup)."""
        current_time = time.monotonic()
        valid_count = 0
        expired_keys = []

        for key, node in self.cache.items():
            if current_time >= node.expiry:
                expired_keys.append(key)
            else:
                valid_count += 1

        for key in expired_keys:
            self.delete(key)

        return valid_count


# ==============================================================================
# TESTS
# ==============================================================================

import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10)


def test_basic_get_put(cache):
    """Test basic insertion and retrieval."""
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None


def test_lru_eviction(cache):
    """Test that the least recently used item is evicted when at capacity."""
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")  # a becomes MRU, b is now LRU
    cache.put("d", 4)  # should evict b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_ttl_expiry(cache):
    """Test that items expire after default_ttl."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)
        assert cache.get("a") == 1

        mock_time.return_value = 110.0  # Exactly at expiry
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_ttl(cache):
    """Test that custom TTL overrides default."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache.put("short", 1, ttl=2)
        cache.put("long", 2, ttl=20)

        mock_time.return_value = 103.0
        assert cache.get("short") is None
        assert cache.get("long") == 2


def test_delete(cache):
    """Test explicit deletion."""
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False


def test_size_with_mixed_expiry(cache):
    """Test size() with a mix of expired and valid items."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)  # expires 110
        cache.put("b", 2)  # expires 110
        cache.put("c", 3)  # expires 110

        mock_time.return_value = 105.0
        assert cache.size() == 3

        mock_time.return_value = 111.0
        assert cache.size() == 0  # All expired, lazy cleanup
        assert len(cache.cache) == 0