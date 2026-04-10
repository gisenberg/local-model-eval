import time
from typing import Any, Optional


class Node:
    """A node in the doubly-linked list representing a cache entry."""

    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None


class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    Uses a doubly-linked list and a hash map to ensure O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of non-expired items allowed.
        :param default_ttl: Default time in seconds until an item expires.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: dict[str, Node] = {}

        # Dummy head and tail nodes to simplify DLL operations
        self.head: Node = Node("", None, 0.0)
        self.tail: Node = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Detach a node from its neighbors in the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Insert a node immediately after the dummy head (most-recently-used)."""
        first_node = self.head.next
        node.next = first_node
        node.prev = self.head
        self.head.next = node
        if first_node:
            first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front of the list."""
        self._remove_node(node)
        self._add_to_front(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Return the value if the key exists and is not expired, else None.
        Accessing a key makes it the most-recently-used.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if node.expiry < time.monotonic():
            self.delete(key)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key with a specific value and optional TTL.
        If at capacity, evicts the LRU non-expired item.
        If all items are expired, clears them all first.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Handle capacity and expiration
        if len(self.cache) >= self.capacity:
            # 1. Remove all expired items starting from the tail (LRU)
            while self.tail.prev != self.head and self.tail.prev.expiry < now:
                expired_node = self.tail.prev
                self._remove_node(expired_node)
                del self.cache[expired_node.key]

            # 2. If still at capacity, evict the current LRU (which is now non-expired)
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]

        # Insert the new node
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if it existed."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of items currently in the cache.
        Note: Expired items are removed lazily on access.
        """
        return len(self.cache)


# --- Tests ---

import pytest
from unittest.mock import patch


def test_basic_get_put():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.size() == 2


def test_capacity_eviction():
    # Capacity 2, add 3 items. 'a' should be evicted.
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        
        # Advance time past TTL
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100) # Long default
        
        # Custom short TTL
        cache.put("short", "val", ttl=5)
        # Default long TTL
        cache.put("long", "val")
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None
        assert cache.get("long") == "val"


def test_delete():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None
    assert cache.size() == 0


def test_size_mixed_expired_valid():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        
        cache.put("valid", 1)      # expires 110
        cache.put("expired", 2, 5)  # expires 105
        
        # At time 106, 'expired' is technically expired but not yet removed via lazy cleanup
        mock_time.return_value = 106.0
        
        # size() returns len(cache), which includes the expired item until accessed
        assert cache.size() == 2
        
        # Accessing 'expired' triggers lazy cleanup
        assert cache.get("expired") is None
        assert cache.size() == 1
        assert cache.get("valid") == 1