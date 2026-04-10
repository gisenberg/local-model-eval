import time
from typing import Any, Optional, Dict


class Node:
    """A node in the doubly linked list."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Uses a Doubly Linked List for O(1) LRU management and a Hash Map for O(1) access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of non-expired items allowed.
        :param default_ttl: Default seconds until an item expires.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail for the doubly linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front (most recently used) of the list."""
        first_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = first_node
        first_node.prev = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        if self._is_expired(node):
            self.delete(key)
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key. If capacity is reached, evict LRU non-expired item.
        If all items are expired, clear them all.
        """
        expiry_duration = ttl if ttl is not None else self.default_ttl
        expiry_time = time.monotonic() + expiry_duration

        if key in self.cache:
            self.delete(key)

        # Check capacity and handle eviction
        if len(self.cache) >= self.capacity:
            # Try to find the first non-expired item from the tail (LRU)
            curr = self.tail.prev
            found_evictable = False
            
            # Attempt to find an expired item first to clear space efficiently
            # or simply the LRU item.
            while curr != self.head:
                if self._is_expired(curr):
                    self.delete(curr.key)
                    found_evictable = True
                    break
                curr = curr.prev
            
            # If no expired items found, evict the LRU (the tail.prev)
            if not found_evictable and self.cache:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        # If after cleanup we are still at capacity (edge case: all items expired but not cleared)
        # The requirement says: "If all items are expired, clear them all first."
        if len(self.cache) >= self.capacity:
            # Force clear all expired
            keys_to_del = [k for k, v in self.cache.items() if self._is_expired(v)]
            for k in keys_to_del:
                self.delete(k)

        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key and return True if it existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Return count of non-expired items."""
        # Lazy cleanup: remove expired items while counting
        expired_keys = [k for k, node in self.cache.items() if self._is_expired(node)]
        for k in expired_keys:
            self.delete(k)
        return len(self.cache)


# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_lru_eviction():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3)  # Should evict 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)  # Expires at 110.0
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("short", "val", ttl=5)  # Expires at 105.0
        cache.put("long", "val", ttl=20)   # Expires at 120.0
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None
        assert cache.get("long") == "val"

def test_delete():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        cache.put("valid", 1)
        cache.put("expired", 2, ttl=5)
        
        mock_time.return_value = 110.0
        # 'expired' is now expired, 'valid' is still valid
        assert cache.size() == 1
        assert "expired" not in cache.cache # size() performs lazy cleanup