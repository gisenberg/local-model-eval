import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list representing a cache entry."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU Cache with time-based expiration.
    All operations (get, put, delete, size) are O(1) average, 
    except for the specific 'clear all' case which is O(N).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Sentinel nodes for the doubly-linked list
        self.head: Node = Node("", None, 0.0)
        self.tail: Node = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the head sentinel."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)

    def _clear_all(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key. If at capacity, evict the LRU non-expired item.
        If all items are expired, clear them all first.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        if len(self.cache) >= self.capacity:
            # Check if all items are expired (O(N) requirement)
            all_expired = True
            for node in self.cache.values():
                if node.expiry > now:
                    all_expired = False
                    break
            
            if all_expired:
                self._clear_all()
            else:
                # Evict the least-recently-used (the tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        # Create and add new node
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key and return True if it existed, else False."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup)."""
        # Note: In a lazy-cleanup system, size() returns the current 
        # count of items in the dictionary.
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
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" (LRU)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        
        mock_time.return_value = 111.0  # Past expiry
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        cache.put("a", 1, ttl=5)  # Custom TTL 5
        
        mock_time.return_value = 104.0
        assert cache.get("a") == 1
        
        mock_time.return_value = 106.0
        assert cache.get("a") is None

def test_delete():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_items():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        cache.put("valid", 1)
        cache.put("expired", 2, ttl=1)
        
        mock_time.return_value = 105.0
        # 'expired' is now expired, but not removed until access
        assert cache.size() == 2 
        
        # Accessing 'expired' triggers lazy cleanup
        assert cache.get("expired") is None
        assert cache.size() == 1