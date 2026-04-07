import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly linked list."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Uses a Doubly Linked List and a Hash Map for O(1) operations.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of non-expired items allowed.
        :param default_ttl: Default seconds until an item expires.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        self.head: Optional[Node] = None  # Most Recently Used
        self.tail: Optional[Node] = None  # Least Recently Used

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Adds a node to the head (MRU position)."""
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node

    def _cleanup_expired(self) -> None:
        """Removes all expired items from the cache."""
        now = time.monotonic()
        # We iterate from tail (LRU) because older items are more likely to be expired
        curr = self.tail
        while curr and curr.expiry <= now:
            prev_node = curr.prev
            self.delete(curr.key)
            curr = prev_node

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value if exists and not expired. Moves key to MRU.
        :param key: The key to look up.
        :return: Value or None if expired/missing.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None
        
        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item.
        :param key: Key to insert.
        :param value: Value to store.
        :param ttl: Custom TTL in seconds.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        # If key exists, remove old node first
        if key in self.cache:
            self.delete(key)

        # Check capacity: first clear all expired items
        self._cleanup_expired()

        # If still at capacity, evict LRU (tail)
        if len(self.cache) >= self.capacity:
            if self.tail:
                self.delete(self.tail.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if it existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """Returns count of non-disposed, non-expired items."""
        self._cleanup_expired()
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1) # Expires at 110.0
        
        mock_time.returnly_value = 111.0
        assert cache.get("a") is None

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("short", 1, ttl=2) # Expires at 102.0
        cache.put("long", 2)        # Expires at 110.0
        
        mock_time.return_value = 105.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

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
        cache.put("a", 1) # Exp 110
        cache.put("b", 2) # Exp 110
        
        mock_time.return_value = 115.0
        cache.put("c", 3) # Exp 125
        
        # At 115, 'a' and 'b' are expired. Only 'c' is valid.
        assert cache.size() == 1
        assert "a" not in cache.cache