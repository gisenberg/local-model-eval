import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list representing a cache entry."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU Cache with time-based expiration.
    Uses a Doubly Linked List and a Hash Map to achieve O(1) operations.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of non-expired items allowed.
        :param default_ttl: Default seconds until an item expires.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail for the doubly linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node to the front (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Moves an existing node to the front."""
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node has passed its expiry time."""
        return time.monotonic() > node.expiry

    def _cleanup_expired(self) -> None:
        """Removes all expired nodes from the cache."""
        current = self.tail.prev
        while current != self.head:
            if self._is_expired(current):
                # We must capture the previous node before removing current
                prev_node = current.prev
                self.delete(current.key)
                current = prev_node
            else:
                current = current.prev

    def get(self, key: str) -> Optional[Any]:
        """
        Returns the value if key exists and is not expired.
        Updates the key to be most recently used.
        """
        node = self.cache.get(key)
        if not node:
            return None

        if self._is_expired(node):
            self.delete(key)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key. 
        Evicts LRU item if capacity is reached.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = selfcap_node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Check capacity and handle eviction
        self._cleanup_expired()
        
        if len(self.cache) >= self.capacity:
            # Evict the Least Recently Used (node before tail)
            lru_node = self.tail.prev
            if lru_node != self.head:
                self.delete(lru_node.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Removes a key from the cache. Returns True if it existed."""
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Returns the count of non-expired items."""
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
    """Test that the Least Recently Used item is evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_tll=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiry():
    """Test that items expire after the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put("a", 1)
        
        mock_time.return_value = 104.0
        assert cache.get("a") == 1 # Not expired yet
        
        mock_time.return_value = 106.0
        assert cache.get("a") is None # Expired

def test_custom_ttl():
    """Test that custom TTL overrides the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1, ttl=1) # Very short TTL
        
        mock_time.return_value = 105.0
        assert cache.get("a") is None # Should be expired

def test_delete():
    """Test the delete functionality."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_with_mixed_items():
    """Test size() correctly ignores expired items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        cache.put("a", 1) # Expires at 110
        cache.put("b", 2, ttl=20) # Expires at 120
        
        mock_time.return_value = 115.0
        # 'a' is expired, 'b' is valid
        assert cache.size() == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2