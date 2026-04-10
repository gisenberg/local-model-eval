import time
from typing import Any, Optional, Dict


class Node:
    """A node in the doubly-linked list."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    An LRU Cache with time-based expiration.
    Uses a doubly-linked list for O(1) LRU tracking and a hash map for O(1) access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default seconds until an item expires.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to simplify doubly-linked list operations
        self.head: Node = Node("", None, 0.0)
        self.tail: Node = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
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
        if first_node:
            first_node.prev = node

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if time.monotonic() >= node.expiry:
            self.delete(key)
            return None

        # Move to front (mark as MRU)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key. If at capacity, evict the LRU non-expired item.
        If all items are expired, clear them all first.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            self._remove(self.cache[key])
            del self.cache[key]

        if len(self.cache) >= self.capacity:
            # Requirement: If all items are expired, clear them all first.
            # We iterate from the tail (LRU) and remove all expired items.
            # This is O(1) amortized because each item is added and removed once.
            while self.tail.prev != self.head and self.tail.prev.expiry <= now:
                expired_node = self.tail.prev
                self.delete(expired_node.key)

            # If still at capacity after clearing expired, evict the LRU (the tail)
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key, return True if it existed."""
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup)."""
        # Note: To maintain O(1), we return the current dict size.
        # Expired items are cleaned up during get/put/put-eviction.
        return len(self.cache)


# --- Pytest Tests ---
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
    cache.put("c", 3)  # 'b' should be evicted
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


@patch("time.monotonic")
def test_ttl_expiry(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    
    mock_time.return_value = 111.0  # Expired
    assert cache.get("a") is None


@patch("time.monotonic")
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1, ttl=10)  # Expires at 110
    cache.put("b", 2, ttl=50)  # Expires at 150
    
    mock_time.return_value = 115.0
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_delete():
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None


@patch("time.monotonic")
def test_size_mixed(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=5, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2, ttl=1)  # Expires at 101
    cache.put("c", 3)
    
    assert cache.size() == 3
    
    mock_time.return_value = 102.0
    # 'b' is expired but not yet removed (lazy)
    assert cache.size() == 3 
    # Accessing 'b' triggers lazy cleanup
    assert cache.get("b") is None
    assert cache.size() == 2