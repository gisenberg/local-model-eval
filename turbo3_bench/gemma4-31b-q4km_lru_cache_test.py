import time
from typing import Optional, Any

class Node:
    """Doubly linked list node to store cache entries."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Time Complexity: O(1) average for all operations.
    Space Complexity: O(capacity).
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default expiration time in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Map key -> Node
        
        # Dummy head and tail for the doubly linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Insert a node at the front (MRU position) of the list."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node, current_time: float) -> bool:
        """Check if a node has exceeded its TTL."""
        return current_time >= node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        current_time = time.monotonic()
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node, current_time):
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key. Evicts LRU non-expired item if at capacity.
        Custom ttl overrides default.
        """
        current_time = time.monotonic()
        expiry = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._remove_node(node)
            self._add_to_front(node)
            return

        # Handle capacity: Purge expired items from the tail first
        while self.tail.prev != self.head and self._is_expired(self.tail.prev, current_time):
            expired_node = self.tail.prev
            self._remove_node(expired_node)
            del self.cache[expired_node.key]

        # If still at capacity, evict the LRU (which is now guaranteed non-expired)
        if len(self.cache) >= self.capacity:
            lru_node = self.tail.prev
            if lru_node != self.head:
                self._remove_node(lru_node)
                del self.cache[lru_node.key]

        # Create and add new node
        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Return True if it existed."""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """
        Return count of items currently tracked. 
        Expired items are removed lazily on access.
        """
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

@patch('time.monotonic')
def test_basic_get_put(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

@patch('time.monotonic')
def test_capacity_eviction(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3) # Should evict "a" (LRU)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

@patch('time.monotonic')
def test_ttl_expiry(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=10, default_ttl=1.0)
    cache.put("a", 1)
    
    mock_time.return_value = 102.0 # Advance past TTL
    assert cache.get("a") is None

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=10, default_ttl=10.0)
    cache.put("short", 1, ttl=1.0)
    cache.put("long", 2, ttl=20.0)
    
    mock_time.return_value = 102.0
    assert cache.get("short") is None # Expired
    assert cache.get("long") == 2    # Still valid

@patch('time.monotonic')
def test_delete(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=10, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False

@patch('time.monotonic')
def test_size_mixed(mock_time):
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=10, default_ttl=10.0)
    cache.put("valid", 1, ttl=10.0)
    cache.put("expired", 2, ttl=1.0)
    
    mock_time.return_value = 102.0
    # Size is 2 because cleanup is lazy
    assert cache.size() == 2 
    # Accessing expired item triggers cleanup
    assert cache.get("expired") is None
    assert cache.size() == 1

if __name__ == "__main__":
    pytest.main([__file__])