import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list to track LRU order and expiration."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL Cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to avoid null checks during node manipulation
        self.head = Node("", None, 0.0)  # Most Recently Used (MRU)
        self.tail = Node("", None, 0.0) # Least Recently Used (LRU)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Unlinks a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node immediately after the dummy head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Checks if a node's expiry time has passed."""
        return time.monotonic() > node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value. Returns None if key doesn't exist or is expired.
        Accessing a key updates it to the most-recently-used position.
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
        Insert or update a value. Evicts LRU non-expired items if at capacity.
        Custom ttl overrides the default_ttl.
        """
        if key in self.cache:
            self.delete(key)

        # Handle capacity and eviction
        while len(self.cache) >= self.capacity:
            # Check the LRU item (at the tail)
            lru_node = self.tail.prev
            if lru_node == self.head: # Cache is effectively empty
                break
            
            # If the LRU item is expired, remove it and keep looking for space
            if self._is_expired(lru_node):
                self.delete(lru_node.key)
            else:
                # If the LRU item is NOT expired, we must evict it to make room
                self.delete(lru_node.key)
                break

        # Calculate expiry and insert
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if the key existed."""
        if key not in self.cache:
            return False
        
        node = self.cache.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return the count of items currently in the cache.
        Note: Due to lazy cleanup, this includes items that are expired but not yet accessed.
        """
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, no expiration for this test
    cache = TTLCache(2, 100.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")    # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        
        # Advance time by 5s (not expired)
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Advance time by 11s (expired)
        mock_time.return_value = 111.0
        assert cache.get("a") is None

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100.0) # Long default
        cache.put("short", 1, ttl=1.0)
        cache.put("long", 2, ttl=10.0)
        
        mock_time.return_value = 102.0
        assert cache.get("short") is None # Expired
        assert cache.get("long") == 2    # Still valid

def test_delete():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("nonexistent") is False

def test_size_and_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # a and b expire, c remains
        mock_time.return_value = 111.0
        # size() is lazy, so it still reports 3 until items are accessed
        assert cache.size() == 3
        
        # Accessing 'a' triggers lazy cleanup
        cache.get("a") 
        assert cache.size() == 2
        
        # Accessing 'b' triggers lazy cleanup
        cache.get("b")
        assert cache.size() == 1
        
        # 'c' is still there
        assert cache.get("c") == 3
        assert cache.size() == 1