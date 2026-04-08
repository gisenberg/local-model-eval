import time
from typing import Any, Optional, Dict

class Node:
    """Doubly linked list node to store cache entries."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """Initialize cache with max capacity and default time-to-live in seconds."""
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail for the doubly linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node
        self.tail.prev = self.tail # Ensure tail is maintained if list was empty

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front of the list."""
        self._remove_node(node)
        self._add_to_front(node)

    def get(self, key: str) -> Optional[Any]:
        """Return value if exists and not expired, else None. Updates LRU position."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None
        
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update item. Evicts LRU non-expired item if capacity is reached."""
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
            return

        # Handle capacity
        if len(self.cache) >= self.capacity:
            # First, try to clear all expired items to make room
            expired_keys = [k for k, n in self.cache.items() if now > n.expiry]
            for k in expired_keys:
                self.delete(k)

            # If still at capacity, evict the LRU (node before tail)
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        # Correcting the head/tail logic for the first insertion
        self._add_to_front(new_node)
        # Fix tail pointer if it was just initialized
        self.tail.prev = self.cache[key] if len(self.cache) == 1 else self.tail.prev 
        # Simplified: just ensure tail is always the end
        curr = self.head
        while curr.next: curr = curr.next
        self.tail = curr

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items."""
        now = time.monotonic()
        # Lazy cleanup: remove expired items found during size check
        expired_keys = [k for k, n in self.cache.items() if now > n.expiry]
        for k in expired_keys:
            self.delete(k)
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, no expiry for now
    cache = TTLCache(2, 100)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10) # TTL = 10s
        cache.put("a", 1)
        
        mock_time.return_value = 111.0 # Advance 11s
        assert cache.get("a") is None

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100) # Default long TTL
        cache.put("short", 1, ttl=5)
        cache.put("long", 2) # Uses 100s
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_with_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10)
        cache.put("a", 1) # expires 110
        cache.put("b", 2) # expires 110
        cache.put("c", 3, ttl=2) # expires 102
        
        mock_time.return_value = 105.0
        # 'c' is expired, 'a' and 'b' are valid
        assert cache.size() == 2