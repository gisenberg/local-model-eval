import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list to store cache entries."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTL Cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to simplify DLL operations
        # Head is Most Recently Used (MRU), Tail is Least Recently Used (LRU)
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
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
        """Add a node immediately after the dummy head (MRU position)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the MRU position."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Updates the item to be the most recently used.
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
        Insert or update a value. Evicts LRU non-expired item if at capacity.
        If all items are expired, they are cleared first.
        """
        current_time = time.monotonic()
        expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._move_to_front(node)
            return

        # Eviction logic
        if len(self.cache) >= self.capacity:
            # 1. Lazy cleanup: remove expired items starting from the LRU (tail)
            while self.cache and self.tail.prev != self.head:
                lru_node = self.tail.prev
                if current_time > lru_node.expiry:
                    self.delete(lru_node.key)
                else:
                    break
            
            # 2. If still at capacity, evict the LRU non-expired item
            if len(self.cache) >= self.capacity:
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if key existed."""
        if key not in self.cache:
            return False
        
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Return count of items currently in the cache."""
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    """Test that the least recently used non-expired item is evicted."""
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' becomes MRU, 'b' becomes LRU
    cache.put("c", 3) # 'b' should be evicted
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1 # Not expired yet
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None # Expired

def test_custom_ttl():
    """Test that per-key TTL overrides the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10.0)
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        mock_time.return_value = 105.0
        assert cache.get("short") is None # Expired
        assert cache.get("long") == 2   # Still valid

def test_delete():
    """Test explicit deletion of keys."""
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False

def test_size_mixed_expiry():
    """Test size calculation with mixed expired and valid items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        mock_time.return_value = 111.0 # a, b, c all expired
        # size() returns current map length; lazy cleanup happens on access
        assert cache.size() == 3 
        
        # Accessing 'a' triggers lazy cleanup for 'a'
        assert cache.get("a") is None
        assert cache.size() == 2 # 'a' removed, 'b' and 'c' still in map
        
        # put() triggers cleanup of expired items from the tail
        cache.put("d", 4)
        # Since a, b, c were all expired, they should be cleared before adding d
        # if the logic follows "If all items are expired, clear them all first"
        assert cache.size() == 1 # Only 'd' remains