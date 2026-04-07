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
        self.head: Optional[Node] = None  # Most Recently Used
        self.tail: Optional[Node] = None  # Least Recently Used

    def _remove_node(self, node: Node):
        """Internal helper to remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _add_to_front(self, node: Node):
        """Internal helper to add a node to the head (MRU)."""
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node

    def get(self, key: str) -> Optional[Any]:
        """Return value if exists and not expired, else None. Updates LRU position."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None
        
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Insert or update a key. Evicts LRU if capacity is reached."""
        current_time = time.monotonic()
        expiry = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._remove_node(node)
            self._add_to_front(node)
        else:
            # Eviction logic
            if len(self.cache) >= self.capacity:
                # Try to find an expired item to remove first
                evicted = False
                # Check from tail (LRU) to head to see if any are expired
                curr = self.tail
                while curr:
                    if current_time > curr.expiry:
                        self.delete(curr.key)
                        evicted = True
                        break
                    curr = curr.prev
                
                # If no expired items, evict the absolute LRU (tail)
                if not evicted and self.tail:
                    self.delete(self.tail.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup)."""
        current_time = time.monotonic()
        # To maintain O(1) average, we don't iterate the whole list here.
        # However, the prompt asks for count of non-expired items.
        # To strictly follow "lazy cleanup", we filter the current map.
        # Note: In a production system, size() might be approximate or trigger a cleanup.
        count = 0
        for key, node in list(self.cache.items()):
            if current_time > node.expiry:
                self.delete(key)
            else:
                count += 1
        return count

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, put 3 items. 'a' should be evicted as LRU.
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # 'a' becomes MRU, 'b' is now LRU
    cache.put("c", 3) # 'b' should be evicted
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10) # TTL = 10s
        cache.put("a", 1)
        
        mock_time.return_value = 111.0 # Past expiry
        assert cache.get("a") is None

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10)
        cache.put("short", 1, ttl=2)
        cache.put("long", 2, ttl=100)
        
        mock_time.return_value = 105.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False

def test_size_mixed():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3, ttl=1) # expires at 101
        
        mock_time.return_value = 102.0
        # 'c' is expired, 'a' and 'b' are valid
        assert cache.size() == 2