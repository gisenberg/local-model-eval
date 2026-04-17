```python
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

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front (most recently used) of the list."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node
        self.tail.prev = node if not self.tail.prev == self.head else self.tail

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front."""
        self._remove(node)
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
        """Insert or update item. Evicts LRU if capacity reached."""
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
            # Try to find an expired item first to clear space
            # If none expired, evict the LRU (tail.prev)
            curr = self.tail.prev
            evicted = False
            while curr != self.head:
                if now > curr.expiry:
                    self.delete(curr.key)
                    evicted = True
                    break
                curr = curr.prev
            
            if not evicted:
                # Evict the absolute least recently used
                lru_key = self.tail.prev.key
                self.delete(lru_key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items."""
        now = time.monotonic()
        count = 0
        # We must iterate to check expiry for size, but we only remove 
        # items lazily during get/put to maintain O(1) for those.
        # To keep size() accurate to "non-expired", we check all.
        for node in self.cache.values():
            if now <= node.expiry:
                count += 1
        return count

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, no expiry for this test
    cache = TTLCache(2, 100.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a") # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10.0) # Expires at 110.0
        cache.put("a", 1)
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100.0)
        cache.put("short", 1, ttl=5.0) # Expires 105.0
        cache.put("long", 2, ttl=50.0) # Expires 150.0
        
        mock_time.return_value = 110.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete():
    cache = TTLCache(2, 10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10.0)
        cache.put("a", 1) # Exp 110
        cache.put("b", 2, ttl=2.0) # Exp 102
        cache.put("c", 3, ttl=20.0) # Exp 120
        
        mock_time.return_value = 105.0
        # 'a' and 'c' are valid, 'b' is expired
        assert cache.size() == 2
```