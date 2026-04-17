```python
import time
from typing import Any, Optional, Dict

class Node:
    """Doubly Linked List Node to store cache entries."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Combines a hash map for O(1) access and a doubly-linked list for O(1) LRU tracking.
    """
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to avoid null checks during pointer updates
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Removes a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node to the front (most recently used) of the list."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: str) -> Optional[Any]:
        """Return value if exists and not expired, else None. Updates LRU position."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None
        
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert/update item. Evicts LRU non-expired item if capacity is reached."""
        expiry_time = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            self._remove(self.cache[key])
        
        new_node = Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_front(new_node)

        if len(self.cache) > self.capacity:
            # Evict the least recently used item (from the tail)
            # We iterate from tail to find the first non-expired item to evict
            # If all are expired, they will be cleaned up by size() or get()
            curr = self.tail.prev
            while curr != self.head:
                # If we found an item, evict it and break
                self.delete(curr.key)
                break
                curr = curr.prev

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed."""
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items. Performs lazy cleanup."""
        now = time.monotonic()
        # We must iterate keys to check expiry; to keep O(1) average, 
        # we only count valid items.
        valid_count = 0
        keys_to_delete = []
        
        for key, node in self.cache.items():
            if now > node.expiry:
                keys_to_delete.append(key)
            else:
                valid_count += 1
        
        for key in keys_to_delete:
            self.delete(key)
            
        return valid_count

# --- Tests ---
import pytest
from unittest.mock import patch

def test_basic_get_put():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction():
    # Capacity 2, items added: a, b, c. 'a' should be evicted.
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3) # Evicts 'a'
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 10) # TTL = 10s
        cache.put("a", 1)
        
        mock_time.return_value = 105.0
        assert cache.get("a") == 1 # Not expired yet
        
        mock_time.return_value = 111.0
        assert cache.get("a") is None # Expired

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(2, 100) # Default 100s
        cache.put("short", 1, ttl=1) # Custom 1s
        
        mock_time.return_value = 102.0
        assert cache.get("short") is None # Expired despite default TTL

def test_delete():
    cache = TTLCache(2, 10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_mixed_expiry():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(5, 10)
        cache.put("a", 1, ttl=1)  # expires at 101
        cache.put("b", 2, ttl=20) # expires at 120
        cache.put("c", 3, ttl=30) # expires at 130
        
        mock_time.return_value = 105.0
        # 'a' is expired, 'b' and 'c' are valid
        assert cache.size() == 2
        assert "a" not in cache.cache
```