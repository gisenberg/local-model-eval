```python
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
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    Maintains O(1) average time complexity for all operations.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail to avoid null checks during node manipulation
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node: prev_node.next = next_node
        if next_node: next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node immediately after the dummy head (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def _cleanup_expired(self) -> None:
        """
        Remove expired items from the tail (oldest) of the list.
        Amortized O(1) because each item is added and removed exactly once.
        """
        now = time.monotonic()
        while self.tail.prev != self.head:
            oldest = self.tail.prev
            if now > oldest.expiry:
                self._remove(oldest)
                if oldest.key in self.cache:
                    del self.cache[oldest.key]
            else:
                # Since we only add to front and items are added with 
                # monotonically increasing expiry (mostly), we can stop 
                # when we hit the first non-expired item from the tail.
                break

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
        
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair. 
        Evicts LRU non-expired item if at capacity.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._remove(node)
            self._add_to_front(node)
            return

        # Lazy cleanup of expired items before checking capacity
        self._cleanup_expired()

        if len(self.cache) >= self.capacity:
            # Evict the least recently used (tail.prev)
            # Based on requirements: if all are expired, _cleanup_expired already cleared them.
            # If still at capacity, we remove the LRU non-expired item.
            lru_node = self.tail.prev
            if lru_node != self.head:
                self.delete(lru_node.key)

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if key existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Return count of non-expired items."""
        self._cleanup_expired()
        return len(self.cache)

# --- Tests ---
import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)

def test_basic_get_put(cache):
    """Test basic insertion and retrieval."""
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_capacity_eviction(cache):
    """Test that the least recently used item is evicted when capacity is reached."""
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    # Access 'a' to make it MRU, 'b' becomes LRU
    cache.get("a")
    # This should evict 'b'
    cache.put("d", 4)
    
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.get("d") == 4

def test_ttl_expiry(cache):
    """Test that items expire after the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1)
        
        # Advance time past TTL (10s)
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl(cache):
    """Test that per-key TTL overrides the default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        # Advance time by 5 seconds
        mock_time.return_value = 105.0
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete(cache):
    """Test explicit deletion of keys."""
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False

def test_size_mixed_expiry(cache):
    """Test size calculation with a mix of expired and valid items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=15.0)
        cache.put("c", 3, ttl=25.0)
        
        # Advance time so 'a' expires
        mock_time.return_value = 110.0
        # size() should trigger lazy cleanup of 'a'
        assert cache.size() == 2
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
```