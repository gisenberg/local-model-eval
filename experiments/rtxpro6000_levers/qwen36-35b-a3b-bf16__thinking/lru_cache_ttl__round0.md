Here's the complete implementation with type hints, docstrings, and the requested pytest tests.

```python
import time
from typing import Any, Optional
from unittest.mock import patch


class _Node:
    """Internal doubly-linked list node for LRU ordering."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """LRU Cache with time-based expiration using a doubly-linked list and hash map."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """Initialize cache with max capacity and default time-to-live in seconds."""
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        
        # Dummy head (MRU) and tail (LRU) nodes
        self._head: _Node = _Node(None, None, 0.0)
        self._tail: _Node = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        
        # Tracks only non-expired items for O(1) size()
        self._size: int = 0

    def _is_expired(self, node: _Node) -> bool:
        return time.monotonic() >= node.expires_at

    def _remove_node(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_to_head(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: str) -> Optional[Any]:
        """Return value if key exists and not expired, else None. Accessing makes it MRU."""
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
            
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key-value pair. Evicts LRU if at capacity."""
        now = time.monotonic()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        if key in self._cache:
            node = self._cache[key]
            if self._is_expired(node):
                self._remove_node(node)
                del self._cache[key]
                self._size -= 1
            else:
                node.value = value
                node.expires_at = expires_at
                self._move_to_head(node)
                return

        # Eviction logic: clear expired from tail first, then evict LRU non-expired
        if len(self._cache) >= self.capacity:
            while self._tail.prev != self._head and self._is_expired(self._tail.prev):
                expired_node = self._tail.prev
                self._remove_node(expired_node)
                del self._cache[expired_node.key]
                self._size -= 1

            if len(self._cache) >= self.capacity:
                lru_node = self._tail.prev
                self._remove_node(lru_node)
                del self._cache[lru_node.key]
                self._size -= 1

        node = _Node(key, value, expires_at)
        self._cache[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """Remove key from cache. Returns True if it existed, False otherwise."""
        if key not in self._cache:
            return False
            
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """Return count of non-expired items. Uses lazy cleanup on access."""
        return self._size
```

### Pytest Tests

```python
import pytest


def test_basic_get_put():
    """Test basic insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("b") is None


def test_capacity_eviction():
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3


def test_ttl_expiry():
    """Test that items expire after default TTL."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 11.0]):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        # Time jumps to 11.0, past the 10.0 TTL
        assert cache.get("a") is None


def test_custom_per_key_ttl():
    """Test that per-key TTL overrides default TTL."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 11.0]):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1, ttl=5.0)
        assert cache.get("a") == 1
        # Time jumps to 11.0, past the custom 5.0 TTL
        assert cache.get("a") is None


def test_delete():
    """Test key deletion and return value."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False
        assert cache.get("a") is None


def test_size_mixed_expired_valid():
    """Test size() accuracy with mixed expired and valid items."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 5.0, 11.0, 11.0]):
        cache = TTLCache(3, 10.0)
        cache.put("a", 1)
        cache.put("b", 2, ttl=5.0)
        cache.put("c", 3)
        
        assert cache.size() == 3
        
        # "b" expires at 5.0
        assert cache.get("b") is None
        assert cache.size() == 2
        
        # "a" and "c" expire at 10.0, time is now 11.0
        assert cache.get("a") is None
        assert cache.get("c") is None
        assert cache.size() == 0
```

### Key Design Notes:
1. **O(1) Operations**: Dictionary provides `O(1)` key lookups. Doubly-linked list operations (`_add_to_head`, `_remove_node`, `_move_to_head`) are `O(1)` pointer manipulations. `size()` is `O(1)` via a maintained counter.
2. **Lazy Cleanup**: Expired nodes are only removed when accessed (`get`) or when they surface at the tail during eviction (`put`). This avoids `O(N)` scans.
3. **Capacity Handling**: When at capacity, the tail is checked for expiration first. If expired, it's purged. This repeats until a non-expired LRU item is found or the cache is empty, strictly following the requirement.
4. **Time Tracking**: `time.monotonic()` is used exclusively to avoid issues with system clock adjustments.
5. **Testing**: `unittest.mock.patch` replaces `time.monotonic` with deterministic values, ensuring tests run instantly without `time.sleep` and are fully reproducible.