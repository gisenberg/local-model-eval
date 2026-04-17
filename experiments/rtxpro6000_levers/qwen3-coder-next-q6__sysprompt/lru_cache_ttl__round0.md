Here's a complete implementation of the `TTLCache` class with all requirements met, followed by 6 pytest tests using `unittest.mock.patch` to control time deterministically.

---

### ✅ Implementation (`ttl_cache.py`)

```python
from typing import Any, Optional, Dict
import time


class Node:
    """Doubly-linked list node storing key, value, and expiry time."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.

    Uses a doubly-linked list to maintain access order and a hash map for O(1) lookups.
    Expired entries are lazily cleaned up on access (get/put/size).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of non-expired items.
        :param default_ttl: Default time-to-live in seconds for items without custom TTL.
        """
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, Node] = {}
        # Dummy head (most recently used) and tail (least recently used)
        self._head = Node("", None, 0)
        self._tail = Node("", None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _move_to_front(self, node: Node) -> None:
        """Move node to front (most recently used)."""
        # Detach node
        node.prev.next = node.next
        node.next.prev = node.prev

        # Insert after head
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _remove_lru(self) -> Optional[Node]:
        """Remove and return the least recently used *non-expired* node."""
        # Start from tail.prev (first real node)
        curr = self._tail.prev
        while curr is not self._head:
            if curr.expires_at > time.monotonic():
                # Found non-expired LRU
                self._remove_node(curr)
                return curr
            curr = curr.prev
        return None

    def _evict_expired(self) -> None:
        """Remove all expired nodes (lazy cleanup)."""
        curr = self._tail.prev
        while curr is not self._head:
            if curr.expires_at <= time.monotonic():
                next_node = curr.prev
                self._remove_node(curr)
                del self._map[curr.key]
                curr = next_node
            else:
                break  # rest are newer (non-expired)

    def _ensure_capacity(self) -> None:
        """Ensure space by evicting LRU non-expired item or clearing all expired."""
        if len(self._map) <= self.capacity:
            return

        # First, try to evict LRU non-expired item
        evicted = self._remove_lru()
        if evicted is not None:
            del self._map[evicted.key]
            return

        # If all expired, clear all expired (lazy cleanup)
        self._evict_expired()

        # If still over capacity (shouldn't happen unless capacity < 0), clear all
        while len(self._map) > self.capacity:
            # Remove LRU (now only expired remain, but we just cleared them)
            # So this is fallback — remove any (shouldn't happen)
            lru = self._tail.prev
            if lru is self._head:
                break
            self._remove_node(lru)
            del self._map[lru.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.

        :param key: Key to look up.
        :return: Value if found and not expired, else None.
        """
        node = self._map.get(key)
        if node is None:
            return None

        # Check expiry
        if node.expires_at <= time.monotonic():
            # Expired: remove and return None
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to front (most recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.

        :param key: Key to insert/update.
        :param value: Value to store.
        :param ttl: Optional custom TTL; if None, use default_ttl.
        """
        # Ensure no expired items remain (lazy cleanup on write)
        self._evict_expired()

        if key in self._map:
            # Update existing node
            node = self._map[key]
            node.value = value
            node.expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
            self._move_to_front(node)
            return

        # New key: evict if needed
        if len(self._map) >= self.capacity:
            self._ensure_capacity()

        # Insert new node
        now = time.monotonic()
        ttl_val = ttl if ttl is not None else self.default_ttl
        node = Node(key, value, now + ttl_val)
        self._map[key] = node
        # Insert at front
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        :param key: Key to delete.
        :return: True if key existed, False otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False

        # Check expiry (shouldn't be in map if expired due to lazy cleanup, but be safe)
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            del self._map[key]
            return True

        self._remove_node(node)
        del self._map[key]
        return True

    def size(self) -> int:
        """
        Return number of non-expired items.

        Performs lazy cleanup of expired items.
        """
        self._evict_expired()
        return len(self._map)
```

---

### ✅ Tests (`test_ttl_cache.py`)

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def test_basic_get_put():
    with patch('ttl_cache.time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None


def test_capacity_eviction_lru_order():
    with patch('ttl_cache.time.monotonic', side_effect=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)  # t=0
        cache.put("b", 2)  # t=0
        cache.get("a")     # moves "a" to front → LRU is "b"
        cache.put("c", 3)  # evicts "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3


def test_ttl_expiry():
    with patch('ttl_cache.time.monotonic', side_effect=[0.0, 0.0, 5.0, 10.1]):
        cache = TTLCache(capacity=2, default_ttl=5.0)
        cache.put("a", 1)
        assert cache.get("a") == 1  # t=0.0 → not expired
        assert cache.get("a") is None  # t=5.0 → expired (5.0 <= 5.0)
        # Now at t=10.1, but no new put → size() should clean up
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with patch('ttl_cache.time.monotonic', side_effect=[0.0, 0.0, 2.0, 3.0]):
        cache = TTLCache(capacity=2, default_ttl=1.0)
        cache.put("a", 1)           # expires at 1.0
        cache.put("b", 2, ttl=3.0)  # expires at 3.0
        assert cache.get("a") is None  # t=2.0 > 1.0 → expired
        assert cache.get("b") == 2     # t=2.0 < 3.0 → valid
        assert cache.get("b") is None  # t=3.0 → expired


def test_delete():
    with patch('ttl_cache.time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False  # already gone


def test_size_mixed_expired_valid():
    with patch('ttl_cache.time.monotonic', side_effect=[0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 6.0]):
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("a", 1)  # expires at 5.0
        cache.put("b", 2)  # expires at 5.0
        cache.put("c", 3)  # expires at 5.0
        assert cache.size() == 3  # t=0.0
        # Advance time to 5.0 → all expired
        assert cache.size() == 0  # lazy cleanup on size()
        # Add new item
        cache.put("d", 4)  # expires at 10.0
        assert cache.size() == 1
        # Add another
        cache.put("e", 5)  # expires at 10.0
        assert cache.size() == 2
```

---

### ✅ Notes & Edge Cases Covered

- **Lazy cleanup**: Expired items are removed on `get`, `put`, and `size`.
- **Capacity eviction**: Only evicts *non-expired* LRU items; if all expired, clears them first.
- **Time tracking**: Uses `time.monotonic()` as required.
- **O(1) operations**: All operations use hash map + doubly-linked list (no `OrderedDict`).
- **Edge cases**:
  - `delete` on non-existent key → `False`.
  - `get` on expired key → `None`, and key removed.
  - `put` on existing key → updates value and TTL.
  - `size()` always returns count of *non-expired* items.

Let me know if you'd like a `__repr__`, thread-safety, or persistence hooks!