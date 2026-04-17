Here's a complete implementation of `TTLCache` meeting all requirements, followed by 6 pytest tests using `unittest.mock.patch` for deterministic time control.

---

### ✅ Implementation: `ttl_cache.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict
import time


@dataclass
class Node:
    """Doubly-linked list node storing key, value, expiry time, and prev/next pointers."""
    key: str
    value: Any
    expiry: float  # monotonic time when item expires
    prev: Optional['Node'] = None
    next: Optional['Node'] = None


class TTLCache:
    """
    A thread-unsafe LRU cache with TTL expiration.
    Uses a doubly-linked list (for O(1) LRU ordering) and a hash map (for O(1) lookup).
    All operations are O(1) average time.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of non-expired items.
        :param default_ttl: Default time-to-live in seconds for items without custom TTL.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, Node] = {}  # key -> Node
        self._head: Optional[Node] = None  # MRU end (most recently used)
        self._tail: Optional[Node] = None  # LRU end (least recently used)
        self._current_time = time.monotonic  # for testability

    def _remove(self, node: Node) -> None:
        """Remove node from doubly-linked list (in O(1))."""
        prev = node.prev
        next_ = node.next

        if prev:
            prev.next = next_
        else:
            # node was head
            self._head = next_

        if next_:
            next_.prev = prev
        else:
            # node was tail
            self._tail = prev

        node.prev = node.next = None

    def _move_to_head(self, node: Node) -> None:
        """Move node to head (most recently used) in O(1)."""
        self._remove(node)
        self._add_to_head(node)

    def _add_to_head(self, node: Node) -> None:
        """Add node to head (most recently used) in O(1)."""
        node.prev = None
        node.next = self._head

        if self._head:
            self._head.prev = node
        self._head = node

        if not self._tail:
            self._tail = node

    def _evict_lru(self) -> None:
        """Evict the least recently used (non-expired) item."""
        # Tail is LRU; remove it
        if self._tail:
            self._remove(self._tail)
            del self._map[self._tail.key]

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache (lazy cleanup on access)."""
        now = self._current_time()
        # Traverse from tail (oldest) until we hit a non-expired item
        while self._tail and self._tail.expiry <= now:
            self._remove(self._tail)
            del self._map[self._tail.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.

        :param key: Cache key.
        :return: Value if found and not expired, else None.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        now = self._current_time()

        if node.expiry <= now:
            # Expired: remove and return None
            self._remove(node)
            del self._map[key]
            return None

        # Move to head (most recently used)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        :param key: Cache key.
        :param value: Value to store.
        :param ttl: Optional custom TTL (seconds); uses default_ttl if None.
        """
        now = self._current_time()

        # If key exists, update value and expiry, then move to head
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = now + (ttl if ttl is not None else self.default_ttl)
            self._move_to_head(node)
            return

        # Clean up expired items first (if needed)
        self._cleanup_expired()

        # If at capacity, evict LRU non-expired item
        if len(self._map) >= self.capacity:
            self._evict_lru()

        # Insert new node
        ttl_val = ttl if ttl is not None else self.default_ttl
        node = Node(key=key, value=value, expiry=now + ttl_val)
        self._add_to_head(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        :param key: Cache key.
        :return: True if key existed, False otherwise.
        """
        if key not in self._map:
            return False

        node = self._map[key]
        self._remove(node)
        del self._map[key]
        return True

    def size(self) -> int:
        """
        Return count of non-expired items.

        Performs lazy cleanup: expired items are removed on access.
        """
        self._cleanup_expired()
        return len(self._map)
```

---

### ✅ Tests: `test_ttl_cache.py`

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def test_basic_get_put():
    with patch('ttl_cache.time.monotonic', side_effect=[0.0, 0.0, 1.0]):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("b") is None


def test_capacity_eviction_lru_order():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put("a", 1)  # time=0
        cache.put("b", 2)  # time=0
        cache.put("c", 3)  # time=0
        cache.get("a")     # time=1 → moves "a" to MRU
        cache.put("d", 4)  # time=1 → evicts "b" (LRU)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4


def test_ttl_expiry():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 5, 10]):
        cache = TTLCache(capacity=3, default_ttl=5.0)
        cache.put("x", 100)
        assert cache.get("x") == 100  # time=0
        assert cache.get("x") == 100  # time=5 (still valid)
        assert cache.get("x") is None  # time=10 (expired)


def test_custom_per_key_ttl():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 2, 3]):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("short", 1, ttl=1.0)   # expires at t=1
        cache.put("long", 2, ttl=5.0)    # expires at t=5
        assert cache.get("short") == 1   # t=0
        assert cache.get("short") is None  # t=2 (expired)
        assert cache.get("long") == 2    # t=3 (still valid)


def test_delete():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0]):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False  # already gone


def test_size_mixed_expired_valid():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 0, 5, 5, 5]):
        cache = TTLCache(capacity=5, default_ttl=3.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.size() == 3  # all valid
        assert cache.get("a") == 1  # time=5 → "a" expired, removed
        assert cache.size() == 2  # "b", "c" still valid
        cache.put("d", 4)  # time=5 → "b", "c" still valid, "d" added
        assert cache.size() == 3
```

---

### ✅ How to Run

1. Save the implementation as `ttl_cache.py`.
2. Save the tests as `test_ttl_cache.py`.
3. Install pytest: `pip install pytest`
4. Run tests: `pytest test_ttl_cache.py -v`

---

### ✅ Notes

- **O(1) operations**: All operations use hash map + doubly-linked list (no traversal).
- **Lazy cleanup**: Expired items are removed only when accessed (on `get`, `put`, `size`).
- **Time tracking**: Uses `time.monotonic()` via a mockable reference (`self._current_time`).
- **LRU eviction**: Only evicts non-expired items; expired items are cleaned up first.
- **No OrderedDict**: Pure manual doubly-linked list implementation.

Let me know if you'd like async support or thread-safety added!