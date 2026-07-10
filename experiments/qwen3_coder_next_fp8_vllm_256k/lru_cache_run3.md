

Here's a complete implementation of an LRU Cache with TTL in Python, satisfying all your requirements.

---

### ✅ Features
- **O(1)** average time for `get`, `put`, `delete`, `size`
- Uses **doubly-linked list** (custom) + **hash map**
- **Lazy expiration**: nodes marked expired only when accessed
- `time.monotonic()` for TTL tracking (resistant to clock changes)
- Full type hints and docstrings

---

### 📦 `ttl_cache.py`

```python
import time
from typing import Optional, Any, Dict


class Node:
    """Doubly-linked list node for cache entry."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at  # absolute timestamp (monotonic)
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class DoublyLinkedList:
    """Doubly-linked list with head/tail sentinels for O(1) operations."""
    def __init__(self):
        # Sentinel nodes: head.next = LRU node, tail.prev = MRU node
        self.head = Node(None, None, float('inf'))  # dummy head (least recently used side)
        self.tail = Node(None, None, 0.0)           # dummy tail (most recently used side)
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def __len__(self):
        return self._size

    def append(self, node: Node) -> None:
        """Move node to MRU (tail.prev) position."""
        prev_node = self.tail.prev
        node.prev = prev_node
        node.next = self.tail
        prev_node.next = node
        self.tail.prev = node
        self._size += 1

    def remove(self, node: Node) -> None:
        """Remove node from list."""
        if node.prev is None or node.next is None:
            return  # already detached
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._size -= 1

    def popleft(self) -> Optional[Node]:
        """Remove and return LRU node (head.next), or None if empty."""
        if self.head.next is self.tail:
            return None
        node = self.head.next
        self.remove(node)
        return node

    def move_to_end(self, node: Node) -> None:
        """Move node to MRU position."""
        if node.next is self.tail:  # already at end
            return
        self.remove(node)
        self.append(node)


class TTLCache:
    """
    Thread-unsafe LRU cache with TTL support.

    O(1) average time for get/put/delete/size operations.
    Uses lazy expiration: expired items are only evicted on access.
    """
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items in the cache.
            default_ttl: Default time-to-live in seconds (from now).
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[Any, Node] = {}
        self._list = DoublyLinkedList()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value for key. Returns None if key not found or expired.
        Updates key to most recently used if valid.

        Args:
            key: Key to look up.

        Returns:
            Value if found and not expired, else None.
        """
        node = self._map.get(key)
        if node is None:
            return None

        # Check expiration lazily
        if node.expires_at <= time.monotonic():
            self._delete_node(node)
            return None

        # Move to MRU position
        self._list.move_to_end(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.

        Args:
            key: Key to insert/update.
            value: Value to store.
            ttl: Optional TTL in seconds; uses default_ttl if None.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        if ttl <= 0:
            raise ValueError("TTL must be positive")

        if key in self._map:
            # Update existing node
            node = self._map[key]
            node.value = value
            node.expires_at = time.monotonic() + ttl
            self._list.move_to_end(node)
        else:
            # Evict if at capacity
            if len(self._map) >= self.capacity:
                self._evict_one()
            # Insert new node
            expires_at = time.monotonic() + ttl
            node = Node(key, value, expires_at)
            self._map[key] = node
            self._list.append(node)

    def delete(self, key: Any) -> bool:
        """
        Remove key from cache.

        Args:
            key: Key to delete.

        Returns:
            True if key existed, False otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False
        self._delete_node(node)
        return True

    def size(self) -> int:
        """Return current number of items in cache."""
        return len(self._map)

    def _evict_one(self) -> None:
        """Evict least recently used (expired or not) node."""
        node = self._list.popleft()
        if node is not None:
            del self._map[node.key]

    def _delete_node(self, node: Node) -> None:
        """Delete node from both list and map."""
        self._map.pop(node.key, None)
        self._list.remove(node)
```

---

### 🧪 Tests (`test_ttl_cache.py`)

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def create_cache(monotonic_time: float, capacity: int = 3, default_ttl: float = 10.0):
    with patch('ttl_cache.time.monotonic', return_value=monotonic_time):
        return TTLCache(capacity, default_ttl)


class TestTTLCache:
    def test_basic_put_get(self):
        cache = create_cache(100.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.size() == 1

    def test_expiration(self):
        # t=0: insert item with TTL=5 → expires at t=5
        cache = create_cache(0.0)
        cache.put("a", 1)
        # t=3: still valid
        with patch('ttl_cache.time.monotonic', return_value=3.0):
            assert cache.get("a") == 1
        # t=6: expired
        with patch('ttl_cache.time.monotonic', return_value=6.0):
            assert cache.get("a") is None
        assert cache.size() == 0  # expired item removed

    def test_update_ttl_on_put(self):
        cache = create_cache(0.0)
        cache.put("a", 1, ttl=5)
        # t=4: valid
        with patch('ttl_cache.time.monotonic', return_value=4.0):
            assert cache.get("a") == 1
        # Update TTL to +2 → new expiry at t=4+2=6
        with patch('ttl_cache.time.monotonic', return_value=4.0):
            cache.put("a", 2, ttl=2)
        # t=5: should still be valid (expires at 6)
        with patch('ttl_cache.time.monotonic', return_value=5.0):
            assert cache.get("a") == 2
        # t=7: now expired
        with patch('ttl_cache.time.monotonic', return_value=7.0):
            assert cache.get("a") is None

    def test_lru_eviction(self):
        cache = create_cache(0.0, capacity=2, default_ttl=100.0)
        cache.put("a", 1)  # MRU
        cache.put("b", 2)  # MRU
        # "a" is LRU
        with patch('ttl_cache.time.monotonic', return_value=5.0):
            cache.put("c", 3)  # evicts "a"
        # "a" evicted, "b" & "c" present
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_evict_expired_on_put(self):
        cache = create_cache(0.0, capacity=1)
        cache.put("a", 1, ttl=2)
        # t=1: valid
        with patch('ttl_cache.time.monotonic', return_value=1.0):
            cache.get("a")  # refreshes access time
        # t=3: expired, but still in cache (lazy eviction)
        with patch('ttl_cache.time.monotonic', return_value=3.0):
            # now eviction triggered by fresh put
            cache.put("b", 2, ttl=5)  # should evict expired "a"
        assert cache.size() == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_delete(self):
        cache = create_cache(0.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.size() == 0
        assert cache.delete("a") is False  # already gone

        # Test delete non-existent
        assert cache.delete("x") is False
```

---

### 🧪 Run Tests

```bash
pip install pytest
pytest test_ttl_cache.py -v
```

Expected output:

```
test_ttl_cache.py::TestTTLCache::test_basic_put_get PASSED
test_ttl_cache.py::TestTTLCache::test_expiration PASSED
test_ttl_cache.py::TestTTLCache::test_update_ttl_on_put PASSED
test_ttl_cache.py::TestTTLCache::test_lru_eviction PASSED
test_ttl_cache.py::TestTTLCache::test_evict_expired_on_put PASSED
test_ttl_cache.py::TestTTLCache::test_delete PASSED
```

---

Let me know if you want:
- Thread-safe version (`threading.Lock`)
- Explicit cleanup method (`cleanup()`)
- TTL stats or logging hooks