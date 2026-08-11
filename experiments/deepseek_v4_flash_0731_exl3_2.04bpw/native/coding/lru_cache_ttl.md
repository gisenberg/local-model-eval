```python
import time
from typing import Any, Optional


class _Node:
    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    LRU cache with time-based expiration.

    Internally uses a doubly linked list for LRU ordering and a hash map
    for O(1) key lookup. Expired items are removed lazily on access.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items to hold.
            default_ttl: Default time-to-live in seconds for new items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, _Node] = {}
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def _remove_node(self, node: _Node) -> None:
        """Unlink a node from the list and remove it from the hash map."""
        node.prev.next = node.next
        node.next.prev = node.prev
        if node.key in self._cache:
            del self._cache[node.key]
        self._size -= 1

    def _add_to_head(self, node: _Node) -> None:
        """Insert a node right after the dummy head (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the MRU position."""
        if node.prev is self._head:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _purge_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.monotonic()
        node = self._head.next
        while node != self._tail:
            next_node = node.next
            if node.expiry <= now:
                self._remove_node(node)
            node = next_node

    def _evict_if_needed(self, now: float) -> None:
        """
        Evict the least-recently-used non-expired item if at capacity.

        Expired items at the tail are removed first. If all items are expired,
        they are all cleared before inserting a new item.
        """
        while self._size >= self.capacity:
            node = self._tail.prev
            if node == self._head:
                return
            if node.expiry <= now:
                self._remove_node(node)
            else:
                self._remove_node(node)
                break

    def get(self, key: str) -> Optional[Any]:
        """
        Return the value for `key` if it exists and is not expired.

        Accessing a key makes it the most-recently-used item.
        Expired items are removed lazily.
        """
        node = self._cache.get(key)
        if node is None:
            return None
        if node.expiry <= time.monotonic():
            self._remove_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update `key` with `value`.

        If `ttl` is provided, it overrides the default TTL for this item.
        If the cache is at capacity, the least-recently-used non-expired item
        is evicted.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        node = self._cache.get(key)
        if node is not None:
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
            return

        self._evict_if_needed(now)
        if self._size >= self.capacity:
            return

        node = _Node(key, value, expiry)
        self._cache[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """
        Remove `key` from the cache.

        Returns True if the key existed, False otherwise.
        """
        node = self._cache.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.

        Expired items are purged before counting.
        """
        self._purge_expired()
        return self._size


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------
import pytest
from unittest.mock import patch


def test_basic_get_put():
    with patch("time.monotonic", return_value=0.0):
        cache = TTLCache(2, 100.0)
        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None


def test_capacity_eviction_lru():
    with patch("time.monotonic", return_value=0.0):
        cache = TTLCache(2, 100.0)
        cache.put("a", 1)
        cache.put("b", 2)

        # Access "a" so it becomes MRU; "b" is now LRU.
        assert cache.get("a") == 1

        # Inserting "c" should evict "b".
        cache.put("c", 3)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3


def test_ttl_expiry():
    with patch("time.monotonic", side_effect=[0.0, 0.0, 11.0, 11.0]):
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)

        # At time 11.0, both items have expired.
        assert cache.get("a") is None
        assert cache.get("b") is None


def test_custom_ttl():
    with patch("time.monotonic", side_effect=[0.0, 0.0, 6.0, 6.0]):
        cache = TTLCache(2, 100.0)
        cache.put("a", 1, ttl=5)   # expires at t=5
        cache.put("b", 2)          # expires at t=100

        # At t=6, "a" is expired, "b" is still valid.
        assert cache.get("a") is None
        assert cache.get("b") == 2


def test_delete():
    with patch("time.monotonic", return_value=0.0):
        cache = TTLCache(2, 100.0)
        cache.put("a", 1)

        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False


def test_size_mixed():
    with patch("time.monotonic", side_effect=[0.0, 0.0, 0.0, 6.0, 6.0]):
        cache = TTLCache(3, 100.0)
        cache.put("a", 1, ttl=5)   # expires at t=5
        cache.put("b", 2)          # expires at t=100
        cache.put("c", 3, ttl=5)   # expires at t=5

        # At t=6, only "b" is non-expired.
        assert cache.size() == 1

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") is None
```