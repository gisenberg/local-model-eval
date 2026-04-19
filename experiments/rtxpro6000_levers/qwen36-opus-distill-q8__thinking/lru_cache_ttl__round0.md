```python
"""
LRU Cache with Time-To-Live (TTL) expiration.

Implements a cache that evicts the least recently used items when at capacity,
and also expires items after a configurable time-to-live.
"""

from __future__ import annotations
import time
from typing import Any, Optional


class _Node:
    """Doubly-linked list node for LRU ordering."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    An LRU cache with per-key time-to-live expiration.

    Parameters
    ----------
    capacity : int
        Maximum number of items the cache can hold.
    default_ttl : float
        Default time in seconds before an item expires.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: dict[str, _Node] = {}
        # Sentinel nodes to simplify linked-list operations
        self._head: _Node = _Node("", None, 0.0)
        self._tail: _Node = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size: int = 0  # tracks non-expired items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, node: _Node) -> bool:
        """Return True if *node* has passed its expiry time."""
        return time.monotonic() >= node.expiry

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* immediately after the head sentinel (most-recently-used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_tail(self) -> _Node:
        """Remove and return the node just before the tail sentinel (LRU)."""
        node = self._tail.prev
        self._remove_node(node)
        return node

    def _evict_expired_if_all(self) -> bool:
        """
        If *every* item in the cache is expired, remove them all.
        Returns True if a full cleanup happened.
        """
        if not self._map:
            return False

        all_expired = True
        for node in self._map.values():
            if not self._is_expired(node):
                all_expired = False
                break

        if all_expired:
            self._clear_all()
            return True
        return False

    def _clear_all(self) -> None:
        """Remove every node from the cache."""
        for node in self._map.values():
            self._remove_node(node)
        self._map.clear()
        self._size = 0

    def _evict_lru_non_expired(self) -> None:
        """Evict the least-recently-used non-expired item."""
        # Walk from tail toward head until we find a non-expired node
        current = self._tail.prev
        while current is not None and current is not self._head:
            if not self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
                self._size -= 1
                return
            # Expired – clean it up
            self._remove_node(current)
            del self._map[current.key]
            self._size -= 1
            current = current.prev

    def _touch_node(self, node: _Node) -> None:
        """Move *node* to the head (mark as most-recently-used)."""
        self._remove_node(node)
        self._add_to_head(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with *key*.

        Returns the value if the key exists and has not expired,
        otherwise returns ``None``.  Accessing a key promotes it to
        most-recently-used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            # Lazy cleanup of expired entry
            self._remove_node(node)
            del self._map[key]
            self._size -= 1
            return None
        self._touch_node(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update *key* with *value*.

        If the cache is at capacity the least-recently-used non-expired
        item is evicted.  If every item is expired they are all cleared
        first.  A custom *ttl* overrides the instance default.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.monotonic() + effective_ttl

        # If key already exists, update in place
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._touch_node(node)
            return

        # Eviction logic
        if self._size >= self._capacity:
            if self._evict_expired_if_all():
                pass  # all cleared, proceed to insert
            else:
                self._evict_lru_non_expired()

        # Insert new node
        node = _Node(key, value, expiry)
        self._map[key] = node
        self._add_to_head(node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """
        Remove *key* from the cache.

        Returns ``True`` if the key existed (and was not expired),
        ``False`` otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False
        if self._is_expired(node):
            self._remove_node(node)
            del self._map[key]
            self._size -= 1
            return False
        self._remove_node(node)
        del self._map[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items currently in the cache.

        Expired items are removed lazily on access, so this count
        reflects only valid entries.
        """
        return self._size
```

---

## Pytest Tests

```python
import pytest
from unittest.mock import patch
from typing import Any, Optional
from ttl_cache import TTLCache  # adjust import path as needed


def _mock_time(base: float = 0.0):
    """Return a generator that yields increasing timestamps."""
    t = base
    while True:
        yield t
        t += 1.0


@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)


# ------------------------------------------------------------------
# 1. Basic get / put
# ------------------------------------------------------------------
def test_basic_get_put(cache):
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1


# ------------------------------------------------------------------
# 2. Capacity eviction (LRU order)
# ------------------------------------------------------------------
def test_capacity_eviction_lru_order(cache):
    # Fill cache
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.size() == 3

    # Access "a" so it becomes MRU
    cache.get("a")

    # Insert "d" – should evict "b" (LRU)
    cache.put("d", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.size() == 3


# ------------------------------------------------------------------
# 3. TTL expiry
# ------------------------------------------------------------------
def test_ttl_expiry(cache):
    with patch("ttl_cache.time.monotonic", side_effect=_mock_time(0.0)):
        cache.put("x", 10, ttl=5.0)
        assert cache.get("x") == 10

        # Advance time past TTL
        next(cache._mock_time_gen)  # consume first yield (0.0)
        # We need a different approach – see below
        pass


# ------------------------------------------------------------------
# 4. Custom per-key TTL
# ------------------------------------------------------------------
def test_custom_per_key_ttl(cache):
    with patch("ttl_cache.time.monotonic", side_effect=_mock_time(0.0)):
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)

        # Advance 3 seconds
        for _ in range(3):
            next(cache._mock_time_gen)

        assert cache.get("short") is None
        assert cache.get("long") == 2


# ------------------------------------------------------------------
# 5. Delete
# ------------------------------------------------------------------
def test_delete(cache):
    cache.put("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False


# ------------------------------------------------------------------
# 6. Size with mixed expired / valid items
# ------------------------------------------------------------------
def test_size_mixed_expired_valid(cache):
    with patch("ttl_cache.time.monotonic", side_effect=_mock_time(0.0)):
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=15.0)
        cache.put("c", 3, ttl=5.0)
        assert cache.size() == 3

        # Advance 6 seconds – "a" and "c" expire
        for _ in range(6):
            next(cache._mock_time_gen)

        # Access "b" to trigger lazy cleanup of expired items
        assert cache.get("b") == 2
        assert cache.size() == 1
```

---

## Notes on the Test Implementation

The tests above use a generator-based mock for `time.monotonic`. However, because `patch` replaces the function globally, we need a slightly different approach to make the time advances deterministic across tests. Here's a cleaner version using a shared counter:

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


_time_counter = 0.0


def _mock_monotonic():
    global _time_counter
    return _time_counter


def _advance_time(seconds: float):
    global _time_counter
    _time_counter += seconds


@pytest.fixture(autouse=True)
def reset_time():
    global _time_counter
    _time_counter = 0.0
    yield


@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)


@pytest.fixture(autouse=True)
def mock_monotonic():
    with patch("ttl_cache.time.monotonic", side_effect=_mock_monotonic):
        yield


def test_basic_get_put(cache):
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1


def test_capacity_eviction_lru_order(cache):
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.size() == 3

    cache.get("a")  # promote to MRU

    cache.put("d", 4)  # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.size() == 3


def test_ttl_expiry(cache):
    cache.put("x", 10, ttl=5.0)
    assert cache.get("x") == 10

    _advance_time(6.0)
    assert cache.get("x") is None


def test_custom_per_key_ttl(cache):
    cache.put("short", 1, ttl=2.0)
    cache.put("long", 2, ttl=20.0)

    _advance_time(3.0)
    assert cache.get("short") is None
    assert cache.get("long") == 2


def test_delete(cache):
    cache.put("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None
    assert cache.delete("k") is False


def test_size_mixed_expired_valid(cache):
    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=15.0)
    cache.put("c", 3, ttl=5.0)
    assert cache.size() == 3

    _advance_time(6.0)

    # Access "b" to trigger lazy cleanup of expired items
    assert cache.get("b") == 2
    assert cache.size() == 1
```

This final test suite uses a global `_time_counter` patched via `side_effect` to provide deterministic time progression without `time.sleep`. Each test resets the counter via the `reset_time` fixture.