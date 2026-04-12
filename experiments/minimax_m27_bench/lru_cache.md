# MiniMax-M2.7 UD-IQ3_S (empty-think) — LRU Cache with TTL

## Output (11834 chars)

```python
"""
LRU Cache with TTL (Time‑To‑Live) support.

The implementation uses a doubly‑linked list together with a hash map
to achieve O(1) average time for all standard operations.
"""

import time
from typing import Any, Optional

# ----------------------------------------------------------------------
# Node for the doubly‑linked list
# ----------------------------------------------------------------------
class _Node:
    """A node storing a key, a value, and the expiration timestamp."""

    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expiry: float = expiry
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


# ----------------------------------------------------------------------
# TTLCache
# ----------------------------------------------------------------------
class TTLCache:
    """
    Least‑Recently‑Used cache with per‑item time‑to‑live expiration.

    Parameters
    ----------
    capacity : int
        Maximum number of non‑expired items the cache may hold.
    default_ttl : float
        Default TTL in seconds for items that do not specify a custom TTL.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be a positive number")

        self._capacity: int = capacity
        self._default_ttl: float = default_ttl

        # Sentinel nodes simplify edge‑case handling (no None checks)
        self._head = _Node("", None, -1.0)  # dummy head (most‑recent side)
        self._tail = _Node("", None, -1.0)  # dummy tail (least‑recent side)
        self._head.next = self._tail
        self._tail.prev = self._head

        # Hash map: key -> _Node
        self._cache: dict[str, _Node] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        """Return the current monotonic time (seconds)."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Return True if the node has passed its expiration time."""
        return self._now() >= node.expiry

    def _remove_node(self, node: _Node) -> None:
        """Detach a node from the doubly‑linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_front(self, node: _Node) -> None:
        """Insert a node right after the dummy head (makes it most‑recent)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """Remove the least‑recent (just before dummy tail) non‑expired node."""
        candidate = self._tail.prev
        # Skip any already‑expired nodes that may linger
        while candidate is not self._head and self._is_expired(candidate):
            self._remove_node(candidate)
            del self._cache[candidate.key]
            candidate = self._tail.prev

        if candidate is not self._head:
            self._remove_node(candidate)
            del self._cache[candidate.key]

    def _enforce_capacity(self) -> None:
        """Ensure the cache does not exceed its capacity."""
        while len(self._cache) >= self._capacity:
            self._evict_lru()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for ``key`` if it exists and has not expired.

        Accessing a key makes it the most‑recently‑used item.

        Returns
        -------
        Optional[Any]
            The cached value, or None if the key is missing or expired.
        """
        node = self._cache.get(key)
        if node is None:
            return None

        # Lazy cleanup: discard if the node has expired
        if self._is_expired(node):
            self._remove_node(node)
            del self._cache[key]
            return None

        # Move to front (most‑recent)
        self._remove_node(node)
        self._insert_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update ``key`` with ``value``.

        If the cache is at capacity, the least‑recently‑used non‑expired
        item is evicted. If all items are already expired, they are cleared
        first.

        Parameters
        ----------
        key : str
            Unique identifier for the cached entry.
        value : Any
            The value to store.
        ttl : Optional[float]
            Per‑key TTL in seconds. If omitted, ``default_ttl`` is used.
        """
        # Determine expiration moment
        ttl_value = ttl if ttl is not None else self._default_ttl
        expiry = self._now() + ttl_value

        # If key already exists, refresh its position and expiry
        if key in self._cache:
            node = self._cache[key]
            # Even if the current node is expired we replace it
            self._remove_node(node)
            node.value = value
            node.expiry = expiry
            self._insert_front(node)
            self._cache[key] = node
            return

        # Evict until we have space (also clears any expired entries)
        self._enforce_capacity()

        # Create and insert the new node
        new_node = _Node(key, value, expiry)
        self._insert_front(new_node)
        self._cache[key] = new_node

    def delete(self, key: str) -> bool:
        """
        Remove ``key`` from the cache.

        Returns
        -------
        bool
            True if the key was present, False otherwise.
        """
        node = self._cache.get(key)
        if node is None:
            return False
        self._remove_node(node)
        del self._cache[key]
        return True

    def size(self) -> int:
        """
        Return the number of non‑expired items currently stored.

        Expired entries are lazily removed during this call.
        """
        # Perform a lazy sweep to discard any expired nodes
        to_remove = [
            node
            for node in self._cache.values()
            if self._is_expired(node)
        ]
        for node in to_remove:
            self._remove_node(node)
            del self._cache[node.key]
        return len(self._cache)


# ----------------------------------------------------------------------
# Pytest tests
# ----------------------------------------------------------------------
import pytest
from unittest.mock import patch
from functools import wraps

# ----------------------------------------------------------------------
# Helper to make time.monotonic() return a controlled sequence
# ----------------------------------------------------------------------
class _TimeMachine:
    """Yield increasing timestamps on each call to monotonic()."""

    def __init__(self, start: float = 0.0, step: float = 1.0):
        self._current = start
        self._step = step

    def __call__(self) -> float:
        result = self._current
        self._current += self._step
        return result


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def time_machine():
    """Provides a deterministic time source starting at 100.0."""
    return _TimeMachine(start=100.0, step=1.0)


@pytest.fixture
def cache(time_machine):
    """Creates a TTLCache with capacity 3 and default TTL 10 seconds."""
    with patch("time.monotonic", side_effect=time_machine):
        yield TTLCache(capacity=3, default_ttl=10.0)


# ----------------------------------------------------------------------
# Test 1 – Basic get / put
# ----------------------------------------------------------------------
def test_basic_get_put(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        cache.put("a", 1)
        assert cache.get("a") == 1

        cache.put("b", 2)
        assert cache.get("b") == 2

        # Missing key
        assert cache.get("nonexistent") is None


# ----------------------------------------------------------------------
# Test 2 – Capacity eviction respects LRU order
# ----------------------------------------------------------------------
def test_capacity_eviction_lru(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        cache.put("a", 1)   # a most recent
        cache.put("b", 2)   # b most recent
        cache.put("c", 3)   # c most recent

        # Access 'a' so it becomes MRU
        cache.get("a")

        # Adding a new item should evict 'b' (LRU)
        cache.put("d", 4)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") is None
        assert cache.get("d") == 4


# ----------------------------------------------------------------------
# Test 3 – TTL expiry removes items automatically
# ----------------------------------------------------------------------
def test_ttl_expiry(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        cache.put("x", 10)

        # Advance time past the default TTL (10 seconds)
        time_machine._current += 11.0  # time = 111.0

        # Access should miss because the entry expired
        assert cache.get("x") is None


# ----------------------------------------------------------------------
# Test 4 – Custom per‑key TTL
# ----------------------------------------------------------------------
def test_custom_ttl(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        # Store y with a short TTL of 3 seconds
        cache.put("y", 20, ttl=3.0)

        # After 2 seconds, still alive
        time_machine._current += 2.0
        assert cache.get("y") == 20

        # After 2 more seconds (total 4), expired
        time_machine._current += 2.0
        assert cache.get("y") is None


# ----------------------------------------------------------------------
# Test 5 – Delete existing and non‑existing keys
# ----------------------------------------------------------------------
def test_delete(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        cache.put("k1", 100)
        cache.put("k2", 200)

        assert cache.delete("k1") is True
        assert cache.get("k1") is None
        assert cache.get("k2") == 200

        # Deleting a non‑existent key returns False
        assert cache.delete("nonexistent") is False


# ----------------------------------------------------------------------
# Test 6 – Size with mixed expired / valid items
# ----------------------------------------------------------------------
def test_size_mixed_expired(cache, time_machine):
    with patch("time.monotonic", side_effect=time_machine):
        cache.put("a", 1)   # TTL 10
        cache.put("b", 2)   # TTL 10
        cache.put("c", 3)   # TTL 10

        # Advance time so only 'a' and 'b' expire
        time_machine._current += 11.0

        # 'c' is still alive, 'a' and 'b' are expired
        assert cache.size() == 1

        # Adding a new item should not resurrect expired entries
        cache.put("d", 4)
        assert cache.size() == 2
        assert cache.get("c") is None   # c was evicted during put
        assert cache.get("d") == 4
```
