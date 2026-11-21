Here's a complete implementation. It uses a hash map for O(1) key lookup plus an intrusive doubly-linked list (head = most-recently-used, tail = least-recently-used) with sentinel nodes, with no `OrderedDict`.

**`ttl_cache.py`**

```python
"""LRU cache with per-item time-to-live (TTL) expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _ListNode:
    """Intrusive doubly-linked-list node tracking recency order.

    Ordered from most-recently-used (head) to least-recently-used (tail).
    """

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_ListNode] = None
        self.next: Optional[_ListNode] = None


class TTLCache:
    """A least-recently-used (LRU) cache with per-item expiration.

    Internally: a hash map from key -> node, plus a doubly-linked list of
    nodes with sentinel head/tail nodes. get/put/delete are O(1) average;
    size() performs lazy expiry cleanup, so it is O(n). Time comes from
    time.monotonic().
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items to retain. Must be >= 1.
            default_ttl: Default seconds until an item expires. Must be >= 0.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl < 0:
            raise ValueError("default_ttl must be >= 0")
        self._capacity = capacity
        self._default_ttl = float(default_ttl)
        self._map: Dict[str, _ListNode] = {}
        # Sentinels: head is the MRU end, tail is the LRU end.
        self._head = _ListNode("", None, 0.0)
        self._tail = _ListNode("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: str) -> Optional[Any]:
        """Return the value for key if present and not expired, else None.

        A hit makes the key most-recently-used. Expired items are removed
        lazily and treated as missing.
        """
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return None
        if now >= node.expires_at:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update key and make it most-recently-used.

        The item expires ``ttl`` seconds from now (default_ttl if ttl is
        None). If at capacity, expired items are cleared first; then the
        least-recently-used non-expired item is evicted.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else float(ttl)
        expires_at = now + effective_ttl

        node = self._map.get(key)
        if node is not None:  # update existing key
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return

        if len(self._map) >= self._capacity:
            # Clear expired items first (covers the all-expired case).
            self._purge_expired(now)
            if len(self._map) >= self._capacity:
                # Evict the least-recently-used (now definitely non-expired) item.
                lru_node = self._tail.prev
                if lru_node is not self._head:
                    self._remove_node(lru_node)

        new_node = _ListNode(key, value, expires_at)
        self._map[key] = new_node
        self._link_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove key; return True if it existed and was not expired."""
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return now < node.expires_at

    def size(self) -> int:
        """Return the number of non-expired items (lazy cleanup)."""
        now = time.monotonic()
        self._purge_expired(now)
        return len(self._map)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _unlink(self, node: _ListNode) -> None:
        """Unlink a node from the list (map untouched)."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _link_front(self, node: _ListNode) -> None:
        """Insert node as the most-recently-used item (after head sentinel)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _ListNode) -> None:
        """Make node the most-recently-used item."""
        if node is self._head.next:
            return
        self._unlink(node)
        self._link_front(node)

    def _remove_node(self, node: _ListNode) -> None:
        """Remove node from the list and the hash map."""
        self._unlink(node)
        del self._map[node.key]

    def _purge_expired(self, now: float) -> None:
        """Remove every item whose expires_at has passed."""
        node = self._tail.prev
        while node is not self._head:
            prev_node = node.prev
            if now >= node.expires_at:
                self._remove_node(node)
            node = prev_node
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. Time is mocked for determinism (no sleeps)."""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

from ttl_cache import TTLCache


class _FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def fake_time(start: float = 1000.0) -> Generator[_FakeClock, None, None]:
    """Yield a _FakeClock while time.monotonic is mocked to return it."""
    clock = _FakeClock(start)
    with patch("time.monotonic", side_effect=clock):
        yield clock


def test_basic_get_put():
    with fake_time():
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        cache.put("a", 2)  # update existing key
        assert cache.get("a") == 2
        assert cache.size() == 1


def test_capacity_eviction_lru_order():
    with fake_time():
        cache = TTLCache(capacity=2, default_ttl=60.0)
        cache.put("k1", 1)
        cache.put("k2", 2)
        assert cache.get("k1") == 1  # k1 -> MRU; k2 is now the LRU item
        cache.put("k3", 3)           # evicts k2
        assert cache.get("k2") is None
        assert cache.get("k1") == 1
        assert cache.get("k3") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with fake_time() as clock:
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("x", "value")
        assert cache.get("x") == "value"
        clock.advance(9.9)
        assert cache.get("x") == "value"
        clock.advance(0.1)  # exactly at the expiry instant -> expired
        assert cache.get("x") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with fake_time() as clock:
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("short", "s", ttl=2.0)
        cache.put("long", "l")
        clock.advance(5.0)
        assert cache.get("short") is None
        assert cache.get("long") == "l"
        clock.advance(6.0)  # total 11s -> default item expires too
        assert cache.get("long") is None


def test_delete():
    with fake_time() as clock:
        cache = TTLCache(capacity=5, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.size() == 1
        assert cache.delete("a") is False  # already removed
        cache.put("c", 3)
        clock.advance(11.0)
        assert cache.delete("c") is False  # expired -> treated as absent


def test_size_with_mixed_expired_and_valid_items():
    with fake_time() as clock:
        cache = TTLCache(capacity=10, default_ttl=5.0)
        cache.put("a", 1)   # expires at t=1005
        cache.put("b", 2)   # expires at t=1005
        clock.advance(3.0)  # t=1003
        cache.put("c", 3)   # expires at t=1008
        clock.advance(3.0)  # t=1006: a and b expired, c still valid
        assert cache.size() == 1
        assert cache.get("c") == 3
        assert cache.size() == 1
```

**Design notes**

- **Recency order**: nodes are chained with sentinels — head is MRU, tail is LRU. `get` and `put` move the touched node to the front; eviction takes the node at the tail (`tail.prev`), both O(1).
- **Expiry**: each node stores `expires_at = now + ttl` (custom `ttl` wins over the default). Expiry is evaluated lazily — on `get`/`delete`/`size` and during `put`'s capacity check — and is exact at the expiry boundary (`now >= expires_at`).
- **Eviction**: `put` only purges when the map is full. First it clears expired items (which covers the "all items expired" case); only if still full does it evict the tail's non-expired LRU item. Updates to an existing key refresh its value, TTL, and recency without triggering eviction.
- **Honest complexity note**: `get`/`put`/`delete` are O(1) average (as required). `size()` performs lazy cleanup across all items, so it's O(n); `put` is O(n) only in the worst case where it must purge expired items at capacity. This is inherent to exact TTL counting without an extra time-sorted structure.
- `delete` treats an expired item as absent (returns `False`), consistent with `get`/`size`.
- The clock is fully mocked via `unittest.mock.patch("time.monotonic")` with a callable `side_effect`, so tests are deterministic and sleep-free.

Run with:

```bash
pytest test_ttl_cache.py -v
```