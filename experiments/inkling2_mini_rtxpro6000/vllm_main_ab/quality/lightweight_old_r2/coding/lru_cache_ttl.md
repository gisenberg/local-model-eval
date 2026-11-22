Here's the implementation and the test suite.

**`ttl_cache.py`**

```python
"""An LRU cache with per-entry time-based expiration (TTL)."""

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node backing the TTLCache hash map."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Least-recently-used cache with per-entry TTL expiration.

    Internal structure: a hash map (key -> node) plus a doubly-linked list
    ordered most-recently-used (head) -> least-recently-used (tail).
    Expired entries are removed lazily on access. O(1) average time per
    operation (cleanup passes are amortized O(1), since each node is removed
    at most once).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` items, each living `default_ttl` seconds."""
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        # Dummy sentinels so head/tail handling needs no special cases.
        self._head = _Node("", None, float("inf"))
        self._tail = _Node("", None, float("inf"))
        self._head.next = self._tail
        self._tail.prev = self._head

    # ---------- internal helpers ----------

    def _insert_front(self, node: _Node) -> None:
        """Link `node` as the most-recently-used item."""
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        first.prev = node
        self._head.next = node

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the list (without removing it from the map)."""
        prev = node.prev
        nxt = node.next
        assert prev is not None and nxt is not None
        prev.next = nxt
        nxt.prev = prev
        node.prev = None
        node.next = None

    def _remove_node(self, node: _Node) -> None:
        """Remove `node` from the hash map and the list."""
        self._map.pop(node.key, None)
        self._unlink(node)

    def _move_to_front(self, node: _Node) -> None:
        """Promote `node` to most-recently-used."""
        self._unlink(node)
        self._insert_front(node)

    def _purge_expired(self, now: float) -> None:
        """Remove all expired nodes from the list and map (lazy cleanup)."""
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            assert nxt is not None
            if now >= node.expires_at:
                self._remove_node(node)
            node = nxt

    def _enforce_capacity(self, now: float) -> None:
        """Drop expired items, then evict LRU items until within capacity."""
        if len(self._map) <= self._capacity:
            return
        # Lazy cleanup first: if every item is expired, they are all cleared here.
        self._purge_expired(now)
        # After the cleanup pass every node is non-expired, so the tail is
        # the least-recently-used non-expired item.
        while len(self._map) > self._capacity:
            tail_node = self._tail.prev
            assert tail_node is not None
            self._remove_node(tail_node)

    # ---------- public API ----------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and not expired; otherwise None.

        Accessing a valid key marks it most-recently-used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if time.monotonic() >= node.expires_at:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`, expiring `ttl` seconds from now (overrides default_ttl).

        When at capacity, the least-recently-used non-expired item is evicted;
        if every item is expired, all of them are cleared first.
        """
        if ttl is not None and ttl < 0:
            raise ValueError("ttl must be non-negative")
        now = time.monotonic()
        expires_at = now + (self._default_ttl if ttl is None else ttl)

        existing = self._map.get(key)
        if existing is not None:
            if now < existing.expires_at:
                existing.value = value
                existing.expires_at = expires_at
                self._move_to_front(existing)
                return
            # Stale entry: drop it and insert fresh below.
            self._remove_node(existing)

        node = _Node(key, value, expires_at)
        self._map[key] = node
        self._insert_front(node)
        self._enforce_capacity(now)

    def delete(self, key: str) -> bool:
        """Remove `key`; return True if it existed (expired or not)."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non-expired items (expired ones are purged lazily)."""
        now = time.monotonic()
        self._purge_expired(now)
        return len(self._map)
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. time.monotonic is mocked for deterministic time control."""

from unittest.mock import patch

from ttl_cache import TTLCache


@patch("time.monotonic")
def test_basic_get_put(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None

    cache.put("a", 2)  # update existing key
    assert cache.get("a") == 2
    assert cache.size() == 1


@patch("time.monotonic")
def test_capacity_eviction_lru_order(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" -> MRU, "b" is LRU
    cache.put("c", 3)           # evicts least-recently-used item "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size() == 2


@patch("time.monotonic")
def test_ttl_expiry(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    current["now"] = 10.0  # exactly at the expiry boundary -> expired
    assert cache.get("a") is None
    assert cache.size() == 0

    # When every item is expired, a new put clears all of them first.
    cache.put("x", 1)
    cache.put("y", 2)
    current["now"] = 50.0
    cache.put("z", 3)
    assert cache.size() == 1
    assert cache.get("z") == 3
    assert cache.get("x") is None


@patch("time.monotonic")
def test_custom_per_key_ttl(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("short", "s", ttl=5.0)
    cache.put("long", "l", ttl=50.0)
    cache.put("default", "d")

    current["now"] = 6.0
    assert cache.get("short") is None   # expired at t=5
    assert cache.get("long") == "l"     # valid until t=50
    assert cache.get("default") == "d"  # valid until t=10

    current["now"] = 11.0
    assert cache.get("default") is None
    assert cache.get("long") == "l"


@patch("time.monotonic")
def test_delete(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.delete("a") is True
    assert cache.delete("a") is False  # already removed
    assert cache.get("a") is None
    assert cache.size() == 1
    assert cache.delete("b") is True
    assert cache.size() == 0


@patch("time.monotonic")
def test_size_mixed_expired_valid(mock_monotonic):
    current = {"now": 0.0}
    mock_monotonic.side_effect = lambda: current["now"]

    cache = TTLCache(capacity=5, default_ttl=10.0)
    cache.put("short1", 1, ttl=5.0)
    cache.put("long", 2, ttl=100.0)
    cache.put("short2", 3, ttl=5.0)
    assert cache.size() == 3

    current["now"] = 6.0
    assert cache.size() == 1  # only "long" still valid
    assert cache.get("long") == 2

    current["now"] = 150.0
    assert cache.size() == 0
```

**Design notes**

- Structure: a `Dict[str, _Node]` for O(1) lookup plus a doubly-linked list (head = most-recently-used, tail = least-recently-used) with dummy sentinels so unlinking needs no special cases. No `OrderedDict`.
- `get` promotes the node to the head; `put` on a live key updates value/expiry and promotes; `put` on a dead key (or a new key) inserts at the head, then `_enforce_capacity` runs: it lazily purges expired nodes (which covers the "all items expired → clear them all" case) and then evicts from the tail (guaranteed non-expired after the purge) until within capacity.
- Expiry is `now >= expires_at`; time comes from `time.monotonic()`.
- Complexity: lookups, inserts, deletes and list moves are O(1); the lazy cleanup passes are amortized O(1) since each node is removed at most once (worst-case O(n) for a single cleanup pass, which is inherent to lazy expiry without a heap).
- One subtlety worth knowing: `size()`/`get()`/`put()` purge expired nodes on access, so an expired entry is never returned even if it's still linked.

Run the tests with:

```bash
python -m pytest test_ttl_cache.py -v
```