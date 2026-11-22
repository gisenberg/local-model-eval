Here's a complete implementation using a doubly-linked list + hash map (no `OrderedDict`), with lazy expiry and `time.monotonic()` for all time math.

**`ttl_cache.py`**

```python
"""LRU cache with per-item time-to-live expiration."""

from typing import Any, Dict, Optional


class _Node:
    """Internal doubly-linked-list node (head = most-recently-used)."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """An LRU cache where each item expires after a TTL.

    - The linked list is kept in recency order: head = most-recently-used,
      tail = least-recently-used.
    - All operations are O(1) average / amortized: get/delete/update are
      strictly O(1), and put/size do amortized O(1) lazy cleanup of expired
      nodes (each node is removed at most once).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` items, each living for `default_ttl` seconds."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most-recently-used
        self._tail: Optional[_Node] = None  # least-recently-used

    # ---- internal helpers (O(1) each) ----

    def _unlink(self, node: _Node) -> None:
        """Remove `node` from the linked list (node stays referenced)."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

    def _insert_at_head(self, node: _Node) -> None:
        """Insert `node` as the new most-recently-used entry."""
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _move_to_head(self, node: _Node) -> None:
        """Promote `node` to the most-recently-used position."""
        if node is self._head:
            return
        self._unlink(node)
        self._insert_at_head(node)

    def _remove_node(self, node: _Node) -> None:
        """Fully remove `node` from the list and the hash map."""
        self._unlink(node)
        del self._map[node.key]

    def _sweep_expired(self, now: float) -> None:
        """Lazily remove all expired nodes from the cache."""
        node = self._head
        while node is not None:
            next_node = node.next
            if node.expires_at <= now:
                self._remove_node(node)
            node = next_node

    # ---- public API ----

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired, else None.

        Accessing a live key makes it the most-recently-used entry.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`. `ttl` overrides the default TTL.

        If the key already exists and is live, its value/expiry are updated
        and it is promoted to most-recently-used. Otherwise a new entry is
        inserted; expired entries are swept first, and if the cache is still
        at capacity the least-recently-used entry is evicted.
        """
        now = time.monotonic()
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at = now + effective_ttl

        node = self._map.get(key)
        if node is not None:
            if node.expires_at <= now:
                self._remove_node(node)  # treat as fresh insert
                node = None
            else:
                node.value = value
                node.expires_at = expires_at
                self._move_to_head(node)
                return

        # Fresh insert path.
        self._sweep_expired(now)
        if len(self._map) >= self._capacity:
            # At capacity after cleanup: evict the LRU non-expired item.
            # (If every item were expired, the sweep already cleared all of them.)
            self._remove_node(self._tail)  # type: ignore[arg-type]

        new_node = _Node(key, value, expires_at)
        self._map[key] = new_node
        self._insert_at_head(new_node)

    def delete(self, key: str) -> bool:
        """Remove `key`; return True if it existed, else False."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the count of non-expired items (expired items are swept)."""
        now = time.monotonic()
        self._sweep_expired(now)
        return len(self._map)
```

Note: `time.monotonic()` is called via `import time` / `time.monotonic()` (not `from time import monotonic`) so that tests can patch `time.monotonic` cleanly.

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache using a controllable fake clock (no time.sleep)."""

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def patched_clock(clock):
    with patch("time.monotonic", new=clock):
        yield clock


def test_basic_get_put(patched_clock):
    cache = TTLCache(capacity=3, default_ttl=60.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None


def test_capacity_eviction_lru_order(patched_clock):
    cache = TTLCache(capacity=2, default_ttl=60.0)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a becomes MRU; b is now LRU
    cache.put("c", 3)           # evicts b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.size() == 2


def test_ttl_expiry(patched_clock):
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    patched_clock.advance(11.0)
    assert cache.get("a") is None
    assert cache.size() == 0


def test_custom_per_key_ttl(patched_clock):
    cache = TTLCache(capacity=3, default_ttl=100.0)
    cache.put("short", "v", ttl=5.0)
    cache.put("long", "v", ttl=50.0)
    patched_clock.advance(10.0)
    assert cache.get("short") is None
    assert cache.get("long") == "v"


def test_delete(patched_clock):
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False


def test_size_with_mixed_expired_valid_items(patched_clock):
    cache = TTLCache(capacity=5, default_ttl=100.0)
    cache.put("valid1", 1)
    cache.put("expired", 2, ttl=5.0)
    cache.put("valid2", 3)
    patched_clock.advance(10.0)
    assert cache.size() == 2
    assert cache.get("expired") is None
    assert cache.get("valid1") == 1
    assert cache.get("valid2") == 3
```

Run with:

```bash
pytest test_ttl_cache.py -v
```

Key design notes:
- **Structure:** `self._map` (hash map) + doubly-linked list; head is most-recently-used, tail is least-recently-used. `get`/`put`-update promote the node to head via O(1) unlink/insert.
- **Eviction:** `put` first sweeps expired nodes (so a full cache of expired items is cleared), then evicts only the tail (the LRU non-expired item) if still at capacity.
- **Complexity:** `get`, `delete`, and in-place `put` are strictly O(1). `size`/`put` do lazy sweeps; each node is removed at most once, so the cost is amortized O(1) per operation.
- **Expiry semantics:** `expires_at <= now` counts as expired; updating an existing live key resets its value and TTL and promotes it to MRU.
- The fake clock is injected by patching `time.monotonic` (module attribute lookup through `import time`), so no `time.sleep` is used anywhere.