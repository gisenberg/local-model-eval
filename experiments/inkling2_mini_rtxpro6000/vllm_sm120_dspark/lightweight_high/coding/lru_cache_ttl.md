Here's a complete implementation. Design: a hash map (`key → node`) for O(1) lookup, plus a doubly-linked list ordered from most-recently-used (head side) to least-recently-used (tail side). Expiration is lazy — expired entries are removed when touched.

**`ttl_cache.py`**

```python
"""LRU cache with per-entry time-to-live (TTL) expiration."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _Node:
    """A single doubly-linked-list node."""

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """Fixed-capacity LRU cache with per-entry TTL expiration.

    Internal structure: a hash map for O(1) lookup and a doubly-linked
    list ordered most-recently-used (head side) to least-recently-used
    (tail side). Expiration is lazy: expired entries are removed when
    they are touched.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` entries with the given default TTL (seconds)."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        # Sentinel head/tail so end-of-list never requires None checks.
        self._head: _Node = _Node("", None, 0.0)  # MRU end
        self._tail: _Node = _Node("", None, 0.0)  # LRU end
        self._head.next = self._tail
        self._tail.prev = self._head

    # ---------- internal helpers ----------

    def _insert_front(self, node: _Node) -> None:
        """Link node immediately behind the head sentinel (most recently used)."""
        first = self._head.next
        self._head.next = node
        node.prev = self._head
        node.next = first
        first.prev = node

    def _detach(self, node: _Node) -> None:
        """Unlink node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _move_to_front(self, node: _Node) -> None:
        """Promote node to the most-recently-used position."""
        self._detach(node)
        self._insert_front(node)

    def _remove_node(self, node: _Node) -> None:
        """Unlink node and drop it from the hash map."""
        self._detach(node)
        self._map.pop(node.key, None)

    def _evict_if_needed(self, now: float) -> None:
        """Make room for one new entry.

        Drops expired entries starting from the LRU end (so if every entry
        is expired, they are all cleared). If the cache is still full, the
        least-recently-used live entry is evicted.
        """
        node = self._tail.prev
        while node is not self._head and node.expires_at <= now:
            prev_node = node.prev
            self._remove_node(node)
            node = prev_node
        if len(self._map) >= self.capacity:
            node = self._tail.prev
            if node is not self._head:
                self._remove_node(node)

    # ---------- public API ----------

    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if present and unexpired, else None.

        A hit promotes the key to most-recently-used; an expired entry is
        removed as a side effect.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            self._remove_node(node)
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key`. `ttl` overrides the default TTL.

        At capacity, the least-recently-used non-expired entry is evicted
        (expired entries are cleared first). The inserted/updated key
        becomes the most-recently-used entry.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if effective_ttl <= 0:
            raise ValueError("ttl must be > 0")

        node = self._map.get(key)
        if node is not None and node.expires_at > now:
            # Live entry: update in place and promote.
            node.value = value
            node.expires_at = now + effective_ttl
            self._move_to_front(node)
            return
        if node is not None:
            # Expired entry: drop it before re-inserting.
            self._remove_node(node)

        self._evict_if_needed(now)
        new_node = _Node(key, value, now + effective_ttl)
        self._map[key] = new_node
        self._insert_front(new_node)

    def delete(self, key: str) -> bool:
        """Remove `key` from the cache. Return True if it was present."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of live (non-expired) entries.

        Expired entries are removed as a side effect (lazy cleanup).
        Note: this sweeps the list, so it is O(n); get/put/delete are O(1)
        (amortized for `put`'s eviction pass).
        """
        now = time.monotonic()
        node = self._tail.prev
        while node is not self._head:
            prev_node = node.prev
            if node.expires_at <= now:
                self._remove_node(node)
            node = prev_node
        return len(self._map)
```

**`test_ttl_cache.py`**

```python
"""Tests for TTLCache. Run with: python -m pytest test_ttl_cache.py -v"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Dict
from unittest.mock import patch

from ttl_cache import TTLCache


@contextmanager
def fake_clock(start: float = 0.0) -> Iterator[Dict[str, float]]:
    """Patch time.monotonic with a controllable fake clock.

    Yields a state dict whose "now" key can be mutated to advance time.
    """
    state: Dict[str, float] = {"now": start}

    def _monotonic() -> float:
        return state["now"]

    with patch("time.monotonic", new=_monotonic):
        yield state


def test_basic_get_put():
    with fake_clock(100.0):
        cache = TTLCache(3, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None
        cache.put("a", 99)  # update existing key
        assert cache.get("a") == 99
        assert cache.size() == 2


def test_capacity_evicts_least_recently_used():
    with fake_clock(0.0):
        cache = TTLCache(2, 100.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1  # promotes "a" -> "b" is now LRU
        cache.put("c", 3)           # must evict "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with fake_clock(0.0) as clock:
        cache = TTLCache(5, 10.0)
        cache.put("a", 1)
        clock["now"] = 9.9
        assert cache.get("a") == 1   # not yet expired
        clock["now"] = 10.0          # boundary: expired at exactly expires_at
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with fake_clock(0.0) as clock:
        cache = TTLCache(5, 100.0)
        cache.put("short", "s", ttl=5.0)
        cache.put("long", "l")  # uses default TTL (100.0)
        clock["now"] = 6.0
        assert cache.get("short") is None
        assert cache.get("long") == "l"


def test_delete():
    with fake_clock(0.0):
        cache = TTLCache(3, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.delete("a") is True
        assert cache.delete("a") is False  # already gone
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size() == 1


def test_size_with_mixed_expired_and_valid():
    with fake_clock(0.0) as clock:
        cache = TTLCache(5, 10.0)
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=100.0)
        cache.put("c", 3, ttl=5.0)
        clock["now"] = 6.0
        assert cache.size() == 1  # only "b" survives; expired items lazily removed
        assert cache.get("b") == 2
        assert cache.get("a") is None
        assert cache.size() == 1


def test_put_clears_all_expired_items_first():
    """Bonus edge case: when everything is expired, put clears it all."""
    with fake_clock(0.0) as clock:
        cache = TTLCache(2, 10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        clock["now"] = 11.0  # both expired
        cache.put("c", 3)    # must clear expired items, not evict a live one
        assert cache.get("c") == 3
        assert cache.size() == 1
```

Notes on the design:

- **O(1) behavior:** `get`, `delete`, and the update path of `put` are strictly O(1). `put`'s eviction pass walks the tail removing *expired* nodes, but each node is removed permanently once, so it's amortized O(1). `size()` is O(n) by nature — it sweeps the list for lazy cleanup, which is exactly what "expired items removed on access" implies.
- **Expiry boundary:** an entry expires when `now >= expires_at` (i.e., `expires_at <= now`), which is deterministic and covered by the boundary test.
- **Eviction order:** at capacity, expired entries at the LRU end are dropped first; only if the cache is still full is the LRU *live* entry evicted. If everything is expired, the walk removes them all before the insert.
- **Tests:** `time.monotonic` is patched via `unittest.mock.patch("time.monotonic", ...)` with a mutable fake clock, so no `time.sleep` and fully deterministic time control.

Run with:

```bash
python -m pytest test_ttl_cache.py -v
```

One caveat to be aware of: `size()` is O(n), which is the honest cost of an exact live-count with lazy expiration and no background timer. If you need a strictly O(1) `size()`, you'd add a min-heap of `(expires_at, key)` for amortized cleanup — happy to sketch that variant if you want it.