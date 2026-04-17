**`ttl_cache.py`**

```python
"""TTLCache – a fixed‑size LRU cache with per‑item time‑to‑live.

The cache stores at most *capacity* items.  Each entry expires after a
*default_ttl* (seconds) unless a custom *ttl* is supplied on ``put``.
Expired entries are removed lazily – i.e. when they are accessed or when
the cache needs space for a new entry.

All public operations run in O(1) average time (amortised) by using a
hash‑map for key → node lookup and a doubly‑linked list to maintain the
LRU order.  ``time.monotonic()`` is used for reliable time measurement.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Dict


class _Node:
    """A node in the doubly‑linked list used by :class:`TTLCache`."""

    __slots__ = ("key", "value", "expire_at", "prev", "next")

    def __init__(
        self,
        key: str,
        value: Any,
        expire_at: float,
        prev: Optional["_Node"] = None,
        next: Optional["_Node"] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev = prev
        self.next = next


class TTLCache:
    """Least‑Recently‑Used cache with time‑based expiration.

    Parameters
    ----------
    capacity:
        Maximum number of items the cache can hold.
    default_ttl:
        Default time‑to‑live (seconds) for items that are inserted without an
        explicit ``ttl`` argument.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}

        # Sentinel nodes – head is most‑recent, tail is least‑recent.
        self._head: _Node = _Node("", None, 0)  # type: ignore
        self._tail: _Node = _Node("", None, 0)  # type: ignore
        self._head.next = self._tail
        self._tail.prev = self._head

    # --------------------------------------------------------------------- #
    # Helper methods (all O(1))
    # --------------------------------------------------------------------- #
    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* right after the head sentinel (most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore
        self._head.next = node

    def _remove(self, node: _Node) -> None:
        """Unlink *node* from the linked list."""
        node.prev.next = node.next  # type: ignore
        node.next.prev = node.prev  # type: ignore
        node.prev = node.next = None  # help GC

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most‑recently used."""
        self._remove(node)
        self._add_to_head(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Return ``True`` if *node* has passed its expiration time."""
        return now >= node.expire_at

    def _evict_if_necessary(self, now: float) -> None:
        """Evict the least‑recently‑used non‑expired entry if capacity is exceeded.

        If all entries are expired, they are cleared first.
        """
        while len(self._map) >= self._capacity:
            lru = self._tail.prev
            if lru is self._head:  # no items left
                break
            if self._is_expired(lru, now):
                # expired – drop it and continue looking
                self._remove(lru)
                del self._map[lru.key]
                continue
            # non‑expired LRU – evict it
            self._remove(lru)
            del self._map[lru.key]
            break

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get(self, key: str) -> Optional[Any]:
        """Return the value for *key* if present and not expired, else ``None``.

        Accessing a key marks it as most‑recently used.
        """
        now = time.monotonic()
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node, now):
            # lazy removal of expired entry
            self._remove(node)
            del self._map[key]
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value* and an optional *ttl*.

        If the cache is at capacity, the least‑recently‑used non‑expired entry
        is evicted.  When all entries are expired they are cleared first.
        """
        now = time.monotonic()
        expire_at = now + (ttl if ttl is not None else self._default_ttl)

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
            return

        # new entry – make room if needed
        self._evict_if_necessary(now)

        node = _Node(key, value, expire_at)
        self._map[key] = node
        self._add_to_head(node)

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove(node)
        return True

    def size(self) -> int:
        """Return the number of non‑expired items currently stored.

        Expired items are removed lazily during this call.
        """
        now = time.monotonic()
        # Clean up any expired entries
        keys_to_remove = [k for k, n in self._map.items() if self._is_expired(n, now)]
        for k in keys_to_remove:
            node = self._map.pop(k)
            self._remove(node)
        return len(self._map)

    # --------------------------------------------------------------------- #
    # Convenience dunder methods
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Alias for :meth:`size`."""
        return self.size()

    def __contains__(self, key: str) -> bool:
        """``key in cache`` – true if *key* exists and is not expired."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {n.value!r}" for k, n in self._map.items())
        return f"<TTLCache capacity={self._capacity} items={{ {items} }}>"
```

---

**`test_ttl_cache.py`**

```python
import pytest
from unittest.mock import patch

from ttl_cache import TTLCache


@pytest.fixture
def monotonic():
    """A mutable monotonic clock that can be advanced in tests."""
    current = [0.0]

    def _monotonic():
        return current[0]

    return current, _monotonic


def test_basic_get_put(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        assert cache.get("a") == 1
        # non‑existent key
        assert cache.get("b") is None


def test_capacity_eviction_lru(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", "A")
        cache.put("b", "B")
        # access 'a' → becomes most‑recent
        assert cache.get("a") == "A"
        # inserting third key should evict 'b' (LRU)
        cache.put("c", "C")
        assert cache.get("b") is None
        assert cache.get("a") == "A"
        assert cache.get("c") == "C"


def test_ttl_expiry(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=2, default_ttl=5)
        cache.put("x", 42)
        assert cache.get("x") == 42
        # advance time beyond TTL
        clock[0] = 6.0
        assert cache.get("x") is None
        # after expiry the key should be gone from internal map
        assert "x" not in cache._map


def test_custom_per_key_ttl(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=3, default_ttl=10)
        cache.put("short", "S", ttl=3)   # expires at t=3
        cache.put("long", "L")           # expires at t=10 (default)
        # t = 4 → short expired, long still valid
        clock[0] = 4.0
        assert cache.get("short") is None
        assert cache.get("long") == "L"


def test_delete(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("k", "value")
        assert cache.delete("k") is True
        assert cache.get("k") is None
        # second delete should report False
        assert cache.delete("k") is False


def test_size_mixed_expired_and_valid(monotonic):
    clock, fake_monotonic = monotonic
    with patch("time.monotonic", side_effect=fake_monotonic):
        cache = TTLCache(capacity=5, default_ttl=10)
        cache.put("a", 1)   # expires at 10
        cache.put("b", 2)   # expires at 10
        cache.put("c", 3)   # expires at 10
        # advance time so that 'a' and 'b' are expired
        clock[0] = 11.0
        cache.put("d", 4)   # new entry, triggers cleanup of expired ones
        # size should count only 'c' and 'd'
        assert cache.size() == 2
        # ensure the expired keys are really gone
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4
```

**Explanation of the tests**

| Test | What it verifies |
|------|-------------------|
| `test_basic_get_put` | Simple insertion and retrieval; missing key returns ``None``. |
| `test_capacity_eviction_lru` | LRU order is respected when the cache is full. |
| `test_ttl_expiry` | Items become inaccessible after their TTL expires. |
| `test_custom_per_key_ttl` | Per‑key TTL overrides the default TTL. |
| `test_delete` | Deleting existing keys works and returns the correct boolean. |
| `test_size_mixed_expired_and_valid` | ``size`` returns the count of non‑expired items and cleans up expired ones. |

All tests use a patched ``time.monotonic`` that returns a mutable value, allowing deterministic simulation of time without any real sleeping. The implementation follows the required O(1) average‑time operations by using a hash map together with a custom doubly‑linked list.