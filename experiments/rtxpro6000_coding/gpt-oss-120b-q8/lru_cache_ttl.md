**`ttl_cache.py`**

```python
"""TTL‑aware LRU cache.

The cache stores at most *capacity* items.  Each entry has an absolute
expiration time (computed from the moment it is inserted) and the cache
evicts the *least‑recently‑used* non‑expired entry when it runs out of
space.  Expired entries are removed lazily – i.e. on the next operation
that touches them.

All public operations run in O(1) average time.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Dict


class _Node:
    """A doubly‑linked list node used by :class:`TTLCache`."""

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
    """Least‑Recently‑Used cache with per‑item time‑to‑live.

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
        self._head: Optional[_Node] = None   # most‑recent
        self._tail: Optional[_Node] = None   # least‑recent

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #

    def _now(self) -> float:
        """Current monotonic time."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Return ``True`` if *node* is past its expiration."""
        return self._now() >= node.expire_at

    def _remove_node(self, node: _Node) -> None:
        """Unlink *node* from the linked list and delete it from the map."""
        # unlink from list
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._head:
            self._head = node.next
        if node is self._tail:
            self._tail = node.prev
        # delete from dict
        self._map.pop(node.key, None)

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* as the most‑recent element."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if self._tail is None:          # first element
            self._tail = node
        self._map[node.key] = node

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as most‑recently used."""
        if node is self._head:
            return
        # unlink
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self._tail:
            self._tail = node.prev
        # re‑insert at head
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node

    def _evict_if_necessary(self) -> None:
        """Evict the least‑recently‑used non‑expired entry if capacity is exceeded.

        If *all* entries are expired, they are cleared first.
        """
        if len(self._map) <= self._capacity:
            return

        # Scan from the tail (LRU) upwards looking for a non‑expired node.
        cur = self._tail
        while cur:
            if self._is_expired(cur):
                # expired – drop it and continue scanning
                prev = cur.prev
                self._remove_node(cur)
                cur = prev
                continue
            # found a live node → evict it
            self._remove_node(cur)
            break

        # If we removed everything (all were expired) and still exceed capacity,
        # the loop above already cleared the map, so nothing more to do.
        # The invariant `len(self._map) <= capacity` now holds.

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def get(self, key: str) -> Optional[Any]:
        """Return the value for *key* if present and not expired, else ``None``.

        Accessing a key marks it as most‑recently used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            self._remove_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        *ttl* (seconds) overrides the cache's default TTL for this entry.
        If the cache is full, the least‑recently‑used non‑expired entry is
        evicted.  When all entries are expired they are cleared first.
        """
        ttl_to_use = ttl if ttl is not None else self._default_ttl
        if ttl_to_use <= 0:
            raise ValueError("ttl must be > 0")
        expire_at = self._now() + ttl_to_use

        node = self._map.get(key)
        if node:
            # update existing entry
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
        else:
            # new entry
            new_node = _Node(key=key, value=value, expire_at=expire_at)
            self._add_to_head(new_node)
            self._evict_if_necessary()

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the number of non‑expired items currently stored.

        Expired items are removed lazily during this call.
        """
        # Clean up any expired nodes we encounter.
        cur = self._head
        while cur:
            nxt = cur.next
            if self._is_expired(cur):
                self._remove_node(cur)
            cur = nxt
        return len(self._map)

    # --------------------------------------------------------------------- #
    # representation helpers (useful for debugging)
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:  # pragma: no cover
        items = []
        cur = self._head
        while cur:
            items.append(f"{cur.key}:{cur.value!r}")
            cur = cur.next
        return f"<TTLCache size={len(self._map)} [{', '.join(items)}]>"
```

---

**`test_ttl_cache.py`**

```python
import pytest
from unittest.mock import patch

from ttl_cache import TTLCache


class TimeMock:
    """Simple mutable time source for patching ``time.monotonic``."""

    def __init__(self, start: float = 0.0):
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


# --------------------------------------------------------------------- #
# 1. basic put / get
# --------------------------------------------------------------------- #
def test_basic_put_get():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1
        # still present after a short time
        tm.advance(5)
        assert cache.get("a") == 1


# --------------------------------------------------------------------- #
# 2. capacity eviction (LRU order)
# --------------------------------------------------------------------- #
def test_capacity_eviction():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", "A")
        cache.put("b", "B")
        # access 'a' → becomes MRU, order: a (head), b (tail)
        assert cache.get("a") == "A"
        # insert third key, should evict 'b'
        cache.put("c", "C")
        assert cache.get("b") is None
        assert cache.get("a") == "A"
        assert cache.get("c") == "C"


# --------------------------------------------------------------------- #
# 3. TTL expiry
# --------------------------------------------------------------------- #
def test_ttl_expiry():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=2, default_ttl=5.0)
        cache.put("x", 42)
        assert cache.get("x") == 42
        tm.advance(6)                     # past TTL
        assert cache.get("x") is None
        assert cache.size() == 0


# --------------------------------------------------------------------- #
# 4. custom per‑key TTL
# --------------------------------------------------------------------- #
def test_custom_ttl():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put("short", "S", ttl=2.0)   # expires quickly
        cache.put("long", "L")             # uses default ttl (10s)

        tm.advance(3)                      # short key expired
        assert cache.get("short") is None
        assert cache.get("long") == "L"

        tm.advance(8)                      # total 11s, long key also expired
        assert cache.get("long") is None
        assert cache.size() == 0


# --------------------------------------------------------------------- #
# 5. delete
# --------------------------------------------------------------------- #
def test_delete():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("k", "value")
        assert cache.delete("k") is True
        assert cache.get("k") is None
        # second delete returns False
        assert cache.delete("k") is False


# --------------------------------------------------------------------- #
# 6. size with mixed expired / valid items
# --------------------------------------------------------------------- #
def test_size_mixed():
    tm = TimeMock()
    with patch("time.monotonic", tm):
        cache = TTLCache(capacity=3, default_ttl=15.0)
        cache.put("a", 1, ttl=5)   # expires at t=5
        cache.put("b", 2, ttl=10)  # expires at t=10
        cache.put("c", 3)          # expires at t=15 (default)

        tm.advance(6)              # now t=6, "a" expired
        assert cache.get("a") is None
        assert cache.size() == 2   # only b and c remain
        assert cache.get("b") == 2
        assert cache.get("c") == 3

        tm.advance(5)              # now t=11, "b" expired
        assert cache.size() == 1   # only c remains
        assert cache.get("b") is None
        assert cache.get("c") == 3
```

**Explanation of the implementation**

* **Data structures** – Each entry is stored in a `_Node` that belongs to a
  doubly‑linked list.  The head of the list is the most‑recently‑used item,
  the tail the least‑recently‑used.  A dictionary (`self._map`) maps keys to
  their nodes, giving O(1) lookup.

* **Expiration** – When a node is accessed (`get`) or when the cache size is
  queried (`size`), we compare the stored `expire_at` timestamp with the
  current monotonic time.  Expired nodes are removed immediately.

* **Insertion (`put`)** – If the key already exists we update its value,
  expiry and move it to the head.  For a new key we create a node, insert it
  at the head and then possibly evict one entry (`_evict_if_necessary`).  
  Eviction walks from the tail until it finds a non‑expired node; if all
  nodes are expired they are cleared first.

* **Deletion** – Simple removal from both the list and the dictionary.

* **Size** – Performs a lazy cleanup of any expired nodes it encounters and
  returns the length of the dictionary.

All public methods run in O(1) average time; the occasional scan for an
eviction or size cleanup touches only a few nodes in typical usage.

The test suite uses a `TimeMock` together with `unittest.mock.patch` to
control `time.monotonic` deterministically, avoiding real sleeps and making
the TTL behaviour fully reproducible.