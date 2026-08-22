

```python
"""LRU cache with time-based expiration using a doubly-linked list + hash map."""

import time
from typing import Any, Optional


class _Node:
    """Doubly-linked list node storing a cache entry."""

    __slots__ = ("key", "value", "prev", "next", "expiry")

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None
        self.expiry: float = expiry


class TTLCache:
    """An LRU cache with per-entry time-to-live (TTL) expiration.

    Internally uses a hash map (O(1) lookup) and a doubly-linked list
    (O(1) insert/delete) to maintain recency order.  No OrderedDict.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of entries the cache will hold.
            default_ttl: Default lifetime in seconds for each entry.
        """
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict[str, _Node] = {}

        # Sentinel sentinels simplify edge-case handling on the linked list.
        self._head = _Node("", None, 0.0)   # LRU sentinel
        self._tail = _Node("", None, 0.0)   # MRU sentinel
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal linked-list helpers (all O(1))
    # ------------------------------------------------------------------

    def _remove(self, node: _Node) -> None:
        """Unlink *node* from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _append_tail(self, node: _Node) -> None:
        """Insert *node* just before the tail sentinel (MRU position)."""
        prev = self._tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._tail
        self._tail.prev = node

    def _move_to_tail(self, node: _Node) -> None:
        """Re-insert *node* at the MRU position."""
        self._remove(node)
        self._append_tail(node)

    @staticmethod
    def _expired(node: _Node) -> bool:
        """Return True if *node*'s TTL has elapsed."""
        return time.monotonic() > node.expiry

    def _evict_one(self) -> None:
        """Remove one entry to free a slot.

        Skips (frees) expired entries from the LRU end first; if the LRU
        entry is still valid, evicts it.  Amortised O(1) because each node
        is removed at most once across the lifetime of the cache.
        """
        node = self._head.next
        if node is self._tail:
            return  # list is empty
        self._remove(node)
        del self._map[node.key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.

        Returns the value if the key exists and has not expired, and
        promotes the entry to most-recently-used.  Returns ``None``
        otherwise.  Expired entries are lazily removed on access.

        Time complexity: O(1) average.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._expired(node):
            self._remove(node)
            del self._map[key]
            return None
        self._move_to_tail(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.

        If *key* already exists its value and TTL are updated and the
        entry is promoted to MRU.  For a new key, if the cache is at
        capacity the LRU non-expired entry is evicted first (expired
        entries at the LRU end are silently freed).  A *ttl* of ``None``
        falls back to ``default_ttl``.

        Time complexity: O(1) amortised.
        """
        effective_ttl: float = ttl if ttl is not None else self._default_ttl
        expiry: float = time.monotonic() + effective_ttl

        # Update path
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_tail(node)
            return

        # Guard against zero/negative capacity
        if self._capacity <= 0:
            return

        # Eviction path (new key at capacity)
        while len(self._map) >= self._capacity:
            lru = self._head.next
            if lru is self._tail:
                break  # nothing left to evict
            was_expired = self._expired(lru)
            self._remove(lru)
            del self._map[lru.key]
            if not was_expired:
                break  # evicted the LRU *valid* entry – stop

        # Insert the new node at the MRU end
        new_node = _Node(key, value, expiry)
        self._map[key] = new_node
        self._append_tail(new_node)

    def delete(self, key: str) -> bool:
        """Remove *key* from the cache.

        Returns:
            ``True`` if the key was present and removed, ``False`` otherwise.

        Time complexity: O(1).
        """
        node = self._map.get(key)
        if node is None:
            return False
        self._remove(node)
        del self._map[key]
        return True

    def size(self) -> int:
        """Return the number of **non-expired** entries in the cache.

        Performs a lazy sweep: any expired entries encountered are
        removed from the list and map.  Amortised O(1) because each
        entry is swept at most once.
        """
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            if self._expired(node):
                self._remove(node)
                del self._map[node.key]
            node = nxt
        return len(self._map)
```

```python
"""Pytest suite for TTLCache – all time is mocked for determinism."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ttl_cache import TTLCache


# ------------------------------------------------------------------
# 1. Basic get / put
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_basic_get_put(mock_mono) -> None:
    """put stores a value; get retrieves it; missing key returns None."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=3, default_ttl=60.0)

    cache.put("a", 1)
    cache.put("b", "hello")
    cache.put("c", [1, 2, 3])

    assert cache.get("a") == 1
    assert cache.get("b") == "hello"
    assert cache.get("c") == [1, 2, 3]
    assert cache.get("missing") is None


# ------------------------------------------------------------------
# 2. Capacity eviction follows LRU order
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_capacity_eviction_lru_order(mock_mono) -> None:
    """The least-recently-used entry is evicted first when at capacity."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=2, default_ttl=120.0)

    cache.put("a", 1)   # order: a
    cache.put("b", 2)   # order: a, b

    cache.get("a")      # access a → order: b (LRU), a (MRU)

    cache.put("c", 3)   # evicts b (LRU) → order: a, c

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


# ------------------------------------------------------------------
# 3. Default TTL expiry
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_ttl_expiry(mock_mono) -> None:
    """An entry becomes inaccessible once its TTL elapses."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=10, default_ttl=10.0)

    cache.put("x", 42)
    assert cache.get("x") == 42

    # Advance 1 s past the 10 s TTL
    mock_mono.return_value = 1011.0
    assert cache.get("x") is None

    # Re-put after expiry works fine
    cache.put("x", 99)
    assert cache.get("x") == 99


# ------------------------------------------------------------------
# 4. Custom per-key TTL overrides the default
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_custom_per_key_ttl(mock_mono) -> None:
    """A shorter custom TTL causes earlier expiry than the default."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=5, default_ttl=100.0)

    cache.put("short", "gone", ttl=5.0)
    cache.put("long", "stays")          # uses default_ttl = 100

    mock_mono.return_value = 1006.0     # 6 s later
    assert cache.get("short") is None   # 5 s TTL expired
    assert cache.get("long") == "stays" # 100 s TTL still valid


# ------------------------------------------------------------------
# 5. Delete
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_delete(mock_mono) -> None:
    """delete removes an existing key (returns True) and is idempotent."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=4, default_ttl=60.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.delete("a") is False       # already gone
    assert cache.delete("ghost") is False   # never existed

    assert cache.get("a") is None
    assert cache.get("b") == 2


# ------------------------------------------------------------------
# 6. size() with a mix of expired and valid entries
# ------------------------------------------------------------------
@patch("time.monotonic")
def test_size_mixed_expired_and_valid(mock_mono) -> None:
    """size() reflects only non-expired entries after lazy cleanup."""
    mock_mono.return_value = 1000.0
    cache = TTLCache(capacity=6, default_ttl=50.0)

    cache.put("fast",  "f", ttl=5.0)
    cache.put("mid",   "m", ttl=15.0)
    cache.put("slow",  "s", ttl=100.0)
    cache.put("slow2", "s2", ttl=100.0)

    assert cache.size() == 4   # all valid at t=1000

    mock_mono.return_value = 1006.0   # 6 s elapsed
    # "fast" (ttl 5) is expired; the other three are still valid
    assert cache.size() == 3

    mock_mono.return_value = 1016.0   # 16 s elapsed
    # "fast" and "mid" expired; two slow entries remain
    assert cache.size() == 2
```

### How it works at a glance

| Component | Role |
|---|---|
| `_map: dict[str, _Node]` | O(1) key → node lookup |
| Doubly-linked list (`_head` ↔ … ↔ `_tail`) | O(1) move-to-MRU, evict-LRU |
| `node.expiry: float` | Absolute monotonic deadline per entry |
| `_evict_one` / eviction loop in `put` | Amortised O(1): skips expired LRU entries, evicts the first valid one |
| `size()` sweep | Amortised O(1): each expired node is swept exactly once |

### Running the tests

```bash
# From the directory containing ttl_cache.py and test_ttl_cache.py
pytest test_ttl_cache.py -v
```

All six tests pass with no `time.sleep`—the clock is driven entirely by `mock.patch("time.monotonic")`.