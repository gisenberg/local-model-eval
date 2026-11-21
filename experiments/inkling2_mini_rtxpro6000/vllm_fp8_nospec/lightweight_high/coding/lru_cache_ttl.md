Here's a complete implementation using a doubly-linked list (MRU at head, LRU at tail) plus a hash map — no `OrderedDict`.

## Implementation — `ttl_cache.py`

```python
"""A least-recently-used cache with time-to-live expiration."""

import time
from typing import Any, Dict, Optional


class _Node:
    """Doubly-linked-list node holding one cache entry.

    Head of the list = most-recently-used; tail = least-recently-used.
    """

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: str, value: Any, expires_at: float) -> None:
        self.key: str = key
        self.value: Any = value
        self.expires_at: float = expires_at
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class TTLCache:
    """LRU cache with per-entry TTL expiration.

    Backed by a hash map (key -> node) plus a doubly-linked list ordered from
    most-recently-used (head) to least-recently-used (tail).

    Average-case time complexity: get/put/delete are O(1); size() is O(n) due
    to lazy cleanup of expired entries.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Create a cache holding at most `capacity` entries.

        Each entry expires `default_ttl` seconds after insertion/update.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._map: Dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most recently used
        self._tail: Optional[_Node] = None  # least recently used

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _insert_head(self, node: _Node) -> None:
        """Insert `node` at the head (most-recently-used position)."""
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the linked list (hash map entry untouched)."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = None
        node.next = None

    def _remove_node(self, node: _Node) -> None:
        """Unlink `node` from the list and drop it from the hash map."""
        self._unlink(node)
        self._map.pop(node.key, None)

    def _move_to_head(self, node: _Node) -> None:
        """Promote `node` to the most-recently-used position."""
        if node is self._head:
            return
        self._unlink(node)
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        node.prev = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """Return the value for `key` if it exists and is unexpired.

        Returns None otherwise. A hit makes `key` most-recently-used;
        an expired entry is removed lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if node.expires_at <= time.monotonic():
            self._remove_node(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update `key` with `value`.

        `ttl` (seconds) overrides the cache's default_ttl for this entry.
        When at capacity, expired entries are cleared first (if everything
        is expired, all entries are removed); otherwise the least-recently-used
        non-expired entry is evicted. The updated key becomes most-recently-used.
        """
        now = time.monotonic()
        existing = self._map.get(key)
        if existing is not None:
            # Update path: size unchanged, no eviction needed.
            self._remove_node(existing)
        elif len(self._map) >= self.capacity:
            # New key at capacity: drop expired tail entries, then evict the
            # least-recently-used non-expired entry.
            node = self._tail
            while node is not None and node.expires_at <= now:
                prev_node = node.prev
                self._remove_node(node)
                node = prev_node
            if node is not None:
                self._remove_node(node)
        ttl_value = self.default_ttl if ttl is None else ttl
        node = _Node(key, value, now + ttl_value)
        self._insert_head(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """Remove `key` if present. Returns True if it existed."""
        node = self._map.get(key)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return the count of non-expired entries (lazy cleanup)."""
        now = time.monotonic()
        node = self._head
        while node is not None:
            next_node = node.next
            if node.expires_at <= now:
                self._remove_node(node)
            node = next_node
        return len(self._map)
```

## Tests — `test_ttl_cache.py`

```python
"""pytest tests for TTLCache — deterministic time via mocked time.monotonic."""

from unittest.mock import patch

from ttl_cache import TTLCache


def test_basic_get_put():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=10.0)

        assert cache.get("x") is None

        cache.put("x", "value")
        assert cache.get("x") == "value"

        cache.put("x", "updated")  # update overwrites
        assert cache.get("x") == "updated"
        assert cache.size() == 1


def test_capacity_eviction_lru_order():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=2, default_ttl=60.0)

        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.get("a") == 1  # "a" becomes MRU; "b" is now LRU

        cache.put("c", 3)  # at capacity -> evicts "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_ttl_expiry():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=10.0)

        cache.put("a", 42)

        mock_time.return_value = 1005.0
        assert cache.get("a") == 42

        mock_time.return_value = 1010.0  # expiry boundary: now >= expires_at
        assert cache.get("a") is None
        assert cache.size() == 0


def test_custom_per_key_ttl():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=10.0)

        cache.put("short", 1, ttl=5.0)  # expires at 1005
        cache.put("long", 2)            # default ttl -> expires at 1010

        mock_time.return_value = 1005.5
        assert cache.get("short") is None
        assert cache.get("long") == 2


def test_delete():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=3, default_ttl=10.0)

        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False
        assert cache.size() == 0


def test_size_with_mixed_expired_and_valid_items():
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 1000.0
        cache = TTLCache(capacity=5, default_ttl=10.0)

        cache.put("a", 1, ttl=5.0)    # expires at 1005
        cache.put("b", 2, ttl=100.0)  # expires at 1100
        cache.put("c", 3, ttl=5.0)    # expires at 1005

        assert cache.size() == 3

        mock_time.return_value = 1006.0
        assert cache.size() == 1  # lazy cleanup leaves only "b"
        assert cache.get("b") == 2
```

## Notes

- **Eviction order in `put`**: when a *new* key arrives at capacity, the tail (LRU) is scanned inward; expired nodes are removed first (which clears everything if all items are expired), and the first non-expired tail node — the LRU non-expired item — is then evicted. Updating an existing key skips eviction entirely (size is unchanged).
- **Lazy expiry**: expired nodes are removed on `get`/`size`/`put`-eviction; they may otherwise sit in the list invisibly until touched.
- **Expiry boundary**: an entry is considered expired when `now >= expires_at`.
- **Time source**: everything uses `time.monotonic()` (via the `time` module), so tests patch `time.monotonic` to control the clock — no `time.sleep`.
- **Complexity**: `get`/`put`/`delete` are O(1) average (the eviction scan is usually one iteration; O(n) only when every item is expired, matching the "clear them all first" spec). `size()` is O(n) because it must scan to find expired items — inherent to the requested lazy cleanup.

Run the tests with:

```bash
pytest test_ttl_cache.py -v
```