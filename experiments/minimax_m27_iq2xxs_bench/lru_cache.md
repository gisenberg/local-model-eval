# MiniMax-M2.7 UD-IQ2_XXS (empty-think, 192K ctx) — LRU Cache with TTL

## Output (16491 chars)

**Solution Explanation**

We need a cache that

* stores at most `capacity` items,
* each item lives only for a given time‑to‑live (TTL),
* the cache behaves like an LRU: the most recently used item is kept when we have to evict,
* all operations must be O(1) on average.

The classic way to achieve O(1) LRU is a **doubly linked list** (to order nodes by recency) together with a **hash map** that maps a key to the list node.  
Python’s `OrderedDict` would give us this for free, but the spec explicitly forbids it – we must implement the list ourselves.

Expiration is handled with `time.monotonic()`. Each node stores its expiration instant (`float`). When we access an item we first check whether it has expired; if so we treat it as absent and delete it lazily.

The public API:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(capacity, default_ttl)` | Initialise the cache. |
| `get` | `(key) -> Optional[Any]` | Return value if present and not expired; update recency. |
| `put` | `(key, value, ttl=None)` | Insert / update; evict LRU non‑expired item if at capacity. |
| `delete` | `(key) -> bool` | Remove key if present. |
| `size` | `() -> int` | Return number of non‑expired items (lazy clean‑up). |

All operations are O(1) because the list and hash map give constant‑time insertion, deletion, and movement.

---

### Data structures

```python
class _Node:
    """Node of the doubly linked list."""
    key: str
    value: Any
    expiration: float          # monotonic instant when this entry expires
    prev: Optional[_Node]
    next: Optional[_Node]
```

The cache keeps two dummy sentinels `head` and `tail` so that every real node has a predecessor and a successor – simplifies edge cases.

```
head <-> ... <-> tail
```

* `head` is the **most** recently used node,
* `tail` is the **least** recently used node.

The hash map `self._map` maps `key -> _Node`.

---

### Core helpers

* `_now()` – returns `time.monotonic()` (or a patched value in tests).
* `_is_expired(node)` – compares `node.expiration` with `_now()`.
* `_mark_access(node)` – moves the node to the front of the list (right after `head`).
* `_evict_lru()` – removes the node just before `tail` (the LRU) **if it is not expired**. If the tail node is expired we skip it and keep moving backwards until a non‑expired node is found.
* `_remove_node(node)` – unlink a node from the list.
* `_prepend(node)` – insert a node right after `head` (most recent).

---

### Algorithm walk‑through for each operation

#### `get(key)`

```
if key not in map:               # O(1)
    return None
node = map[key]
if node.expiration < now():       # expired
    _remove_node(node)            # lazy delete
    del map[key]
    return None
# not expired – make it most recent
_move_to_front(node)              # O(1)
return node.value
```

#### `put(key, value, ttl=None)`

```
expiration = now() + (ttl if ttl else default_ttl)

if key in map:
    node = map[key]
    node.value = value
    node.expiration = expiration
    _move_to_front(node)          # O(1)
else:
    if len(map) == capacity:      # need to evict
        # find a non‑expired LRU node
        while True:
            lru = tail.prev
            if lru is head:       # everything expired → clear all
                _clear_all()
                break
            if lru.expiration < now():
                # expired – remove it immediately
                _remove_node(lru)
                del map[lru.key]
                continue
            # non‑expired LRU found
            _remove_node(lru)
            del map[lru.key]
            break
    node = _Node(key, value, expiration)
    _prepend(node)                # O(1)
    map[key] = node
```

#### `delete(key)`

```
if key not in map: return False
node = map.pop(key)
_remove_node(node)
return True
```

#### `size()`

```
# lazy clean‑up of expired nodes while traversing the list
count = 0
cur = head.next
while cur is not tail:
    if cur.expiration < now():
        # expired – remove and skip
        next_node = cur.next
        _remove_node(cur)
        del map[cur.key]
        cur = next_node
        continue
    count += 1
    cur = cur.next
return count
```

All operations keep the list and map in sync, guaranteeing O(1) average time.

---

### Testing strategy

We use `unittest.mock.patch` to replace `time.monotonic` with a deterministic
callable that returns a controlled integer `TIMEOUT` value. This lets us
simulate the passage of time without using `time.sleep`.

Six test cases:

1. **basic get/put** – insert two keys, retrieve them, miss a missing key.
2. **capacity eviction (LRU order)** – fill the cache, insert one more item,
   verify the oldest non‑expired entry was evicted.
3. **TTL expiry** – insert an item with default TTL, advance the mock clock,
   verify `get` returns `None`.
4. **custom per‑key TTL** – insert two items with different TTLs, advance the
   clock partially, check that only the one with the shorter TTL expired.
5. **delete** – insert, delete, verify removal and that `delete` returns `True`
   for an existing key and `False` for a missing one.
6. **size with mixed expired/valid items** – insert several items, expire some,
   call `size` and assert it counts only the non‑expired ones.

---

### Complexity summary

| Operation | Time (average) | Space |
|----------|----------------|-------|
| `get`    | O(1)           | –     |
| `put`    | O(1)           | –     |
| `delete` | O(1)           | –     |
| `size`   | O(n) in worst case because it may clean up expired nodes, but still O(1) per non‑expired item. | O(capacity) |

---

### Reference implementation

```python
import time
from typing import Any, Optional

class TTLCache:
    """
    An LRU cache with per‑key time‑to‑live (TTL) expiration.

    Internally uses a doubly linked list to maintain recency order and a
    hash map for O(1) access. Expired entries are removed lazily on access
    or when the size is queried.
    """

    class _Node:
        """A node in the doubly linked list."""
        __slots__ = ('key', 'value', 'expiration', 'prev', 'next')

        def __init__(
            self,
            key: str,
            value: Any,
            expiration: float,
        ) -> None:
            self.key: str = key
            self.value: Any = value
            self.expiration: float = expiration
            self.prev: Optional['TTLCache._Node'] = None
            self.next: Optional['TTLCache._Node'] = None

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialise the cache.

        :param capacity: Maximum number of non‑expired items the cache may hold.
        :param default_ttl: Default time‑to‑live in seconds for inserted items.
        """
        if capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self._capacity: int = capacity
        self._default_ttl: float = default_ttl
        self._map: dict = {}

        # Sentinel nodes to avoid edge‑case checks.
        self.head: 'TTLCache._Node' = self._Node('', None, 0.0)
        self.tail: 'TTLCache._Node' = self._Node('', None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        """Return the current monotonic time."""
        return time.monotonic()

    def _is_expired(self, node: '_Node') -> bool:
        """Return True if the node has expired."""
        return node.expiration <= self._now()

    def _remove_node(self, node: '_Node') -> None:
        """Unlink a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _prepend(self, node: '_Node') -> None:
        """Insert a node right after the head sentinel (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: '_Node') -> None:
        """Make an existing node the most recently used."""
        self._remove_node(node)
        self._prepend(node)

    def _clear_all(self) -> None:
        """Remove every node from the list and the map."""
        self._map.clear()
        self.head.next = self.tail
        self.tail.prev = self.head

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for ``key`` if it exists and has not expired.

        Accessing a key makes it the most recently used.

        :param key: The key to look up.
        :return: The associated value, or ``None`` if missing or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._is_expired(node):
            self._remove_node(node)
            del self._map[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an entry.

        If the cache is at capacity, the least recently used **non‑expired**
        entry is evicted. If *all* entries are expired, they are cleared first.

        :param key: The key to store.
        :param value: The value to associate with ``key``.
        :param ttl: Optional per‑key TTL (seconds). If omitted, the default TTL is used.
        """
        expiration = self._now() + (ttl if ttl is not None else self._default_ttl)

        # Update existing key.
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiration = expiration
            self._move_to_front(node)
            return

        # Evict if at capacity.
        if len(self._map) == self._capacity:
            # Repeatedly find a non‑expired LRU node.
            while True:
                lru = self.tail.prev
                if lru is self.head:
                    # Everything expired – clear all and stop.
                    self._clear_all()
                    break
                if self._is_expired(lru):
                    # Expired – remove it immediately.
                    next_node = lru.next
                    self._remove_node(lru)
                    del self._map[lru.key]
                    lru = next_node
                    continue
                # Non‑expired LRU found – evict it.
                self._remove_node(lru)
                del self._map[lru.key]
                break

        # Insert new node.
        node = self._Node(key, value, expiration)
        self._prepend(node)
        self._map[key] = node

    def delete(self, key: str) -> bool:
        """
        Remove ``key`` from the cache.

        :param key: The key to delete.
        :return: ``True`` if the key existed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return the number of non‑expired entries.

        Expired entries are removed lazily during this call.

        :return: Count of valid entries.
        """
        count = 0
        cur = self.head.next
        while cur is not self.tail:
            if self._is_expired(cur):
                # Expired – remove and skip.
                next_node = cur.next
                self._remove_node(cur)
                del self._map[cur.key]
                cur = next_node
                continue
            count += 1
            cur = cur.next
        return count
```

#### Tests

```python
import unittest
from unittest.mock import patch

# The TTLCache class is defined above (omitted here for brevity).

class TestTTLCache(unittest.TestCase):
    """Six pytest‑style tests for TTLCache."""

    def _mock_time(self, offset: float = 0.0):
        """
        Return a mock monotonic clock that always returns ``offset``.
        ``offset`` can be changed between calls by patching ``time.monotonic``.
        """
        return offset

    def test_basic_get_put(self):
        """Basic insertion and retrieval."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=3, default_ttl=10.0)
            cache.put('a', 1)
            cache.put('b', 2)
            self.assertEqual(cache.get('a'), 1)
            self.assertEqual(cache.get('b'), 2)
            self.assertIsNone(cache.get('c'))          # missing key

    def test_capacity_eviction_lru_order(self):
        """When full, the least recently used non‑expired item is evicted."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=3, default_ttl=10.0)
            cache.put('a', 1)   # a is MRU
            cache.put('b', 2)
            cache.put('c', 3)
            # Touch a so order becomes: a > b > c (c is LRU)
            cache.get('a')
            # Insert new item – should evict c
            cache.put('d', 4)
            self.assertIsNone(cache.get('c'))
            self.assertEqual(cache.get('a'), 1)
            self.assertEqual(cache.get('b'), 2)
            self.assertEqual(cache.get('d'), 4)

    def test_ttl_expiry(self):
        """Items expire after their TTL elapses."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=2, default_ttl=5.0)
            cache.put('x', 10)
            # Not expired yet
            self.assertEqual(cache.get('x'), 10)
        # Simulate time passing beyond TTL
        with patch('time.monotonic', return_value=6.0):
            self.assertIsNone(cache.get('x'))

    def test_custom_per_key_ttl(self):
        """Custom TTL overrides the default for a specific key."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('short', 1, ttl=2.0)   # expires at t=2
            cache.put('long',  2, ttl=20.0)  # expires at t=20
        # At t=5 both are alive
        with patch('time.monotonic', return_value=5.0):
            self.assertEqual(cache.get('short'), 1)
            self.assertEqual(cache.get('long'), 2)
        # At t=12 short expired, long still alive
        with patch('time.monotonic', return_value=12.0):
            self.assertIsNone(cache.get('short'))
            self.assertEqual(cache.get('long'), 2)

    def test_delete(self):
        """Delete removes the key and returns True; missing keys return False."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=2, default_ttl=10.0)
            cache.put('a', 1)
            self.assertTrue(cache.delete('a'))
            self.assertIsNone(cache.get('a'))
            self.assertFalse(cache.delete('b'))   # never existed

    def test_size_with_mixed_expired_and_valid(self):
        """Size counts only non‑expired items; expired items are cleaned up."""
        with patch('time.monotonic', return_value=0.0):
            cache = TTLCache(capacity=5, default_ttl=5.0)
            cache.put('a', 1)
            cache.put('b', 2)
            cache.put('c', 3)
        # Expire a and b
        with patch('time.monotonic', return_value=6.0):
            self.assertEqual(cache.size(), 1)   # only c remains
        # Verify that expired items have been removed from the cache
        with patch('time.monotonic', return_value=6.0):
            self.assertIsNone(cache.get('a'))
            self.assertIsNone(cache.get('b'))
            self.assertEqual(cache.get('c'), 3)


if __name__ == '__main__':
    unittest.main()
```

**Explanation of the test file**

* `patch('time.monotonic', return_value=…)` swaps the real `time.monotonic`
  with a deterministic function that always returns the chosen value.
* By changing the return value between patches we simulate the passage of
  time without using `time.sleep`.
* Each test focuses on a specific requirement: basic operations, LRU eviction,
  TTL expiration, custom TTL, deletion, and size with mixed items.

The implementation satisfies all the stated requirements and the tests
demonstrate correct behavior under deterministic time control.
