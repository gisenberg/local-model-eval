# Gemma 4 26B-A4B 6bit MLX — lru_cache

## Thinking (6267 chars)

```


*   Goal: Implement an LRU (Least Recently Used) cache with TTL (Time To Live).
*   Requirements:
    *   Class `TTLCache(capacity, default_ttl)`.
    *   Methods: `get(key)`, `put(key, value, ttl=None)`, `delete(key)`, `size()`.
    *   Complexity: $O(1)$ average time.
    *   Data Structures: Doubly-linked list + Hash map (no `OrderedDict`).
    *   Time Tracking: `time.monotonic()`.
    *   Cleanup Strategy: Lazy cleanup on access.
    *   Code Quality: Type hints, docstrings.
    *   Testing: 6 `pytest` tests using `unittest.mock.patch` on `time.monotonic`.

    *   *Hash Map (Dictionary):* Maps `key` to a `Node` object. This allows $O(1)$ access.
    *   *Doubly-Linked List:* Stores the nodes. The "head" represents the Most Recently Used (MRU) and the "tail" represents the Least Recently Used (LRU).
    *   *Node Object:* Needs to store `key`, `value`, `expiry_time`, `prev`, and `next`.

    *   `__init__(capacity, default_ttl)`: Initialize capacity, default TTL, empty dict, and dummy head/tail nodes for the linked list.
    *   `_remove(node)`: Helper to detach a node from the list.
    *   `_add_to_front(node)`: Helper to add a node right after the dummy head.
    *   `put(key, value, ttl=None)`:
        1.  Calculate expiry: `now + (ttl if ttl is not None else default_ttl)`.
        2.  If key exists: Update value/expiry, move to front.
        3.  If key doesn't exist:
            *   If at capacity: Remove tail (LRU), delete from dict.
            *   Create new node, add to front, add to dict.
    *   `get(key)`:
        1.  If key not in dict: return `None`.
        2.  Get node.
        3.  Check expiry: `if now > node.expiry_time`:
            *   Delete node (remove from list and dict).
            *   Return `None`.
        4.  If valid: Move node to front, return value.
    *   `delete(key)`:
        1.  If key in dict: remove from list and dict, return `True`.
        2.  Else: return `False`.
    *   `size()`: Return length of dict.

    *   *Edge Case:* Capacity = 0 (though usually capacity > 0).
    *   *Edge Case:* TTL = 0 (immediate expiry).
    *   *Edge Case:* Updating an existing key with a new TTL.
    *   *Edge Case:* Deleting a non-existent key.

    *   Test 1: Basic Put/Get.
    *   Test 2: LRU Eviction (capacity limit).
    *   Test 3: TTL Expiration (lazy cleanup).
    *   Test 4: Updating existing key (updates value and moves to front).
    *   Test 5: Manual deletion.
    *   Test 6: TTL override (specific TTL vs default TTL).

    *   *Imports:* `time`, `typing`.
    *   *Node Class:* `key`, `value`, `expiry`, `prev`, `next`.
    *   *TTLCache Class:*
        *   `self.cache = {}`
        *   `self.head = Node(None, None, 0)`
        *   `self.tail = Node(None, None, 0)`
        *   `self.head.next = self.tail`
        *   `self.tail.prev = self.head`

    *   *Wait, the prompt says "lazy cleanup on access".* This means `get` checks the time. Does `put` check time? Usually, `put` just overwrites. Does `size()` check time? If we want `size()` to be accurate, we'd need to scan, but that's $O(N)$. The prompt says $O(1)$ average. So `size()` will return the count of items in the dict, even if some are expired but not yet accessed. This is standard for lazy cleanup.

    *   *Wait, the prompt says "no OrderedDict".* I will use a manual doubly-linked list.

    *   *Mocking `time.monotonic`:*
        ```python
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 100.0
            # ...
        ```

    *   *Class TTLCache:* Check.
    *   *Methods:* Check.
    *   *O(1):* Check.
    *   *Doubly-linked list + hash map:* Check.
    *   *time.monotonic:* Check.
    *   *Lazy cleanup:* Check.
    *   *Type hints/docstrings:* Check.
    *   *6 pytest tests with mock:* Check.

    *   *Self-Correction during coding:* In `put`, if the key exists, I must update the expiry time. If the key is new and capacity is reached, I must remove the node at `self.tail.prev`.

    *   *Test 1:* `test_basic_put_get`
    *   *Test 2:* `test_lru_eviction`
    *   *Test 3:* `test_ttl_expiration`
    *   *Test 4:* `test_update_key`
    *   *Test 5:* `test_delete`
    *   *Test 6:* `test_custom_ttl`

    *   *Wait, one detail:* If I `put` a key that is already expired, should I treat it as an update or a new entry? Standard behavior is to treat it as an update (overwrite).

    *   *Refining `get`:*
        ```python
        def get(self, key: Any) -> Optional[Any]:
            if key not in self.cache:
                return None
            node = self.cache[key]
            if time.monotonic() > node.expiry:
                self._remove(node)
                del self.cache[key]
                return None
            self._move_to_front(node)
            return node.value
        ```

    *   *Refining `put`:*
        ```python
        def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
            now = time.monotonic()
            expiry = now + (ttl if ttl is not None else self.default_ttl)
            if key in self.cache:
                node = self.cache[key]
                node.value = value
                node.expiry = expiry
                self._move_to_front(node)
            else:
                if len(self.cache) >= self.capacity:
                    # Evict LRU
                    lru_node = self.tail.prev
                    self._remove(lru_node)
                    del self.cache[lru_node.key]
                new_node = Node(key, value, expiry)
                self.cache[key] = new_node
                self._add_to_front(new_node)
        ```

    *   *Wait, what if capacity is 0?* The prompt implies a valid capacity, but I'll assume `capacity > 0`.

    *   *Final check on `size()`:* The prompt asks for `size()`. Since we use lazy cleanup, `size()` returns the number of items in the dictionary. This is $O(1)$.

    *   *Writing the tests:* Use `pytest`.

    (Self-Correction): Ensure `_remove` and `_add_to_front` are robust.
    `_remove(node)`: `node.prev.next = node.next`, `node.next.prev = node.prev`.
    `_add_to_front(node)`: `node.next = self.head.next`, `node.prev = self.head`, `self.head.next.prev = node`, `self.head.next = node`.

    Let's write the code.
```

## Output

```python
import time
from typing import Any, Optional, Dict


class Node:
    """A node in the doubly-linked list."""

    def __init__(
        self,
        key: Any,
        value: Any,
        expiry: float,
        prev: Optional["Node"] = None,
        next: Optional["Node"] = None,
    ):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev = prev
        self.next = next


class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) support.
    Uses a hash map and a doubly-linked list to achieve O(1) average time complexity.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}

        # Dummy head and tail to simplify doubly-linked list operations
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (Most Recently Used position)."""
        first_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = first_node
        if first_node:
            first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front of the list."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        Returns None if key is missing or expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy cleanup: check if the item has expired
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a value in the cache.

        :param key: The key to store.
        :param value: The value to store.
        :param ttl: Optional specific TTL for this key. If None, uses default_ttl.
        """
        now = time.monotonic()
        expiry = now + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict the Least Recently Used (LRU) item
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """
        Remove a key from the cache.

        :return: True if the key was found and deleted, False otherwise.
        """
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Return the current number of items in the cache (including potentially expired ones)."""
        return len(self.cache)


# --- Tests ---

import pytest
from unittest.mock import patch


def test_basic_put_get():
    """Test basic insertion and retrieval."""
    with patch("time.monotonic", return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("b") is None


def test_lru_eviction():
    """Test that the least recently used item is evicted when capacity is reached."""
    with patch("time.monotonic", return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # 'a' is now MRU, 'b' is LRU
        cache.put("c", 3)  # Should evict 'b'
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None


def test_ttl_expiration():
    """Test that items expire after their TTL has passed (lazy cleanup)."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)

        # Advance time past TTL
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0


def test_update_key():
    """Test that putting an existing key updates its value and TTL."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        
        # Update 'a' with new value and new TTL
        cache.put("a", 2, ttl=50)
        
        # Advance time to where original TTL would have expired
        mock_time.return_value = 115.0
        assert cache.get("a") == 2  # Should still be valid due to updated TTL


def test_delete():
    """Test manual deletion of keys."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None


def test_custom_ttl():
    """Test that specific TTLs override the default TTL."""
    with patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        
        # 'a' has short TTL, 'b' has long TTL
        cache.put("a", 1, ttl=5)
        cache.put("b", 2)
        
        mock_time.return_value = 110.0
        assert cache.get("a") is None  # Expired
        assert cache.get("b") == 2     # Still valid
```
