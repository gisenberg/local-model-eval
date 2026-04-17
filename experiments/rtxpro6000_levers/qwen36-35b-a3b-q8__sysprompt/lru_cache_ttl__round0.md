Looking at the requirements carefully, I need to:

1. Build a doubly-linked list + hash map LRU cache with TTL
2. Handle lazy expiration (remove expired items on access)
3. Handle eviction: skip expired items, evict LRU non-expired, or clear all if all expired
4. Maintain O(1) average time via amortized lazy cleanup
5. Track non-expired count for `size()`

Key invariants:
- `_count` tracks non-expired items; updated on every add/remove/evict/expire
- Doubly-linked list: `_head` (LRU end) → items → `_tail` (MRU end)
- Expired items are lazily removed during `get`, `put` (eviction), and `size`

```python
"""LRU Cache with Time-To-Live (TTL) expiration."""

import time
from typing import Any, Optional


class _Node:
    """Doubly-linked list node for LRU cache ordering."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache:
    """Least Recently Used (LRU) cache with time-based expiration.

    Uses a hash map for O(1) lookups and a doubly-linked list for O(1)
    LRU ordering. Expired items are lazily removed on access.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._map: dict[str, _Node] = {}
        self._count: int = 0

        # Sentinel nodes for the doubly-linked list
        self._head = _Node('', None, float('inf'))  # LRU end
        self._tail = _Node('', None, float('inf'))  # MRU end
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_to_tail(self, node: _Node) -> None:
        """Add a node right before the tail sentinel (MRU position)."""
        last = self._tail.prev
        last.next = node
        node.prev = last
        node.next = self._tail
        self._tail.prev = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.expiry

    def get(self, key: str) -> Optional[Any]:
        """Return the value for *key* if it exists and has not expired.

        Accessing a key promotes it to the most-recently-used position.
        Expired items are removed lazily.

        Args:
            key: The key to look up.

        Returns:
            The cached value, or None if the key is missing or expired.
        """
        node = self._map.get(key)
        if node is None:
            return None

        if self._is_expired(node):
            del self._map[key]
            self._remove(node)
            self._count