```python
import time
from typing import Any, Optional, Dict, Union

class TTLCache:
    """
    LRU cache with time-based expiration.

    Uses a doubly-linked list + hash map for O(1) average operations.
    Expired items are removed lazily on access.
    """

    class _Node:
        __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')
        def __init__(self, key: str, value: Any, expires_at: float):
            self.key = key
            self.value = value
            self.expires_at = expires_at
            self.prev = None
            self.next = None

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._size = 0  # count of non-expired items
        self._map: Dict[str, TTLCache._Node] = {}
        # sentinel head and tail
        self._head = TTLCache._Node("", None, 0.0)
        self._tail = TTLCache._Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list and update size."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after head (most recently used)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
        self._size += 1

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """
        Evict the least recently used non-expired item.
        If all items are expired, clear them all first.
        """
        now = time.monotonic()
        # Remove expired items from the tail until we find a non-expired one
        while self._tail.prev != self._head and self._tail.prev.expires_at <= now:
            node = self._tail.prev
            self._remove_node(node)
            del self._map[node.key]
        # If still at capacity, evict the tail (which is now non-expired)
        if self._size >= self._capacity:
            node = self._tail.prev
            self._remove_node(node)
            del self._map[node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Return value if key exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        node = self._map.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            # expired
            self._remove_node(node)
            del self._map[key]
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If key exists, update value and TTL, move to head.
        If at capacity, evict LRU non-expired item.
        Custom ttl overrides default_ttl.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = now + effective_ttl

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_head(node)
        else:
            # If at capacity, evict first
            if self._size >= self._capacity:
                self._evict_lru()
            node = TTLCache._Node(key, value, expires_at)
            self._add_to_head(node)
            self._map[key] = node

    def delete(self, key: str) -> bool:
        """
        Remove key from cache. Return True if it existed, else False.
        """
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return count of non-expired items.
        Lazy cleanup: expired items are removed on access.
        """
        return self._size
```