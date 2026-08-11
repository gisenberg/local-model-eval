```python
from typing import Any, Optional
import time


class _Node:
    """Doubly linked list node for LRU tracking."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """LRU cache with time-based expiration using doubly-linked list + hash map."""

    def __init__(self, capacity: int, default_ttl: float):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._dict: dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # most recently used
        self._tail: Optional[_Node] = None  # least recently used

    def _remove_node(self, node: _Node) -> None:
        """Unlink node from linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = None
        node.next = None

    def _add_to_front(self, node: _Node) -> None:
        """Insert node at head (most recently used)."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if not self._tail:
            self._tail = node

    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to head."""
        if node is self._head:
            return
        self._remove_node(node)
        self._add_to_front(node)

    def _evict_lru(self) -> None:
        """Evict least recently used non-expired item. If all expired, clear all."""
        now = time.monotonic()
        # First, remove all expired nodes from the linked list
        current = self._head
        while current:
            next_node = current.next
            if current.expires_at <= now:
                self._remove_node(current)
                del self._dict[current.key]
            current = next_node

        # If cache is empty after cleanup, nothing to evict
        if not self._tail:
            return

        # Evict the tail (least recently used non-expired)
        node = self._tail
        assert node is not None
        self._remove_node(node)
        del self._dict[node.key]

    def get(self, key: str) -> Optional[Any]:
        """Return value if exists and not expired, else None. Makes key most-recently-used."""
        node = self._dict.get(key)
        if node is None:
            return None
        now = time.monotonic()
        if node.expires_at <= now:
            # Expired - remove from cache
            self._remove_node(node)
            del self._dict[key]
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert/update key with optional custom TTL. Evicts LRU if at capacity."""
        now = time.monotonic()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        existing = self._dict.get(key)
        if existing:
            # Update existing node
            existing.value = value
            existing.expires_at = expires_at
            self._move_to_front(existing)
            return

        # New key
        node = _Node(key, value, expires_at)
        self._dict[key] = node
        self._add_to_front(node)

        # Check capacity - evict if needed
        if len(self._dict) > self.capacity:
            self._evict_lru()

    def delete(self, key: str) -> bool:
        """Remove key, return True if it existed (even if expired)."""
        node = self._dict.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """Return count of non-expired items (lazy cleanup on access)."""
        now = time.monotonic()
        # Remove expired nodes
        current = self._head
        while current:
            next_node = current.next
            if current.expires_at <= now:
                self._remove_node(current)
                del self._dict[current.key]
            current = next_node
        return len(self._dict)
```