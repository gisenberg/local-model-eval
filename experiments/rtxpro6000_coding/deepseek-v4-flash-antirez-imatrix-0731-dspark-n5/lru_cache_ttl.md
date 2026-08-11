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
        """
        Initialize cache with capacity and default TTL.

        Args:
            capacity: Maximum number of items (must be > 0)
            default_ttl: Default time-to-live in seconds for items
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._dict: dict[str, _Node] = {}
        self._head: Optional[_Node] = None  # Most recently used
        self._tail: Optional[_Node] = None  # Least recently used

    def _remove_node(self, node: _Node) -> None:
        """Remove node from linked list."""
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
        """Add node to front (most recently used position)."""
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if not self._tail:
            self._tail = node

    def _move_to_front(self, node: _Node) -> None:
        """Move existing node to front."""
        if node is self._head:
            return
        self._remove_node(node)
        self._add_to_front(node)

    def _is_expired(self, node: _Node, now: float) -> bool:
        """Check if node is expired at given time."""
        return node.expires_at <= now

    def _evict_lru(self, now: float) -> None:
        """Evict least recently used non-expired item. If all expired, clear all."""
        # First check if all items are expired
        if self._tail and self._is_expired(self._tail, now):
            # All items are expired (since tail is LRU, if it's expired, all are)
            self._clear_all()
            return

        # Remove expired items from tail until we find a valid one
        while self._tail and self._is_expired(self._tail, now):
            expired = self._tail
            self._remove_node(expired)
            del self._dict[expired.key]
            # Continue checking next tail

        # Now evict the LRU (which is guaranteed non-expired)
        if self._tail:
            evict = self._tail
            self._remove_node(evict)
            del self._dict[evict.key]

    def _clear_all(self) -> None:
        """Remove all items from cache."""
        self._dict.clear()
        self._head = None
        self._tail = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Value if present and valid, else None
        """
        node = self._dict.get(key)
        if node is None:
            return None

        now = time.monotonic()
        if self._is_expired(node, now):
            # Remove expired node
            self._remove_node(node)
            del self._dict[key]
            return None

        # Move to front (most recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key with value and optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Custom TTL in seconds (overrides default)
        """
        now = time.monotonic()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        if key in self._dict:
            # Update existing node
            node = self._dict[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
        else:
            # Insert new node
            node = _Node(key, value, expires_at)
            self._dict[key] = node
            self._add_to_front(node)

            # Check capacity
            if len(self._dict) > self.capacity:
                self._evict_lru(now)

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed, False otherwise
        """
        node = self._dict.pop(key, None)
        if node is None:
            return False
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return count of non-expired items (lazy cleanup).

        Returns:
            Number of valid (non-expired) items
        """
        now = time.monotonic()
        # Remove expired items from tail
        while self._tail and self._is_expired(self._tail, now):
            expired = self._tail
            self._remove_node(expired)
            del self._dict[expired.key]

        return len(self._dict)
```