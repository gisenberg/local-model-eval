from typing import Any, Optional, Dict
import time


class Node:
    """Doubly-linked list node storing key, value, and expiry time."""
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expires_at: float):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    A thread-unsafe LRU cache with TTL expiration.
    Uses a doubly-linked list (for O(1) LRU ordering) and a hash map (for O(1) lookup).
    All operations are O(1) average time.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of non-expired items.
        :param default_ttl: Default time-to-live in seconds for items without custom TTL.
        """
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, Node] = {}  # key -> Node
        # Dummy head (most recently used) and tail (least recently used)
        self._head = Node("", None, float('inf'))
        self._tail = Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: Node) -> None:
        """Remove node from doubly-linked list (O(1))."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node
        node.prev = node.next = None

    def _append(self, node: Node) -> None:
        """Append node right before tail (most recently used) (O(1))."""
        prev_node = self._tail.prev
        if prev_node:
            prev_node.next = node
            node.prev = prev_node
        node.next = self._tail
        self._tail.prev = node

    def _move_to_end(self, node: Node) -> None:
        """Move node to most-recently-used position (O(1))."""
        self._remove(node)
        self._append(node)

    def _evict_lru(self) -> None:
        """Evict the least-recently-used *non-expired* item (O(1))."""
        # Start from head.next (LRU end)
        curr = self._head.next
        while curr is not self._tail:
            if curr.expires_at > time.monotonic():
                # Found first non-expired item → evict it (it's LRU among valid ones)
                self._remove(curr)
                del self._map[curr.key]
                return
            curr = curr.next
        # All items expired → clear all (lazy cleanup)
        self.clear_expired()

    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired."""
        return node.expires_at <= time.monotonic()

    def _cleanup_if_needed(self, key: str) -> None:
        """
        Ensure key is not expired and remove if so.
        Returns True if key was removed (expired or didn't exist).
        """
        if key not in self._map:
            return
        node = self._map[key]
        if self._is_expired(node):
            self._remove(node)
            del self._map[key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.
        Accessing a key makes it most-recently-used.

        :param key: Key to retrieve.
        :return: Value if found and not expired, else None.
        """
        self._cleanup_if_needed(key)
        if key not in self._map:
            return None
        node = self._map[key]
        self._move_to_end(node)  # Mark as recently used
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.

        :param key: Key to insert/update.
        :param value: Value to store.
        :param ttl: Optional custom TTL in seconds; uses default if None.
        """
        if ttl is None:
            ttl = self.default_ttl
        elif ttl < 0:
            raise ValueError("ttl must be non-negative")

        current_time = time.monotonic()
        expires_at = current_time + ttl

        # If key exists, update and move to end
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expires_at = expires_at
            self._move_to_end(node)
            return

        # If at capacity, evict LRU non-expired item
        if len(self._map) >= self.capacity:
            self._evict_lru()

        # Insert new node
        node = Node(key, value, expires_at)
        self._map[key] = node
        self._append(node)

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        :param key: Key to delete.
        :return: True if key existed (and was removed), False otherwise.
        """
        self._cleanup_if_needed(key)
        if key not in self._map:
            return False
        node = self._map.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """
        Return count of non-expired items.

        :return: Number of non-expired items.
        """
        # Lazy cleanup: remove expired items on access
        expired_keys = [k for k, node in self._map.items() if self._is_expired(node)]
        for k in expired_keys:
            node = self._map.pop(k)
            self._remove(node)
        return len(self._map)

    def clear_expired(self) -> None:
        """Remove all expired items (lazy cleanup helper)."""
        expired_keys = [k for k, node in self._map.items() if self._is_expired(node)]
        for k in expired_keys:
            node = self._map.pop(k)
            self._remove(node)
