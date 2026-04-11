import time
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CacheNode:
    """Node in the doubly-linked list representing cache items."""
    key: str
    value: Any
    ttl: float
    expiry_time: float
    prev: Optional['CacheNode'] = None
    next: Optional['CacheNode'] = None

class TTLCache:
    """LRU cache with time-based expiration for items.

    Uses a doubly-linked list and hash map to maintain O(1) average time complexity
    for all operations. Expired items are only removed when accessed (lazy cleanup).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """Initialize the cache with a given capacity and default TTL.

        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for items without custom TTL
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be a positive number")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheNode] = {}
        self.head: Optional[CacheNode] = None
        self.tail: Optional[CacheNode] = None
        self.size_count = 0

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache (lazy cleanup)."""
        current_time = time.monotonic()
        keys_to_remove = []

        # Find all expired keys
        for key, node in self.cache.items():
            if node.expiry_time <= current_time:
                keys_to_remove.append(key)

        # Remove them
        for key in keys_to_remove:
            self.delete(key)

    def _add_to_head(self, node: CacheNode) -> None:
        """Add a node to the head of the linked list (most recently used)."""
        node.prev = None
        node.next = self.head

        if self.head:
            self.head.prev = node
        self.head = node

        if not self.tail:
            self.tail = node

    def _remove_node(self, node: CacheNode) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _move_to_head(self, node: CacheNode) -> None:
        """Move an existing node to the head of the linked list."""
        if node == self.head:
            return

        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Evict the least recently used item from the cache."""
        if not self.tail:
            return

        lru_node = self.tail
        if lru_node.expiry_time > time.monotonic():
            # Only evict if not expired
            self.delete(lru_node.key)

    def get(self, key: str) -> Optional[Any:
        """Get a value from the cache if it exists and hasn't expired.

        Args:
            key: The key to look up in the cache

        Returns:
            The value if the key exists and hasn't expired, None otherwise
        """
        self._cleanup_expired()

        if key not in self.cache:
            return None

        node = self.cache[key]
        if node.expiry_time <= time.monotonic():
            return None

        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a value in the cache with optional TTL.

        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Time-to-live in seconds for this specific key. If None, uses default_ttl
        """
        self._cleanup_expired()

        current_time = time.monotonic()
        expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.ttl = ttl if ttl is not None else self.default_ttl
            node.expiry_time = expiry_time
            self._move_to_head(node)
        else:
            # Create new node
            node = CacheNode(key, value, expiry_time, expiry_time)
            self.cache[key] = node
            self._add_to_head(node)
            self.size_count += 1

            # Evict if over capacity
            if self.size_count > self.capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """Delete a key from the cache if it exists.

        Args:
            key: The key to delete

        Returns:
            True if the key existed and was deleted, False otherwise
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size_count -= 1
        return True

    def size(self) -> int:
        """Return the number of non-expired items in the cache.

        Returns:
            Count of non-expired items (lazy cleanup)
        """
        self._cleanup_expired()
        return self.size_count
