import time
from typing import Any, Optional, Dict, List
from collections import deque

class _Node:
    """Doubly linked list node for the cache."""
    __slots__ = 'key', 'value', 'ttl_expiry', 'prev', 'next'

    def __init__(self, key: str, value: Any, ttl_expiry: float):
        self.key = key
        self.value = value
        self.ttl_expiry = ttl_expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a doubly-linked list to maintain access order and a hash map for O(1) lookups.
    Expired items are removed lazily upon access or when the cache is full.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items without a custom TTL.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        
        # Dummy head and tail nodes to simplify edge cases
        self._head = _Node("", None, 0)
        self._tail = _Node("", None, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.ttl_expiry

    def _evict_expired(self) -> None:
        """
        Remove all expired items from the cache.
        This is called when the cache is full to make room, or during size calculation.
        """
        current = self._head.next
        while current and current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            current = next_node

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # First, ensure we don't try to evict an expired item if we can avoid it
        # by cleaning up expired ones first if the cache is full
        if len(self._map) == self.capacity:
            self._evict_expired()
            # If after cleaning expired items we still have items, evict the LRU
            # If the cache became empty after cleaning, we don't need to evict
            if len(self._map) == 0:
                return

        # The LRU item is right before the tail
        lru_node = self._tail.prev
        if lru_node and lru_node != self._head:
            self._remove_node(lru_node)
            del self._map[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Returns the value if the key exists and is not expired.
        Accessing a key moves it to the most recently used position.
        Returns None if the key does not exist or is expired.
        """
        node = self._map.get(key)
        if not node:
            return None

        if self._is_expired(node):
            # Lazy removal of expired item
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, update its value and TTL, and move to MRU.
        If the key is new and capacity is reached, evict the LRU non-expired item.
        If all items are expired, clear them first before inserting.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl

        if key in self._map:
            # Update existing
            node = self._map[key]
            node.value = value
            node.ttl_expiry = expiry_time
            self._remove_node(node)
            self._add_to_front(node)
            return

        # New item
        if len(self._map) >= self.capacity:
            # Check if we can just clear expired items to make space
            self._evict_expired()
            
            # If still at capacity, evict LRU
            if len(self._map) >= self.capacity:
                self._evict_lru()

        new_node = _Node(key, value, expiry_time)
        self._map[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed and was removed, False otherwise.
        """
        if key in self._map:
            node = self._map[key]
            self._remove_node(node)
            del self._map[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed during this operation.
        """
        self._evict_expired()
        return len(self._map)
