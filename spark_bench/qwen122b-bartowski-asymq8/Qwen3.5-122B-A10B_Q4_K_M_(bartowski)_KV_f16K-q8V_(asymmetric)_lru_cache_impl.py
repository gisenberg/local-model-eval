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

    def _get_current_time(self) -> float:
        """Helper to get current monotonic time."""
        return time.monotonic()

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """Evict the least recently used non-expired item."""
        # The LRU item is right before the tail
        lru_node = self._tail.prev
        if lru_node == self._head:
            return  # Cache is empty

        # Remove from map and list
        del self._map[lru_node.key]
        self._remove_node(lru_node)

    def _cleanup_expired(self) -> None:
        """
        Remove all expired items from the cache.
        This is called when the cache is full to ensure we have space for new items.
        """
        current_time = self._get_current_time()
        # We iterate through the list. Since we need to remove items, 
        # we traverse carefully.
        node = self._head.next
        while node != self._tail:
            next_node = node.next
            if node.ttl_expiry <= current_time:
                del self._map[node.key]
                self._remove_node(node)
            node = next_node

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

        current_time = self._get_current_time()
        if node.ttl_expiry <= current_time:
            # Item expired, remove it
            del self._map[key]
            self._remove_node(node)
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
        current_time = self._get_current_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl

        # Check if key exists
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.ttl_expiry = expiry_time
            # Move to front
            self._remove_node(node)
            self._add_to_front(node)
            return

        # Key is new
        # If at capacity, we need to make space
        if len(self._map) >= self.capacity:
            # First, try to clean up expired items to free space
            self._cleanup_expired()
            
            # If still at capacity, evict the LRU item
            if len(self._map) >= self.capacity:
                self._evict_lru()

        # Create new node and add
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
            del self._map[key]
            self._remove_node(node)
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup: expired items are removed during this operation
        to ensure the returned size is accurate.
        """
        self._cleanup_expired()
        return len(self._map)
