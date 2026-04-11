import time
from typing import Any, Optional, Dict
from collections import deque

class _Node:
    """Doubly linked list node for the cache."""
    __slots__ = ['key', 'value', 'ttl', 'expiry', 'prev', 'next']

    def __init__(self, key: str, value: Any, ttl: float, current_time: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expiry = current_time + ttl
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    An LRU (Least Recently Used) cache with time-based expiration.
    
    Uses a doubly-linked list for O(1) access and reordering, and a hash map
    for O(1) key lookups. Expired items are cleaned up lazily during access
    or when the cache is full and needs eviction.
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
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[str, _Node] = {}
        
        # Dummy head and tail nodes to simplify edge cases in linked list operations
        self._head = _Node("", None, 0, 0)
        self._tail = _Node("", None, 0, 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        """Add a node right after the head (most recently used position)."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _get_current_time(self) -> float:
        """Wrapper for time.monotonic to facilitate mocking in tests."""
        return time.monotonic()

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node is expired based on current time."""
        return self._get_current_time() > node.expiry

    def _evict_lru_non_expired(self) -> Optional[str]:
        """
        Evict the least recently used non-expired item.
        Returns the key of the evicted item, or None if no valid items exist.
        """
        current = self._tail.prev
        while current != self._head:
            if not self._is_expired(current):
                # Found a valid LRU item
                key_to_remove = current.key
                self._remove_node(current)
                del self._map[key_to_remove]
                return key_to_remove
            # If expired, remove it immediately (lazy cleanup during eviction scan)
            key_to_remove = current.key
            prev_node = current.prev
            self._remove_node(current)
            del self._map[key_to_remove]
            current = prev_node
        
        return None

    def _cleanup_all_expired(self) -> None:
        """Remove all expired items from the cache."""
        current = self._head.next
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with the key.
        
        If the key exists and is not expired, it is moved to the front (most recently used)
        and the value is returned. If the key is expired or does not exist, returns None.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and valid, otherwise None.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        
        if self._is_expired(node):
            # Lazy cleanup: remove expired item on access
            self._remove_node(node)
            del self._map[key]
            return None

        # Move to front (MRU)
        self._remove_node(node)
        self._add_to_front(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If the key exists, it is updated and moved to the front.
        If the cache is at capacity, the least recently used non-expired item is evicted.
        If all items are expired, they are cleared first before insertion.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional custom time-to-live in seconds. If None, uses default_ttl.
        """
        current_time = self._get_current_time()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        # If key exists, update it
        if key in self._map:
            node = self._map[key]
            node.value = value
            node.ttl = effective_ttl
            node.expiry = current_time + effective_ttl
            # Move to front
            self._remove_node(node)
            self._add_to_front(node)
            return

        # Check if we need to evict
        # First, ensure we aren't just full of expired items
        if len(self._map) >= self.capacity:
            # Try to find a non-expired LRU to evict
            evicted = self._evict_lru_non_expired()
            
            # If we couldn't evict a non-expired item, it means all items are expired
            # or the cache was effectively empty of valid items.
            # In this case, we clear all expired items to make room.
            if evicted is None:
                self._cleanup_all_expired()
        
        # Re-check capacity after cleanup (in case cleanup freed up space)
        # If still at capacity, we must have found a valid LRU to evict in the previous step
        # or the map is now smaller than capacity.
        # However, if the previous step evicted a valid item, len(self._map) is now < capacity.
        # If the previous step found no valid items to evict, we cleaned all expired, 
        # so len(self._map) is now 0 (or < capacity).
        
        # Create new node
        new_node = _Node(key, value, effective_ttl, current_time)
        self._map[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key in self._map:
            node = self._map[key]
            self._remove_node(node)
            del self._map[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items in the cache.
        
        Performs a lazy cleanup of expired items during the count.
        
        Returns:
            The number of valid items.
        """
        # We need to count valid items. Since we use lazy cleanup, 
        # we must iterate and remove expired ones to get an accurate count 
        # and maintain consistency for future operations.
        current = self._head.next
        count = 0
        
        while current != self._tail:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                del self._map[current.key]
            else:
                count += 1
            current = next_node
            
        return count
