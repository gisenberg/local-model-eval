import time
from typing import Optional, Any, Dict

class TTLCache:
    """
    A Least Recently Used (LRU) cache with Time-To-Live (TTL) expiration.
    Uses a doubly-linked list for O(1) ordering and a hash map for O(1) lookups.
    """

    class _Node:
        """Internal doubly-linked list node."""
        def __init__(self, key: str, value: Any, expiry: float):
            self.key = key
            self.value = value
            self.expiry = expiry
            self.prev: Optional['TTLCache._Node'] = None
            self.next: Optional['TTLCache._Node'] = None

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache.

        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, _Node] = {}
        
        # Dummy head and tail nodes to simplify list operations
        self.head: _Node = self._Node("", None, 0)
        self.tail: _Node = self._Node("", None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: _Node) -> None:
        """Add a node right after the dummy head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.expiry

    def _evict_lru(self) -> None:
        """
        Evict the least-recently-used non-expired item.
        If the LRU item is expired, remove it and check the next one.
        If all items are expired, clear them all.
        """
        while self.tail.prev != self.head:
            node = self.tail.prev
            if self._is_expired(node):
                # Remove expired node from list and map
                self._remove_node(node)
                del self.cache[node.key]
            else:
                # Found a valid LRU item, evict it
                self._remove_node(node)
                del self.cache[node.key]
                return

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the key if it exists and is not expired.
        Accessing a key makes it most-recently-used.

        Args:
            key: The key to retrieve.

        Returns:
            The value if found and valid, otherwise None.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            # Lazy cleanup: remove expired item on access
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        If at capacity, evict the least-recently-used non-expired item.
        Custom ttl overrides default_ttl.

        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds.
        """
        current_time = time.monotonic()
        expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expiry = expiry_time
            self._remove_node(node)
            self._add_to_head(node)
            return

        # New item
        if len(self.cache) >= self.capacity:
            self._evict_lru()

        new_node = self._Node(key, value, expiry_time)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: The key to remove.

        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False

    def size(self) -> int:
        """
        Return the count of non-expired items.
        Expired items are removed on access (lazy cleanup).

        Returns:
            The number of items currently in the cache.
        """
        return len(self.cache)
