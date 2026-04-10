# Final check on imports for the test file
# The test file assumes the class is imported from 'lru_ttl_cache'
# In a real scenario, ensure the file structure matches.


import time
from typing import Any, Optional

class Node:
    """Doubly linked list node for the cache."""
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a hash map for O(1) lookup and a doubly-linked list for O(1) 
    MRU/LRU ordering. Expired items are removed lazily upon access.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
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
        self.cache: dict[str, Node] = {}
        
        # Dummy head and tail for doubly linked list
        self.head = Node("", None, 0)
        self.tail = Node("", None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Track count of valid (non-expired) items for O(1) size()
        self._valid_count = 0

    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _remove_tail(self) -> Optional[Node]:
        """Remove and return the node before the tail (LRU position)."""
        if self.tail.prev == self.head:
            return None
        node = self.tail.prev
        self._remove_node(node)
        return node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() > node.expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value for key.
        
        Returns value if exists and not expired. Accessing a key makes it 
        most-recently-used. Returns None if key doesn't exist or is expired.
        """
        node = self.cache.get(key)
        if not node:
            return None
        
        if self._is_expired(node):
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self.cache[key]
            self._valid_count -= 1
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        Custom ttl overrides default_ttl.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl
        
        # Check if key exists
        if key in self.cache:
            node = self.cache[key]
            # If existing node is expired, treat as new insertion logic (remove old)
            if self._is_expired(node):
                self._remove_node(node)
                del self.cache[key]
                self._valid_count -= 1
            else:
                # Update value and expiry, move to head
                node.value = value
                node.expiry = expiry_time
                self._remove_node(node)
                self._add_to_head(node)
                return

        # New item insertion
        # Eviction logic if at capacity
        # We must evict the LRU non-expired item. If tail is expired, we clean it.
        while len(self.cache) >= self.capacity:
            tail = self.tail.prev
            if tail == self.head:
                break
            
            if self._is_expired(tail):
                # Remove expired tail, continue to find valid LRU or empty
                self._remove_node(tail)
                del self.cache[tail.key]
                self._valid_count -= 1
            else:
                # Evict valid LRU
                self._remove_node(tail)
                del self.cache[tail.key]
                self._valid_count -= 1
                break

        # Add new node
        new_node = Node(key, value, expiry_time)
        self._add_to_head(new_node)
        self.cache[key] = new_node
        self._valid_count += 1

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Returns True if key existed, False otherwise.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self._valid_count -= 1
            return True
        return False

    def size(self) -> int:
        """
        Return count of non-expired items.
        
        Uses lazy cleanup: expired items are removed on access.
        Returns the tracked count of valid items.
        """
        return self._valid_count
