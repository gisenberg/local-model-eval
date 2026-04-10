    class Node:
        def __init__(self, key, value, expiry):
            self.key = key
            self.value = value
            self.expiry = expiry
            self.prev = None
            self.next = None

    class TTLCache:
        def __init__(self, capacity, default_ttl):
            self.capacity = capacity
            self.default_ttl = default_ttl
            self.cache = {} # key -> Node
            self.head = None # MRU
            self.tail = None # LRU
            self._size = 0

        def _add_to_head(self, node):
            ...
        def _remove_node(self, node):
            ...
        def _evict_lru(self):
            ...
        def get(self, key):
            if key not in self.cache: return None
            node = self.cache[key]
            now = time.monotonic()
            if now > node.expiry:
                # Expired
                self._remove_node(node)
                del self.cache[key]
                self._size -= 1
                return None
            self._move_to_head(node)
            return node.value
        def put(self, key, value, ttl=None):
            expiry = time.monotonic() + (ttl or self.default_ttl)
            if key in self.cache:
                # Update existing
                node = self.cache[key]
                node.value = value
                node.expiry = expiry
                self._move_to_head(node)
            else:
                new_node = Node(key, value, expiry)
                self._add_to_head(new_node)
                self.cache[key] = new_node
                self._size += 1
                if self._size > self.capacity:
                    self._evict_lru()
        def delete(self, key):
            ...
        def size(self):
            return self._size
    

import time
from typing import Any, Optional

class Node:
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        ...


import time
from typing import Any, Optional

class Node:
    """Doubly-linked list node to store cache entries."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a doubly-linked list and hash map for O(1) operations.
    Cleanup of expired items is lazy (happens on access).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items in the cache.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, Node] = {}  # Maps key -> Node
        self.head: Optional[Node] = None  # Most Recently Used (MRU)
        self.tail: Optional[Node] = None  # Least Recently Used (LRU)
        self._size: int = 0

    def _add_to_head(self, node: Node) -> None:
        """Insert a new node at the head of the list."""
        if self.head is not None:
            self.head.prev = node
        node.next = self.head
        node.prev = None
        self.head = node

        if self.tail is None:
            self.tail = node

    def _remove_node(self, node: Node) -> None:
        """Remove a specific node from the list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _evict_lru(self) -> None:
        """Remove the LRU item from the cache to make room."""
        if self.tail is not None:
            node = self.tail
            del self.cache[node.key]
            self._remove_node(node)
            self._size -= 1

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key. If expired or missing, returns None and removes from cache.

        Args:
            key: The key to retrieve.

        Returns:
            The value if found and not expired, else None.
        """
        now = time.monotonic()
        
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        # Check expiration (Lazy cleanup)
        if now > node.expiry:
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None
        
        # Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value for key with optional TTL.

        Args:
            key: The key to set.
            value: The value to store.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing entry
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
        else:
            # Add new entry
            new_node = Node(key, value, expiry)
            self._add_to_head(new_node)
            self.cache[key] = new_node
            self._size += 1
            
            # Evict if capacity exceeded
            if self._size > self.capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if key did not exist.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return True
        return False

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            The count of items currently stored (including potentially expired ones 
            that haven't been accessed yet).
        """
        return self._size
