"""
LRU Cache with Time-To-Live (TTL) implementation.
Uses a Doubly Linked List + Hash Map. No OrderedDictionary.
Lazy cleanup mechanism on access.
"""
from typing import Optional, Dict, Any
from time import monotonic


class _Node:
    """Doubly linked list node."""
    __slots__ = ('key', 'value', 'expire_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expire_at: float) -> None:
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    Thread-safe (logic-wise) LRU Cache with TTL support.
    
    Attributes:
        capacity (int): Maximum number of items the cache can hold.
        default_ttl (float): Default time-to-live in seconds for inserted items.
        
    Complexity:
        get: O(1) amortized.
        put: O(1) amortized.
        delete: O(1).
        size: O(1).
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items. Must be > 0.
            default_ttl: Default expiration duration in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
            
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._cache: Dict[str, _Node] = {}
        
        # Dummy Head and Tail for doubly linked list
        self._head: _Node = _Node("", "", 0)
        self._tail: _Node = _Node("", "", 0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after the head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _get_current_time(self) -> float:
        """Helper to get current monotonic time for easier testing mocking."""
        return monotonic()

    def _is_expired(self, node: _Node, current_time: float) -> bool:
        """Check if a node has passed its expiration time."""
        return current_time >= node.expire_at

    def _evict_lru(self) -> None:
        """Removes the least recently used node (before tail)."""
        if self._head.next is self._tail:
            return  # Empty
        lru_node = self._tail.prev
        self._remove_node(lru_node)
        del self._cache[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value by key. Returns None if key not found or expired.
        Moves found item to Most Recently Used (MRU) position.
        Performs lazy cleanup of expired nodes.
        """
        current_time = self._get_current_time()
        node = self._cache.get(key)

        if node is None:
            return None

        # Check TTL
        if self._is_expired(node, current_time):
            # Lazy deletion of expired entry
            self._remove_node(node)
            del self._cache[key]
            return None

        # Refresh position to MRU
        self._remove_node(node)
        self._add_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a value in the cache.
        
        Args:
            key: Unique identifier.
            value: Value to store.
            ttl: Optional override for expiration time in seconds. 
                 If None, uses default_ttl.
        """
        current_time = self._get_current_time()
        expiration = current_time + (ttl if ttl is not None else self._default_ttl)
        
        if key in self._cache:
            # Update existing
            node = self._cache[key]
            node.value = value
            node.expire_at = expiration
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # New insertion
            if len(self._cache) >= self._capacity:
                self._evict_lru()
            
            new_node = _Node(key, value, expiration)
            self._cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: str) -> None:
        """
        Remove a key from the cache immediately.
        """
        node = self._cache.pop(key, None)
        if node:
            self._remove_node(node)

    def size(self) -> int:
        """
        Returns the number of items currently tracked in the cache.
        Note: Includes potentially expired items that have not been 
        reached via a 'get' operation yet due to lazy cleanup.
        """
        return len(self._cache)