import time
from typing import Any, Optional

class Node:
    """Node for doubly-linked list."""
    __slots__ = ('key', 'value', 'expiration_time', 'prev', 'next')
    
    def __init__(self, key: str, value: Any, expiration_time: float):
        self.key = key
        self.value = value
        self.expiration_time = expiration_time
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with TTL (Time To Live) expiration.
    
    Supports O(1) get, put, delete operations with automatic eviction of
    least recently used items when at capacity, and expiration of items
    after a specified time.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, Node] = {}
        
        # Dummy head and tail for doubly-linked list
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self._size = 0  # Number of non-expired items
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node: Node) -> None:
        """Add node to head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_head(self, node: Node) -> None:
        """Move node to head (mark as most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _is_expired(self, node: Node) -> bool:
        """Check if node has expired."""
        return time.monotonic() > node.expiration_time
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key if exists and not expired.
        
        Accessing a key makes it most-recently-used.
        Returns None if key doesn't exist or is expired.
        
        Args:
            key: The key to look up.
            
        Returns:
            The value if found and not expired, None otherwise.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None
        
        self._move_to_head(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        Custom ttl overrides default_ttl.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom time-to-live in seconds.
        """
        current_time = time.monotonic()
        expiration_time = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            node.expiration_time = expiration_time
            self._move_to_head(node)
            return
        
        # New key - check if we need to evict
        if self._size >= self.capacity:
            # Evict LRU non-expired items until we have space
            while self._size > 0:
                lru_node = self.tail.prev
                if lru_node == self.head:
                    break
                
                if self._is_expired(lru_node):
                    # Remove expired item
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]
                    self._size -= 1
                    continue
                
                # Found non-expired LRU, evict it
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
                self._size -= 1
                break
        
        # Insert new node
        new_node = Node(key, value, expiration_time)
        self.cache[key] = new_node
        self._add_to_head(new_node)
        self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if key existed and was deleted, False otherwise.
        """
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self._size -= 1
        return True
    
    def size(self) -> int:
        """
        Return count of non-expired items.
        
        Note: Expired items are removed on access (lazy cleanup).
        This returns the current count of items that have not been
        found to be expired during access.
        
        Returns:
            Number of non-expired items in the cache.
        """
        return self._size


# Pytest tests
import pytest
from unittest.mock import patch

def test_basic_get_put():
    """Test basic get and put operations."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        cache.put("b", 2)
        assert cache.get("b") == 2
        
        assert cache.get("c") is None

def test_capacity_eviction():
    """Test LRU eviction when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a" (LRU)
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

def test_ttl_expiry():
    """Test TTL expiration."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        # Time 0: put
        mock_time.return_value = 0.0
        cache.put("a", 1)
        
        # Time 5: get (not expired)
        mock_time.return_value = 5.0
        assert cache.get("a") == 1
        
        # Time 15: get (expired)
        mock_time.return_value = 15.0
        assert cache.get("a") is None

def test_custom_ttl():
    """Test custom per-key TTL."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        mock_time.return_value = 0.0
        cache.put("a", 1, ttl=5.0)
        cache.put("b", 2, ttl=20.0)
        
        mock_time.return_value = 6.0
        assert cache.get("a") is None  # expired
        assert cache.get("b") == 2    # valid

def test_delete():
    """Test delete operation."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False
        assert cache.get("a") is None

def test_size_mixed():
    """Test size with mixed expired and valid items."""
    with patch('time.monotonic') as mock_time:
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        mock_time.return_value = 0.0
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.size() == 3
        
        mock_time.return_value = 15.0
        # Access expired item "a", it should be removed
        assert cache.get("a") is None
        assert cache.size() == 2
        
        assert cache.get("b") == 2
        assert cache.size() == 2
        
        assert cache.get("c") == 3
        assert cache.size() == 2