import time
from typing import Optional, Any, Dict


class Node:
    """Doubly-linked list node for LRU cache."""
    def __init__(self, key: str, value: Any, expiration_time: float):
        self.key = key
        self.value = value
        self.expiration_time = expiration_time
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Uses a doubly-linked list + hash map for O(1) average time operations.
    Expired items are removed lazily on access.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live in seconds for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Node] = {}
        
        # Dummy head (MRU end) and tail (LRU end)
        self.head = Node("", None, 0)
        self.tail = Node("", None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from doubly-linked list in O(1)."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_node(self, node: Node) -> None:
        """Add node to head (MRU position) in O(1)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _is_expired(self, node: Node) -> bool:
        """Check if node has expired."""
        return node.expiration_time <= time.monotonic()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key. Returns None if key doesn't exist or is expired.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to head (MRU)
        self._remove_node(node)
        self._add_node(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        Custom TTL overrides default.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom TTL in seconds (uses default if None)
        """
        current_time = time.monotonic()
        expiration_time = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Remove expired node
                self._remove_node(node)
                del self.cache[key]
            else:
                # Update existing node
                node.value = value
                node.expiration_time = expiration_time
                # Move to head (MRU)
                self._remove_node(node)
                self._add_node(node)
                return
        
        # Key not in cache or was expired
        # Check if at capacity
        if len(self.cache) >= self.capacity:
            # Traverse from tail (LRU) to find first non-expired item
            current = self.tail.prev
            while current != self.head:
                if self._is_expired(current):
                    # Remove expired item
                    self._remove_node(current)
                    del self.cache[current.key]
                    current = current.prev
                else:
                    # Found non-expired, evict it
                    self._remove_node(current)
                    del self.cache[current.key]
                    break
            else:
                # All items expired, clear everything
                self.cache.clear()
                self.head.next = self.tail
                self.tail.prev = self.head
        
        # Insert new node
        node = Node(key, value, expiration_time)
        self.cache[key] = node
        self._add_node(node)
    
    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key existed, False otherwise
        """
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        return True
    
    def size(self) -> int:
        """
        Return count of non-expired items.
        
        Performs lazy cleanup: removes expired items during scan.
        
        Returns:
            Number of non-expired items in cache
        """
        current_time = time.monotonic()
        count = 0
        keys_to_remove = []
        
        # Scan cache for expired items
        for key, node in self.cache.items():
            if node.expiration_time <= current_time:
                keys_to_remove.append(key)
            else:
                count += 1
        
        # Remove expired items
        for key in keys_to_remove:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
        
        return count


# Tests
import pytest
from unittest.mock import patch


def test_basic_get_put():
    """Test basic get and put operations."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None


def test_capacity_eviction():
    """Test LRU eviction when at capacity."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1 (LRU)
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"


def test_ttl_expiry():
    """Test time-based expiration."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 2.0]):
        cache = TTLCache(capacity=2, default_ttl=1.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Time advances to 2.0, item expired (TTL=1.0)
        assert cache.get("key1") is None


def test_custom_per_key_ttl():
    """Test custom TTL per key."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 2.0, 2.0]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        # Custom TTL of 1 second
        cache.put("key1", "value1", ttl=1.0)
        # Default TTL of 10 seconds
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        
        # Time advances to 2.0
        assert cache.get("key1") is None  # Expired (custom TTL=1.0)
        assert cache.get("key2") == "value2"  # Not expired (default TTL=10.0)


def test_delete():
    """Test delete operation."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False  # Already deleted


def test_size_with_mixed_expired_valid():
    """Test size() with mixed expired and valid items."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 2.0]):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # Insert items with different TTLs
        cache.put("key1", "value1", ttl=1.0)  # Expires at t=1
        cache.put("key2", "value2", ttl=10.0)  # Expires at t=10
        cache.put("key3", "value3", ttl=1.0)  # Expires at t=1
        
        assert cache.size() == 3  # All valid at t=0
        
        # Time advances to 2.0, key1 and key3 expired
        assert cache.size() == 1  # Only key2 remains