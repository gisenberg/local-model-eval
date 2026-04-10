import time
from typing import Any, Optional


class Node:
    """Doubly-linked list node for cache entries."""
    
    def __init__(self, key: str, value: Any, expiration: float):
        self.key = key
        self.value = value
        self.expiration = expiration
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU cache with TTL (Time To Live) expiration.
    
    Uses a doubly-linked list + hash map for O(1) operations.
    Expired items are removed lazily on access.
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
        
        # Sentinel nodes for doubly-linked list
        self.head = Node("", None, 0)  # LRU end
        self.tail = Node("", None, 0)  # MRU end
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self._size = 0  # Count of non-expired items
    
    def _get_time(self) -> float:
        """Return current monotonic time."""
        return time.monotonic()
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from the doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_tail(self, node: Node) -> None:
        """Add node to the tail (MRU position)."""
        prev_tail = self.tail.prev
        prev_tail.next = node
        node.prev = prev_tail
        node.next = self.tail
        self.tail.prev = node
    
    def _move_to_tail(self, node: Node) -> None:
        """Move existing node to the tail (MRU position)."""
        self._remove_node(node)
        self._add_to_tail(node)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        Accessing a key makes it most-recently-used.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        current_time = self._get_time()
        
        # Check if expired
        if current_time > node.expiration:
            # Remove expired item (lazy cleanup)
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None
        
        # Move to tail (MRU)
        self._move_to_tail(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        
        If at capacity, evict the least-recently-used non-expired item.
        If all items are expired, clear them all first.
        Custom ttl overrides default.
        """
        current_time = self._get_time()
        expiration = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            node.expiration = expiration
            self._move_to_tail(node)
        else:
            # New key - check if we need to evict
            if len(self.cache) == self.capacity:
                # Evict expired items first (lazy cleanup)
                while self.head.next != self.tail:
                    lru_node = self.head.next
                    if current_time > lru_node.expiration:
                        self._remove_node(lru_node)
                        del self.cache[lru_node.key]
                        self._size -= 1
                    else:
                        break
                
                # If still at capacity, evict LRU non-expired item
                if len(self.cache) == self.capacity:
                    lru_node = self.head.next
                    if lru_node != self.tail:
                        self._remove_node(lru_node)
                        del self.cache[lru_node.key]
                        self._size -= 1
            
            # Insert new node
            new_node = Node(key, value, expiration)
            self.cache[key] = new_node
            self._add_to_tail(new_node)
            self._size += 1
    
    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Returns:
            True if key existed and was removed, False otherwise.
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
        
        Note: Expired items are removed on access (lazy cleanup),
        so this count reflects items not yet lazily cleaned up.
        """
        return self._size


# Tests
import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_get_put(mock_monotonic):
    """Test basic get and put operations."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None
    assert cache.size() == 2


@patch('time.monotonic')
def test_capacity_eviction_lru_order(mock_monotonic):
    """Test that LRU items are evicted when at capacity."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU, then add 'c'
    cache.get("a")
    cache.put("c", 3)  # Should evict 'b' (LRU)
    
    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3
    assert cache.size() == 2


@patch('time.monotonic')
def test_ttl_expiry(mock_monotonic):
    """Test that expired items return None and are removed."""
    # Initial time 0
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)  # Expires at time 10
    
    # Advance time to 11 (expired)
    mock_monotonic.return_value = 11.0
    
    assert cache.get("a") is None
    assert cache.size() == 0  # Removed on access


@patch('time.monotonic')
def test_custom_per_key_ttl(mock_monotonic):
    """Test custom TTL per key overrides default."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1, ttl=5.0)  # Expires at 5
    cache.put("b", 2)           # Expires at 10
    
    # Time 6: 'a' expired, 'b' valid
    mock_monotonic.return_value = 6.0
    
    assert cache.get("a") is None  # Expired
    assert cache.get("b") == 2     # Valid
    assert cache.size() == 1


@patch('time.monotonic')
def test_delete(mock_monotonic):
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    
    assert cache.delete("a") is True
    assert cache.delete("a") is False  # Already deleted
    assert cache.get("a") is None
    assert cache.size() == 0


@patch('time.monotonic')
def test_size_mixed_expired_valid(mock_monotonic):
    """Test size with mixed expired and valid items (lazy cleanup)."""
    mock_monotonic.return_value = 0.0
    
    cache = TTLCache(capacity=3, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    
    assert cache.size() == 3
    
    # Advance time to 11 (all expired)
    mock_monotonic.return_value = 11.0
    
    # Size still 3 because expired items not accessed yet (lazy cleanup)
    assert cache.size() == 3
    
    # Access one expired item - it gets removed
    assert cache.get("a") is None
    assert cache.size() == 2
    
    # Access another expired item
    assert cache.get("b") is None
    assert cache.size() == 1