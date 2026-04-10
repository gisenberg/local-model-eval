import time
from typing import Any, Optional


class Node:
    """Doubly-linked list node for cache entries."""
    def __init__(self, key: str, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU cache with time-based expiration.
    
    Maintains O(1) average time complexity for get/put/delete operations
    using a doubly-linked list and hash map.
    """
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
            default_ttl: Default time-to-live in seconds for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: dict[str, Node] = {}
        
        # Dummy head and tail nodes for the doubly-linked list
        self._head = Node("", None, 0.0)
        self._tail = Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _get_current_time(self) -> float:
        """Return current monotonic time."""
        return time.monotonic()
    
    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired."""
        return node.expiry_time <= self._get_current_time()
    
    def _move_to_head(self, node: Node) -> None:
        """Move a node to the head (most recently used position)."""
        # Remove from current position
        node.prev.next = node.next
        node.next.prev = node.prev
        
        # Insert at head
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the list and hash map."""
        node.prev.next = node.next
        node.next.prev = node.prev
        del self._cache[node.key]
    
    def _add_to_head(self, node: Node) -> None:
        """Add a node to the head of the list."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
        self._cache[node.key] = node
    
    def get(self, key: str) -> Optional[Any]:
        """
        Return value if exists and not expired, else None.
        
        Accessing a key makes it most-recently-used.
        Expired items are removed lazily on access.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value if present and not expired, None otherwise
        """
        if key not in self._cache:
            return None
        
        node = self._cache[key]
        
        # Check expiration and remove if expired
        if self._is_expired(node):
            self._remove_node(node)
            return None
        
        # Move to head (most recently used)
        self._move_to_head(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If at capacity, evicts the least-recently-used non-expired item.
        If all items are expired, clears them all first.
        Custom TTL overrides default TTL.
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Optional custom time-to-live in seconds (uses default if None)
        """
        if key in self._cache:
            # Update existing entry
            node = self._cache[key]
            node.value = value
            node.expiry_time = self._get_current_time() + (ttl or self.default_ttl)
            self._move_to_head(node)
        else:
            # Need to insert new item
            # If at capacity, evict LRU non-expired item(s)
            if len(self._cache) == self.capacity:
                # Remove expired items from LRU end first
                while self._tail.prev != self._head:
                    lru_node = self._tail.prev
                    if self._is_expired(lru_node):
                        self._remove_node(lru_node)
                    else:
                        break
                
                # If still at capacity, evict the LRU non-expired item
                if len(self._cache) == self.capacity:
                    lru_node = self._tail.prev
                    self._remove_node(lru_node)
            
            # Insert new node
            expiry_time = self._get_current_time() + (ttl or self.default_ttl)
            new_node = Node(key, value, expiry_time)
            self._add_to_head(new_node)
    
    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key existed (even if expired), False otherwise
        """
        if key not in self._cache:
            return False
        
        node = self._cache[key]
        self._remove_node(node)
        return True
    
    def size(self) -> int:
        """
        Return count of non-expired items.
        
        Note: This operation is O(n) as it must check expiration status
        of all items. Expired items are removed lazily on access.
        
        Returns:
            Number of non-expired items in the cache
        """
        count = 0
        for node in self._cache.values():
            if not self._is_expired(node):
                count += 1
        return count


# Test suite
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_time():
    """Fixture to mock time.monotonic for deterministic testing."""
    with patch('time.monotonic') as mock_monotonic:
        mock_monotonic.return_value = 0.0
        yield mock_monotonic


def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_capacity_eviction(mock_time):
    """Test LRU eviction when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" (LRU)
    
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_ttl_expiry(mock_time):
    """Test that items expire after TTL."""
    cache = TTLCache(capacity=2, default_ttl=1.0)
    
    cache.put("a", 1)
    mock_time.return_value = 2.0  # Advance time by 2.0 seconds
    
    assert cache.get("a") is None


def test_custom_per_key_ttl(mock_time):
    """Test custom TTL per key."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1, ttl=1.0)
    cache.put("b", 2, ttl=5.0)
    
    mock_time.return_value = 2.0
    
    assert cache.get("a") is None  # Expired (1.0 < 2.0)
    assert cache.get("b") == 2    # Not expired (5.0 > 2.0)


def test_delete(mock_time):
    """Test delete operation."""
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False  # Already deleted


def test_size_mixed_expired_valid(mock_time):
    """Test size() with mixed expired and valid items."""
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("a", 1, ttl=1.0)
    cache.put("b", 2, ttl=10.0)
    cache.put("c", 3, ttl=1.0)
    
    mock_time.return_value = 2.0  # "a" and "c" expired, "b" valid
    
    assert cache.size() == 1