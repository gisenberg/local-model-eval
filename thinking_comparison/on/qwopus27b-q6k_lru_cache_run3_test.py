import time
from typing import Any, Optional


class Node:
    """Doubly-linked list node for cache entries."""
    def __init__(self, key: str, value: Any, timestamp: float, ttl: float):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """LRU cache with time-based expiration using doubly-linked list + hash map."""
    
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize cache with given capacity and default TTL.
        
        Args:
            capacity: Maximum number of items to store.
            default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, Node] = {}
        
        # Dummy head and tail for doubly-linked list (MRU at head, LRU at tail)
        self.head = Node("", None, 0, 0)
        self.tail = Node("", None, 0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _is_expired(self, node: Node) -> bool:
        """Check if node is expired based on current time."""
        return time.monotonic() - node.timestamp > node.ttl
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from doubly-linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_head(self, node: Node) -> None:
        """Add node to head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_head(self, node: Node) -> None:
        """Move node to head (mark as most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _evict_lru(self) -> None:
        """Evict least recently used non-expired item, or clear all expired."""
        # Remove expired items from tail until we find non-expired or list is empty
        while self.tail.prev != self.head:
            lru_node = self.tail.prev
            if self._is_expired(lru_node):
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
            else:
                # Found non-expired LRU, evict it
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
                return
        # If we get here, all items were expired and removed
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key. Returns None if key not found or expired.
        Accessing a key makes it most-recently-used.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value if found and not expired, else None.
        """
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        if self._is_expired(node):
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self.cache[key]
            return None
        
        # Move to head (MRU)
        self._move_to_head(node)
        return node.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.
        
        Args:
            key: The key to insert/update.
            value: The value to store.
            ttl: Optional custom TTL in seconds. If None, uses default_ttl.
        """
        actual_ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.monotonic()
        
        if key in self.cache:
            node = self.cache[key]
            if self._is_expired(node):
                # Treat as new insertion: remove expired node first
                self._remove_node(node)
                del self.cache[key]
            else:
                # Update existing non-expired
                node.value = value
                node.timestamp = current_time
                node.ttl = actual_ttl
                self._move_to_head(node)
                return
        
        # Insert new (either new key or replaced expired)
        if len(self.cache) >= self.capacity:
            self._evict_lru()
        
        node = Node(key, value, current_time, actual_ttl)
        self.cache[key] = node
        self._add_to_head(node)
    
    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Args:
            key: The key to remove.
            
        Returns:
            True if key existed and was removed, False otherwise.
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
        Performs lazy cleanup of expired items.
        
        Returns:
            Number of non-expired items in cache.
        """
        current_time = time.monotonic()
        # Iterate and remove expired items
        current = self.head.next
        while current != self.tail:
            next_node = current.next
            if current_time - current.timestamp > current.ttl:
                self._remove_node(current)
                del self.cache[current.key]
            current = next_node
        return len(self.cache)


# ============================================================================
# Pytest Tests
# ============================================================================

import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_basic_get_put(mock_monotonic):
    """Test basic get and put operations."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None


@patch('time.monotonic')
def test_capacity_eviction(mock_monotonic):
    """Test LRU eviction when capacity is reached."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")  # LRU
    cache.put("key2", "value2")  # MRU
    cache.put("key3", "value3")  # Should evict key1 (LRU)
    
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


@patch('time.monotonic')
def test_ttl_expiry(mock_monotonic):
    """Test that items expire after TTL."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    
    cache.put("key1", "value1")
    
    # Advance time beyond TTL
    mock_monotonic.return_value = 6.0
    assert cache.get("key1") is None


@patch('time.monotonic')
def test_custom_per_key_ttl(mock_monotonic):
    """Test custom TTL per key."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=5.0)
    cache.put("key2", "value2", ttl=15.0)
    
    # Advance time to 6.0: key1 expired, key2 valid
    mock_monotonic.return_value = 6.0
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"


@patch('time.monotonic')
def test_delete(mock_monotonic):
    """Test delete operation."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("key1") is False


@patch('time.monotonic')
def test_size_with_mixed_expired_valid(mock_monotonic):
    """Test size() with mixed expired and valid items."""
    mock_monotonic.return_value = 0.0
    cache = TTLCache(capacity=3, default_ttl=10.0)
    
    cache.put("key1", "value1", ttl=5.0)
    cache.put("key2", "value2", ttl=15.0)
    cache.put("key3", "value3", ttl=5.0)
    
    # Advance time to 6.0: key1 and key3 expired, key2 valid
    mock_monotonic.return_value = 6.0
    assert cache.size() == 1