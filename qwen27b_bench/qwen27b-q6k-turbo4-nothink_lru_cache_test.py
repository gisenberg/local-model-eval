import time
from typing import Any, Optional, Dict, List

class Node:
    """Doubly linked list node to store key, value, and expiration time."""
    __slots__ = ('key', 'value', 'ttl', 'expire_at', 'prev', 'next')

    def __init__(self, key: str, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a doubly-linked list and a hash map for O(1) operations.
    Expired items are lazily removed upon access or eviction.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items the cache can hold.
            default_ttl: Default time-to-live in seconds for items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be greater than 0")

        self.capacity = capacity
        self.default_ttl = default_ttl
        
        # Hash map: key -> Node
        self.cache_map: Dict[str, Node] = {}
        
        # Dummy head and tail for the doubly linked list
        # Order: [Head] <-> [Most Recent] <-> ... <-> [Least Recent] <-> [Tail]
        self.head = Node("", None, 0.0)
        self.tail = Node("", None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _get_current_time(self) -> float:
        """Helper to get current monotonic time."""
        return time.monotonic()

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        """Move an existing node to the head (mark as most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_expired(self) -> None:
        """
        Lazy cleanup: Remove all expired items from the cache.
        This is called before eviction or size calculation to ensure accuracy.
        """
        current_time = self._get_current_time()
        keys_to_remove = []
        
        for key, node in self.cache_map.items():
            if node.expire_at <= current_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            node = self.cache_map.pop(key)
            self._remove_node(node)

    def _evict_lru(self) -> None:
        """Evict the least recently used item (node just before tail)."""
        # Ensure we don't evict if empty
        if self.tail.prev == self.head:
            return
            
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        del self.cache_map[lru_node.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the given key.
        
        If the key exists and is not expired, it is moved to the head (MRU).
        Returns None if the key does not exist or has expired.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value associated with the key, or None.
        """
        if key not in self.cache_map:
            return None

        node = self.cache_map[key]
        current_time = self._get_current_time()

        # Check expiration
        if node.expire_at <= current_time:
            # Lazy cleanup: remove expired item
            self._remove_node(node)
            del self.cache_map[key]
            return None

        # Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        If the key exists, update value and TTL, and move to head.
        If the key is new, insert at head. If capacity is exceeded, 
        evict the LRU non-expired item. If all items are expired, 
        they are cleared first.
        
        Args:
            key: The key to insert.
            value: The value to store.
            ttl: Optional custom TTL. If None, uses default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl
            
        current_time = self._get_current_time()
        expire_at = current_time + ttl

        if key in self.cache_map:
            # Update existing
            node = self.cache_map[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
        else:
            # Insert new
            # First, ensure we have space by cleaning expired and evicting LRU if needed
            # We only evict if we are at capacity AFTER removing expired items
            # However, the requirement says: "If at capacity, evict... If all items expired, clear them first."
            # To handle "If all items are expired, clear them first" efficiently without O(N) scan 
            # on every put (which violates O(1) strictly if N is large), we rely on lazy cleanup 
            # triggered by the eviction logic or size check.
            
            # Check if we need to make space
            if len(self.cache_map) >= self.capacity:
                # Perform lazy cleanup of expired items first
                self._evict_expired()
                
                # If still at capacity (meaning non-expired items filled it), evict LRU
                if len(self.cache_map) >= self.capacity:
                    self._evict_lru()

            new_node = Node(key, value, expire_at)
            self.cache_map[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove the key from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key not in self.cache_map:
            return False
        
        node = self.cache_map.pop(key)
        self._remove_node(node)
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup of expired items before counting.
        
        Returns:
            Number of valid (non-expired) items.
        """
        self._evict_expired()
        return len(self.cache_map)

import pytest
from unittest.mock import patch


# Mock time.monotonic to control time deterministically
@patch('ttl_cache.time.monotonic')
def test_basic_get_put(mock_time):
    """Test basic insertion and retrieval."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1
    
    cache.put("b", 2)
    assert cache.get("b") == 2
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_capacity_eviction_lru_order(mock_time):
    """Test that LRU item is evicted when capacity is reached."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    # Access 'a' to make it MRU
    cache.get("a") 
    
    # Now 'b' is LRU. Insert 'c'.
    cache.put("c", 3)
    
    assert cache.get("c") == 3
    assert cache.get("a") == 1  # 'a' should still be there
    assert cache.get("b") is None  # 'b' should be evicted
    assert cache.size() == 2

@patch('ttl_cache.time.monotonic')
def test_ttl_expiry(mock_time):
    """Test that items expire after TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=5.0)
    
    cache.put("a", 1)
    assert cache.get("a") == 1
    
    # Advance time past TTL
    mock_time.return_value = 6.0
    
    assert cache.get("a") is None
    assert cache.size() == 0

@patch('ttl_cache.time.monotonic')
def test_custom_per_key_ttl(mock_time):
    """Test custom TTL overrides default TTL."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=100.0)
    
    # Item with custom short TTL
    cache.put("short", 1, ttl=2.0)
    # Item with default long TTL
    cache.put("long", 2)
    
    # Advance time to 3.0
    mock_time.return_value = 3.0
    
    assert cache.get("short") is None  # Expired
    assert cache.get("long") == 2      # Still valid
    assert cache.size() == 1

@patch('ttl_cache.time.monotonic')
def test_delete(mock_time):
    """Test deletion of existing and non-existing keys."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    
    assert cache.delete("nonexistent") is False

@patch('ttl_cache.time.monotonic')
def test_size_with_mixed_expired_valid(mock_time):
    """Test size() returns count of only non-expired items."""
    mock_time.return_value = 0.0
    cache = TTLCache(capacity=5, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2, ttl=2.0) # Expires quickly
    cache.put("c", 3)
    
    # Advance time to 3.0
    mock_time.return_value = 3.0
    
    # 'b' is expired, 'a' and 'c' are valid
    assert cache.size() == 2
    
    # Verify 'b' is actually gone from map after size check (lazy cleanup)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3