import time
from typing import Any, Optional

class _Node:
    """Doubly-linked list node for O(1) insertion and deletion."""
    __slots__ = ("key", "value", "expiry", "prev", "next")
    
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a hash map for O(1) lookups and a doubly-linked list for O(1) 
    LRU ordering. Expired items are lazily removed upon access.
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
            raise ValueError("TTL must be positive")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        
        # Hash map for O(1) access
        self._cache: dict[str, _Node] = {}
        
        # Doubly-linked list sentinels
        # head points to the LRU item (oldest)
        # tail points to the MRU item (newest)
        self._head = _Node("", None, 0.0)
        self._tail = _Node("", None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
        
        # Current count of items in the linked list (including potentially expired ones)
        self._size = 0

    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expiry

    def _add_to_tail(self, node: _Node) -> None:
        """Move node to the tail (MRU position)."""
        # Insert before tail
        prev_tail = self._tail.prev
        prev_tail.next = node
        node.prev = prev_tail
        node.next = self._tail
        self._tail.prev = node

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _evict_lru(self) -> None:
        """Evict the least recently used item (head.next) if it exists."""
        if self._head.next != self._tail:
            lru_node = self._head.next
            self._remove_node(lru_node)
            del self._cache[lru_node.key]
            self._size -= 1

    def _cleanup_expired(self) -> None:
        """
        Lazy cleanup: Remove all expired items from the cache.
        This is called when the cache is full or during size checks.
        """
        # We iterate through the list. Since we might modify the list,
        # we collect nodes to remove first or traverse carefully.
        # To maintain O(1) amortized behavior, we only clean up when necessary.
        # However, for strict correctness during eviction, we must ensure 
        # we don't count expired items against capacity.
        
        current = self._head.next
        while current != self._tail:
            if self._is_expired(current):
                next_node = current.next
                self._remove_node(current)
                del self._cache[current.key]
                self._size -= 1
                current = next_node
            else:
                current = current.next

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Returns the value if the key exists and is not expired.
        Accessing a key moves it to the MRU position.
        Returns None if key is missing or expired.
        """
        if key not in self._cache:
            return None
            
        node = self._cache[key]
        
        if self._is_expired(node):
            # Lazy removal of expired item
            self._remove_node(node)
            del self._cache[key]
            self._size -= 1
            return None
        
        # Move to MRU (tail)
        self._remove_node(node)
        self._add_to_tail(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a value in the cache.
        
        If the cache is at capacity, the LRU non-expired item is evicted.
        If all items are expired, they are cleared before insertion.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional custom TTL. If None, uses default_ttl.
        """
        current_time = time.monotonic()
        expiry = current_time + (ttl if ttl is not None else self.default_ttl)
        
        if key in self._cache:
            # Update existing key
            node = self._cache[key]
            node.value = value
            node.expiry = expiry
            # Move to MRU
            self._remove_node(node)
            self._add_to_tail(node)
            return

        # Check capacity
        # If we are at capacity, we need to make room.
        # Requirement: "If all items are expired, clear them all first."
        # We check if we are at capacity. If so, we try to evict.
        # If the LRU item is expired, we clean up.
        
        if self._size >= self.capacity:
            # Attempt to evict LRU
            # If the LRU item is expired, we should clean up the whole list 
            # to free up space efficiently as per requirements.
            if self._head.next != self._tail and self._is_expired(self._head.next):
                self._cleanup_expired()
            
            # If still full after cleanup (or if LRU wasn't expired), evict one
            if self._size >= self.capacity:
                self._evict_lru()

        # Insert new node
        new_node = _Node(key, value, expiry)
        self._cache[key] = new_node
        self._add_to_tail(new_node)
        self._size += 1

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Returns True if the key existed and was removed, False otherwise.
        """
        if key not in self._cache:
            return False
            
        node = self._cache[key]
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        
        Performs lazy cleanup of expired items before returning count.
        """
        self._cleanup_expired()
        return self._size

import pytest
from unittest.mock import patch
import time
from typing import Any

# Import the class we just implemented
  # Assuming this is run in a script context, 
                               # otherwise import from the module above.

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

def test_capacity_eviction_lru_order():
    """Test that LRU items are evicted when capacity is reached."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        # Access 'a' to make it MRU
        cache.get("a") 
        
        # Insert 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None

def test_ttl_expiry():
    """Test that items expire after their TTL."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 5.0]):
        # Initial time 0.0
        cache = TTLCache(capacity=2, default_ttl=2.0)
        cache.put("a", 1)
        
        # Time jumps to 5.0 (expired)
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test custom TTL overrides default."""
    with patch('time.monotonic', side_effect=[0.0, 0.0, 1.5, 1.5]):
        cache = TTLCache(capacity=2, default_ttl=1.0)
        
        # 'a' has default TTL (1.0s)
        cache.put("a", 1)
        # 'b' has custom TTL (2.0s)
        cache.put("b", 2, ttl=2.0)
        
        # Time is 1.5s
        # 'a' should be expired (1.5 > 1.0)
        assert cache.get("a") is None
        # 'b' should be valid (1.5 < 2.0)
        assert cache.get("b") == 2

def test_delete_operation():
    """Test deleting keys."""
    with patch('time.monotonic', return_value=0.0):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("a", 1)
        
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False

def test_size_with_mixed_expired_valid():
    """Test size() returns count of non-expired items only."""
    # Sequence of times: 0, 0, 0, 5, 5
    times = [0.0, 0.0, 0.0, 5.0, 5.0]
    with patch('time.monotonic', side_effect=times):
        cache = TTLCache(capacity=3, default_ttl=2.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # All valid initially
        assert cache.size() == 3
        
        # Time jumps to 5.0 (all expired)
        # size() should trigger cleanup and return 0
        assert cache.size() == 0
        
        # Put new item
        cache.put("d", 4)
        assert cache.size() == 1