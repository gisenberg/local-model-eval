# Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3 @ 256K) — lru_cache

**Hypothesis:** H2 real unlock: f16 OOMs at 256K (projected 32 GB), planar3/planar3 fits (projected 27 GB). New context tier.

## Output

Here is the implementation of the `TTLCache` class followed by the pytest test suite.

### Implementation

```python
from typing import Any, Optional
import time

class Node:
    """Doubly linked list node to store key, value, and expiration time."""
    def __init__(self, key: Any, value: Any, expire_at: float):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    An LRU (Least Recently Used) cache with TTL (Time To Live) support.
    
    Uses a hash map for O(1) access and a doubly linked list for O(1) 
    reordering of items based on usage.
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
        
        # Hash map for O(1) lookup
        self.cache_map: dict[Any, Node] = {}
        
        # Doubly linked list head and tail sentinels
        self.head = Node(None, None, 0.0)
        self.tail = Node(None, None, 0.0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: Node) -> None:
        """Insert a node right after the head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_head(self, node: Node) -> None:
        """Move an existing node to the head (mark as recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove the least recently used item (node before tail)."""
        if self.tail.prev != self.head:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache_map[lru_node.key]

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on current monotonic time."""
        return time.monotonic() >= node.expire_at

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        If the item exists and is not expired, it is moved to the head 
        (most recently used) and returned.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value associated with the key, or None if not found or expired.
        """
        if key not in self.cache_map:
            return None

        node = self.cache_map[key]

        # Lazy cleanup: Check expiration
        if self._is_expired(node):
            self.delete(key)
            return None

        # Move to head (update access time)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.
        
        Args:
            key: The key to insert.
            value: The value to store.
            ttl: Optional specific TTL for this item. Uses default_ttl if None.
        """
        current_time = time.monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expire_at = current_time + effective_ttl

        if key in self.cache_map:
            # Update existing key
            node = self.cache_map[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
        else:
            # Insert new key
            if len(self.cache_map) >= self.capacity:
                self._evict_lru()
            
            new_node = Node(key, value, expire_at)
            self.cache_map[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Any) -> bool:
        """
        Remove an item from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key was present and deleted, False otherwise.
        """
        if key not in self.cache_map:
            return False
        
        node = self.cache_map[key]
        self._remove_node(node)
        del self.cache_map[key]
        return True

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache_map)
```

### Tests

```python
import pytest
from unittest.mock import patch
import time
from typing import Any, Optional

# Import the class defined above
# Assuming the class is in the same file or imported as: from my_module import TTLCache
# For this snippet, we assume it's available in the scope.

class TestTTLCache:
    
    def test_put_and_get(self):
        """Test basic put and get functionality."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
        assert cache.size() == 1

    def test_lru_eviction(self):
        """Test that the least recently used item is evicted when capacity is reached."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (LRU)
        cache.put("key3", "value3")
        
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"
        assert cache.size() == 2

    def test_ttl_expiration(self):
        """Test that items expire after their TTL."""
        # Mock time to control expiration
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=2, default_ttl=5.0)
            cache.put("key1", "value1")
            
            # Time is still 0, should be valid
            assert cache.get("key1") == "value1"
            
            # Advance time past TTL
            mock_time.return_value = 6.0
            
            # Should be expired
            assert cache.get("key1") is None
            assert cache.size() == 0

    def test_custom_ttl(self):
        """Test that custom TTL overrides default TTL."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=2, default_ttl=10.0)
            # Put with custom TTL of 2 seconds
            cache.put("key1", "value1", ttl=2.0)
            
            # Advance time to 3 seconds (past custom TTL, but within default)
            mock_time.return_value = 3.0
            
            assert cache.get("key1") is None

    def test_update_existing_key(self):
        """Test updating an existing key refreshes its position and TTL."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0.0
            
            cache = TTLCache(capacity=2, default_ttl=5.0)
            cache.put("key1", "value1")
            cache.put("key2", "value2")
            
            # Advance time to 4.0
            mock_time.return_value = 4.0
            
            # Update key1
            cache.put("key1", "updated_value")
            
            # Advance time to 6.0 (past original TTL of key2, but key1 was refreshed)
            mock_time.return_value = 6.0
            
            assert cache.get("key1") == "updated_value"
            assert cache.get("key2") is None

    def test_delete_key(self):
        """Test deleting a key."""
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False
        assert cache.size() == 0
```
