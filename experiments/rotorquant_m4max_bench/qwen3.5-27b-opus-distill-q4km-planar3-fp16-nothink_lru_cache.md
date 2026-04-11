# Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/f16) — lru_cache

**Hypothesis:** H5: within ±5% of f16 baseline (13 tok/s); sanity check, low signal

## Output

Here is the complete implementation of the LRU Cache with TTL, followed by the requested pytest tests.

### Implementation

```python
import time
from typing import Any, Optional, Dict, List

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
    An LRU (Least Recently Used) Cache with Time-To-Live (TTL) support.
    
    Uses a hash map for O(1) access and a doubly-linked list for O(1) 
    reordering and eviction.
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
        if default_ttl < 0:
            raise ValueError("TTL cannot be negative")

        self.capacity = capacity
        self.default_ttl = default_ttl
        
        # Hash map for O(1) access
        self.cache_map: Dict[Any, Node] = {}
        
        # Doubly linked list sentinels
        self.head: Node = Node(None, None, 0.0)  # Most recently used
        self.tail: Node = Node(None, None, 0.0)  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: Node) -> None:
        """Insert a node right after the head (MRU position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the linked list."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _move_to_head(self, node: Node) -> None:
        """Move an existing node to the head (MRU position)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _evict_lru(self) -> None:
        """Remove the least recently used item (node before tail)."""
        if self.tail.prev != self.head:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache_map[lru_node.key]

    def _is_expired(self, node: Node) -> bool:
        """Check if a node has expired based on monotonic time."""
        return time.monotonic() >= node.expire_at

    def _cleanup_expired(self) -> None:
        """
        Lazy cleanup: Removes expired nodes from the linked list and map.
        Note: In a strict O(1) requirement, we usually only clean up 
        the specific node being accessed or evicted. Iterating the whole 
        list is O(N). However, 'lazy cleanup on access' implies checking 
        validity when interacting with the cache.
        """
        # We iterate carefully. If we find an expired node, we remove it.
        # To maintain O(1) average, we typically only clean up the node 
        # currently being accessed or evicted. 
        # However, to ensure the cache doesn't fill with garbage, 
        # we might need a background thread or aggressive eviction.
        # For this implementation, we will clean up the specific node 
        # during get/put/delete operations to satisfy the prompt's 
        # "lazy cleanup on access" requirement without O(N) scans.
        pass

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        Returns the value if the key exists and is not expired.
        Moves the accessed item to the front (MRU).
        Returns None if the key is missing or expired.
        """
        if key not in self.cache_map:
            return None

        node = self.cache_map[key]

        # Check expiration
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache_map[key]
            return None

        # Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.
        
        Args:
            key: The key to store.
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
        
        Returns True if the item was present and removed, False otherwise.
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

### Pytest Tests

```python
import pytest
import time
from unittest.mock import patch
from typing import Any

# Import the class defined above
# Assuming the class is in a file named 'lru_cache.py' or defined in the same scope
# For this snippet, we assume it's available as TTLCache

class TestTTLCache:
    
    @patch('time.monotonic')
    def test_put_and_get(self, mock_monotonic):
        """Test basic put and get functionality."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put('key1', 'value1')
        
        assert cache.get('key1') == 'value1'
        assert cache.size() == 1

    @patch('time.monotonic')
    def test_lru_eviction(self, mock_monotonic):
        """Test that the Least Recently Used item is evicted when capacity is reached."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        
        # Access key1 to make it MRU
        cache.get('key1')
        
        # Add key3, should evict key2 (LRU)
        cache.put('key3', 'value3')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') is None  # Evicted
        assert cache.get('key3') == 'value3'
        assert cache.size() == 2

    @patch('time.monotonic')
    def test_ttl_expiration(self, mock_monotonic):
        """Test that items expire after their TTL."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put('key1', 'value1')
        
        # Advance time past TTL
        mock_monotonic.return_value = 115.0
        
        assert cache.get('key1') is None
        assert cache.size() == 0

    @patch('time.monotonic')
    def test_custom_ttl(self, mock_monotonic):
        """Test setting a custom TTL for a specific item."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        # Put with custom TTL of 5 seconds
        cache.put('key1', 'value1', ttl=5.0)
        
        # Advance time to 105.0 (past custom TTL, but within default TTL)
        mock_monotonic.return_value = 105.0
        
        assert cache.get('key1') is None

    @patch('time.monotonic')
    def test_delete(self, mock_monotonic):
        """Test the delete method."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put('key1', 'value1')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None
        assert cache.delete('non_existent') is False

    @patch('time.monotonic')
    def test_update_existing_key(self, mock_monotonic):
        """Test updating an existing key updates value and refreshes TTL."""
        mock_monotonic.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put('key1', 'value1')
        
        # Advance time to 109.0 (almost expired)
        mock_monotonic.return_value = 109.0
        
        # Update key
        cache.put('key1', 'value1_updated')
        
        # Should still be valid because TTL was refreshed
        assert cache.get('key1') == 'value1_updated'
        
        # Advance time to 118.0 (past original TTL, but within refreshed TTL)
        mock_monotonic.return_value = 118.0
        
        assert cache.get('key1') == 'value1_updated'
```
