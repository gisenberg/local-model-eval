# Qwen3.5-9B (L40S vLLM) — LRU Cache with TTL — Run 2

## Output

Here is the complete implementation of the `TTLCache` class followed by the pytest test suite.

### Implementation: `ttl_cache.py`

```python
import time
import threading
from typing import Any, Dict, List, Optional

class Node:
    def __init__(self, key: str, value: Any, expiry_time: float):
        self.key = key
        self.value = value
        self.expiry_time = expiry_time
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the LRU cache with a fixed capacity and default TTL.
        
        :param capacity: Maximum number of non-expired items the cache can hold.
        :param default_ttl: Default time-to-live in seconds for new items.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be greater than 0.")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.current_time = 0.0  # Will be updated by time.monotonic() on access
        
        # Hash map: key -> Node
        self.cache_map: Dict[str, Node] = {}
        
        # Doubly linked list pointers
        self.head: Optional[Node] = None  # Most Recently Used
        self.tail: Optional[Node] = None  # Least Recently Used
        
        # Lock for thread safety
        self._lock = threading.Lock()

    def _get_current_time(self) -> float:
        return time.monotonic()

    def _update_time(self):
        """Synchronizes internal clock with real time."""
        self.current_time = self._get_current_time()

    def _is_expired(self, node: Node) -> bool:
        return self.current_time >= node.expiry_time

    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev:
            node.prev.next = node.next
        else:
            # node is head
            self.head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            # node is tail
            self.tail = node.prev
            
        node.prev = None
        node.next = None

    def _add_to_head(self, node: Node) -> None:
        """Add a node right after the head (making it MRU)."""
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            # If head was tail, update tail reference if list was single node
            if self.tail == self.head.prev: 
                # This logic is simplified; standard approach:
                # If list was empty, head/tail set above.
                # If list not empty, head.prev points to old head.
                pass 
            # Correct logic for single node insertion:
            if self.head == node:
                pass # Already handled
            else:
                # Re-evaluating standard doubly linked list insertion
                node.next = self.head
                if self.head:
                    self.head.prev = node
                self.head = node
                if self.tail is None:
                    self.tail = node

    def _evict_lru(self) -> Optional[Node]:
        """Remove the least recently used non-expired item. Returns the node if evicted, None if all expired."""
        # If list is empty or only expired items remain, we might need to clear
        # However, the requirement says: "If all items are expired, clear them all first."
        # We handle this during put. Here we just try to evict the tail.
        
        if self.tail is None:
            return None
            
        # Check if tail is expired
        if self._is_expired(self.tail):
            return None # Should be handled by caller to clear expired
        
        return self.tail

    def _clear_expired(self) -> None:
        """Remove all expired items from the cache."""
        current = self.head
        while current:
            next_node = current.next
            if self._is_expired(current):
                self._remove_node(current)
                if current.key in self.cache_map:
                    del self.cache_map[current.key]
            current = next_node

    def get(self, key: str) -> Optional[Any]:
        """
        Get value by key. If exists and not expired, move to MRU.
        Returns None if key missing or expired.
        """
        self._update_time()
        
        with self._lock:
            if key not in self.cache_map:
                return None
            
            node = self.cache_map[key]
            
            if self._is_expired(node):
                # Remove expired node
                self._remove_node(node)
                del self.cache_map[key]
                return None
            
            # Move to head (MRU)
            self._remove_node(node)
            self._add_to_head(node)
            
            return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        Evicts LRU non-expired items if at capacity.
        Clears all expired items if they exist before insertion.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self._update_time()
        expiry_time = self.current_time + effective_ttl
        
        with self._lock:
            # Step 1: Clear all expired items first
            self._clear_expired()
            
            # Step 2: Check if key exists
            if key in self.cache_map:
                # Update existing
                node = self.cache_map[key]
                node.value = value
                node.expiry_time = expiry_time
                # Move to head
                self._remove_node(node)
                self._add_to_head(node)
            else:
                # Step 3: Check capacity and evict if needed
                # Note: Capacity check happens after clearing expired items
                if len(self.cache_map) >= self.capacity:
                    lru_node = self._evict_lru()
                    if lru_node:
                        del self.cache_map[lru_node.key]
                
                # Step 4: Add new node
                new_node = Node(key, value, expiry_time)
                self.cache_map[key] = new_node
                self._add_to_head(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove key from cache. Returns True if existed, False otherwise.
        """
        with self._lock:
            if key not in self.cache_map:
                return False
            
            node = self.cache_map[key]
            self._remove_node(node)
            del self.cache_map[key]
            return True

    def size(self) -> int:
        """
        Return count of non-expired items.
        Performs lazy cleanup: checks expiry on access, but size() 
        iterates to count valid items efficiently.
        """
        self._update_time()
        count = 0
        with self._lock:
            for node in self.cache_map.values():
                if not self._is_expired(node):
                    count += 1
        return count
```

### Tests: `test_ttl_cache.py`

```python
import pytest
from unittest.mock import patch
from ttl_cache import TTLCache

class TestTTLCache:

    @patch('ttl_cache.time.monotonic')
    def test_basic_get_put(self, mock_time):
        """Test basic insertion and retrieval."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Verify LRU update (head position)
        # We can't easily check head pointer directly without exposing internals,
        # but we can verify subsequent operations work correctly.
        cache.put("key2", "value2")
        assert cache.get("key2") == "value2"

    @patch('ttl_cache.time.monotonic')
    def test_capacity_eviction_lru_order(self, mock_time):
        """Test that LRU eviction works correctly when capacity is reached."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # 'a' should be evicted as it was least recently used
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    @patch('ttl_cache.time.monotonic')
    def test_ttl_expiry(self, mock_time):
        """Test that items expire based on default TTL."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key1", "value1")
        
        # Access immediately
        assert cache.get("key1") == "value1"
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        # Should be expired
        assert cache.get("key1") is None

    @patch('ttl_cache.time.monotonic')
    def test_custom_per_key_ttl(self, mock_time):
        """Test that custom TTL overrides default TTL."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1", ttl=2.0)
        
        # Valid initially
        assert cache.get("key1") == "value1"
        
        # Advance time past custom TTL but before default
        mock_time.return_value = 3.0
        
        # Should be expired due to custom TTL
        assert cache.get("key1") is None

    @patch('ttl_cache.time.monotonic')
    def test_delete(self, mock_time):
        """Test delete operation and return value."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("key1", "value1")
        
        # Delete existing
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Delete non-existing
        assert cache.delete("key2") is False

    @patch('ttl_cache.time.monotonic')
    def test_size_with_mixed_expired_valid(self, mock_time):
        """Test size() returns count of non-expired items only."""
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Initially size is 3
        assert cache.size() == 3
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        # All should be expired, size should be 0
        assert cache.size() == 0
        
        # Add new item
        cache.put("d", 4)
        # Size should be 1
        assert cache.size() == 1
```

### How to Run

1.  Save the first code block as `ttl_cache.py`.
2.  Save the second code block as `test_ttl_cache.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_ttl_cache.py -v
    ```

### Key Design Decisions

1.  **Doubly Linked List**: Implemented manually (`Node` class) instead of using `OrderedDict` to strictly adhere to the requirement. This allows O(1) removal from any position and O(1) insertion at the head.
2.  **Time Tracking**: Used `time.monotonic()` for clock skew resistance. The internal clock is synchronized on every operation (`_update_time`) to ensure consistency between `get`, `put`, and `size`.
3.  **Lazy Cleanup vs. Immediate**:
    *   `get`: Checks expiry immediately. If expired, removes and returns `None`.
    *   `put`: Explicitly calls `_clear_expired` at the start. This satisfies the requirement "If all items are expired, clear them all first" before checking capacity.
    *   `size`: Iterates through the map to count valid items. While technically O(N) in the worst case (if many expired items exist), it avoids the overhead of a full traversal during every `get`/`put` if the cache is hot, and ensures accuracy without modifying the list structure unnecessarily unless accessed.
4.  **Thread Safety**: Added a `threading.Lock` to protect the internal state during concurrent operations.
5.  **Deterministic Testing**: Used `unittest.mock.patch` on `time.monotonic` to control time progression in tests without relying on `time.sleep`, ensuring fast and reliable CI/CD execution.
