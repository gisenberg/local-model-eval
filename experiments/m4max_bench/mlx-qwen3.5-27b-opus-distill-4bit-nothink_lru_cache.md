# Qwen 3.5 27B Opus-Distilled 4bit MLX — lru_cache

## Thinking (434 chars)

```
Let me think about how to approach this coding problem.

Problem: Implement an LRU cache with TTL in Python. Requirements:

1. Class TTLCache with __init__(capacity, default_ttl)
2. get(key) -> Optional[Any], put(key, value, ttl=None), delete(key) -> bool, size() -> int
3. O(1) average time. Doubly-linked list + hash map, no OrderedDict
4. time.monotonic() for time tracking, lazy cleanup on access

Let me implement this solution.

```

## Output



# LRU Cache with TTL Implementation

## Architecture & Design Decisions

### 1. **Doubly-Linked List Node Structure**
I'll implement a custom doubly-linked list node that stores:
- Key and value for cache data
- Expiration timestamp (absolute time)
- Pointers to previous and next nodes

This enables O(1) insertion, deletion, and movement of nodes.

### 2. **Hash Map for O(1) Lookups**
A Python `dict` maps keys directly to their corresponding nodes, enabling O(1) access to any cached entry.

### 3. **LRU Ordering Strategy**
- **Most recently used**: Moved to the **head** of the list
- **Least recently used**: Always at the **tail**, evicted when capacity is reached

### 4. **Lazy TTL Cleanup**
Instead of background threads or periodic cleanup, I'll check expiration **on-demand** during `get()` operations. This is:
- More efficient for read-heavy workloads
- Simpler to implement and test
- Avoids race conditions with background threads

### 5. **Time Tracking**
Using `time.monotonic()` as required - it's monotonic (can't go backward) and suitable for measuring durations.

### 6. **Edge Cases Handled**
- Capacity of 0 or negative values
- TTL of 0 (immediate expiration)
- Duplicate key updates
- Empty cache operations

---

## Implementation

```python
"""
LRU Cache with TTL (Time-To-Live) support.

A thread-unsafe cache implementation that evicts least-recently-used items
when at capacity, and automatically expires items based on their TTL.

Time Complexity: O(1) average for all operations
Space Complexity: O(capacity)
"""

from __future__ import annotations
import time
from typing import Any, Optional


class _Node:
    """
    Doubly-linked list node storing cache entry data.
    
    Each node contains:
    - key: The cache key
    - value: The cached value
    - expires_at: Absolute timestamp when this entry expires
    - prev/next: Pointers for doubly-linked list traversal
    """
    
    __slots__ = ('key', 'value', 'expires_at', 'prev', 'next')
    
    def __init__(self, key: Any, value: Any, expires_at: float) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class TTLCache:
    """
    A Least-Recently-Used (LRU) cache with Time-To-Live (TTL) support.
    
    Features:
    - O(1) get, put, and delete operations
    - Automatic eviction of least-recently-used items at capacity
    - Per-entry TTL with lazy expiration on access
    - Configurable default TTL for all entries
    
    Thread Safety: NOT thread-safe. External synchronization required.
    
    Example:
        cache = TTLCache(capacity=100, default_ttl=60)
        cache.put("key", "value")
        value = cache.get("key")
    """
    
    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize the TTL cache.
        
        Args:
            capacity: Maximum number of entries. Must be positive.
            default_ttl: Default time-to-live in seconds for entries.
        
        Raises:
            ValueError: If capacity is not positive or default_ttl is negative.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if default_ttl < 0:
            raise ValueError("Default TTL must be non-negative")
        
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._size = 0
        
        # Hash map from key to node for O(1) lookups
        self._cache: dict[Any, _Node] = {}
        
        # Doubly-linked list sentinels (dummy nodes)
        # head.next points to most-recently-used
        # tail.prev points to least-recently-used
        self._head = _Node(None, None, 0.0)
        self._tail = _Node(None, None, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head
    
    def _is_expired(self, node: _Node) -> bool:
        """Check if a node has expired."""
        return time.monotonic() >= node.expires_at
    
    def _remove_node(self, node: _Node) -> None:
        """
        Remove a node from the doubly-linked list.
        
        O(1) operation that updates neighboring node pointers.
        """
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def _add_to_head(self, node: _Node) -> None:
        """
        Add a node immediately after the head (most-recently-used position).
        
        O(1) operation.
        """
        node.next = self._head.next
        node.prev = self._head
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node
    
    def _evict_lru(self) -> None:
        """
        Evict the least-recently-used entry (at tail).
        
        Assumes capacity is full and size > 0.
        """
        lru_node = self._tail.prev
        if lru_node and lru_node != self._head:
            self._remove_node(lru_node)
            if lru_node.key is not None:
                del self._cache[lru_node.key]
            self._size -= 1
    
    def _lazy_cleanup(self, node: _Node) -> bool:
        """
        Check if node is expired and remove it if so.
        
        Returns:
            True if node was removed (expired), False otherwise.
        """
        if self._is_expired(node):
            self._remove_node(node)
            if node.key is not None:
                del self._cache[node.key]
            self._size -= 1
            return True
        return False
    
    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most-recently-used position)."""
        self._remove_node(node)
        self._add_to_head(node)
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get the value associated with the given key.
        
        Performs lazy TTL cleanup - expired entries are removed on access.
        Moves accessed entry to head (most-recently-used).
        
        Args:
            key: The key to look up.
            
        Returns:
            The cached value if found and not expired, None otherwise.
            
        Time Complexity: O(1) average
        """
        node = self._cache.get(key)
        
        if node is None:
            return None
        
        # Lazy cleanup: check if entry has expired
        if self._lazy_cleanup(node):
            return None
        
        # Move to head (most-recently-used)
        self._move_to_head(node)
        
        return node.value
    
    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair in the cache.
        
        If the key already exists, updates the value and refreshes the TTL.
        If the cache is at capacity, evicts the least-recently-used entry.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional custom TTL in seconds. Uses default_ttl if not specified.
            
        Time Complexity: O(1) average
        """
        if ttl is None:
            ttl = self._default_ttl
        
        expires_at = time.monotonic() + ttl
        
        # Check if key already exists
        existing_node = self._cache.get(key)
        
        if existing_node is not None:
            # Update existing entry
            existing_node.value = value
            existing_node.expires_at = expires_at
            self._move_to_head(existing_node)
            return
        
        # Create new node
        new_node = _Node(key, value, expires_at)
        
        # Evict LRU if at capacity
        if self._size >= self._capacity:
            self._evict_lru()
        
        # Insert new node
        self._cache[key] = new_node
        self._add_to_head(new_node)
        self._size += 1
    
    def delete(self, key: Any) -> bool:
        """
        Delete an entry from the cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key was found and deleted, False otherwise.
            
        Time Complexity: O(1) average
        """
        node = self._cache.get(key)
        
        if node is None:
            return False
        
        self._remove_node(node)
        del self._cache[key]
        self._size -= 1
        
        return True
    
    def size(self) -> int:
        """
        Get the current number of entries in the cache.
        
        Note: This returns the count of entries that haven't been lazily
        cleaned up yet. For accurate count, call get() on all keys first.
        
        Returns:
            The number of entries in the cache.
            
        Time Complexity: O(1)
        """
        return self._size
```

---

## Tests

```python
"""
Pytest test suite for TTLCache.

Uses unittest.mock.patch to control time.monotonic() for deterministic testing.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from typing import Any

from lru_cache_ttl import TTLCache


class TestTTLCache:
    """Test suite for TTLCache functionality."""
    
    @pytest.fixture
    def mock_time(self) -> MagicMock:
        """Fixture providing a mock for time.monotonic()."""
        with patch('lru_cache_ttl.time.monotonic', return_value=0.0) as mock:
            yield mock
    
    def test_init_valid_parameters(self, mock_time: MagicMock) -> None:
        """Test initialization with valid parameters."""
        cache = TTLCache(capacity=10, default_ttl=60.0)
        
        assert cache._capacity == 10
        assert cache._default_ttl == 60.0
        assert cache._size == 0
        assert cache.size() == 0
    
    def test_init_invalid_capacity(self) -> None:
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            TTLCache(capacity=0, default_ttl=60.0)
        
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            TTLCache(capacity=-1, default_ttl=60.0)
    
    def test_init_invalid_ttl(self) -> None:
        """Test that invalid default_ttl raises ValueError."""
        with pytest.raises(ValueError, match="Default TTL must be non-negative"):
            TTLCache(capacity=10, default_ttl=-1.0)
    
    def test_put_and_get(self, mock_time: MagicMock) -> None:
        """Test basic put and get operations."""
        cache = TTLCache(capacity=3, default_ttl=60.0)
        
        # Put some values
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Verify values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.size() == 3
        
        # Non-existent key returns None
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self, mock_time: MagicMock) -> None:
        """Test that LRU entry is evicted when capacity is reached."""
        cache = TTLCache(capacity=3, default_ttl=60.0)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it most-recently-used
        cache.get("key1")
        
        # Add key4 - should evict key2 (LRU)
        cache.put("key4", "value4")
        
        assert cache.size() == 3
        assert cache.get("key1") == "value1"  # Still exists
        assert cache.get("key2") is None       # Evicted
        assert cache.get("key3") == "value3"  # Still exists
        assert cache.get("key4") == "value4"  # New entry
    
    def test_ttl_expiration(self, mock_time: MagicMock) -> None:
        """Test that entries expire after their TTL."""
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        # Put entry with custom TTL of 5 seconds
        cache.put("key1", "value1", ttl=5.0)
        assert cache.get("key1") == "value1"
        
        # Advance time to 3 seconds - not expired yet
        mock_time.return_value = 3.0
        assert cache.get("key1") == "value1"
        
        # Advance time to 6 seconds - expired
        mock_time.return_value = 6.0
        assert cache.get("key1") is None
        assert cache.size() == 0  # Entry removed on lazy cleanup
    
    def test_delete(self, mock_time: MagicMock) -> None:
        """Test delete operation."""
        cache = TTLCache(capacity=3, default_ttl=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Delete existing key
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.size() == 1
        
        # Delete non-existent key
        assert cache.delete("nonexistent") is False
        assert cache.size() == 1
        
        # Verify remaining entry
        assert cache.get("key2") == "value2"
    
    def test_update_existing_key(self, mock_time: MagicMock) -> None:
        """Test that updating a key refreshes its TTL and moves it to MRU."""
        cache = TTLCache(capacity=3, default_ttl=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Update key1 with new value and custom TTL
        cache.put("key1", "updated_value", ttl=120.0)
        
        # Verify value was updated
        assert cache.get("key1") == "updated_value"
        
        # Verify key1 is now MRU (key2 should be evicted next)
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key2
        
        assert cache.get("key1") == "updated_value"  # Still exists
        assert cache.get("key2") is None             # Was evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_capacity_one(self, mock_time: MagicMock) -> None:
        """Test cache with capacity of 1."""
        cache = TTLCache(capacity=1, default_ttl=60.0)
        
        cache.put("key1", "value1")
        assert cache.size() == 1
        
        cache.put("key2", "value2")
        assert cache.size() == 1
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
    
    def test_ttl_zero(self, mock_time: MagicMock) -> None:
        """Test that TTL of 0 means immediate expiration."""
        cache = TTLCache(capacity=3, default_ttl=60.0)
        
        # Put with TTL of 0
        cache.put("key1", "value1", ttl=0.0)
        
        # Should be immediately expired
        assert cache.get("key1") is None
        assert cache.size() == 0
```

---

## Quick Reference

| Operation | Time Complexity | Description |
|-----------|-----------------|-------------|
| `get(key)` | O(1) | Retrieve value, lazy TTL check, move to MRU |
| `put(key, value, ttl)` | O(1) | Insert/update, evict LRU if needed |
| `delete(key)` | O(1) | Remove specific entry |
| `size()` | O(1) | Return current entry count |

### Running Tests

```bash
# Install dependencies
pip install pytest

# Run tests
pytest -v test_lru_cache_ttl.py

# Run with coverage
pytest --cov=lru_cache_ttl --cov-report=term-missing test_lru_cache_ttl.py
```
