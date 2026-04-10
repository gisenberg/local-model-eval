# Gemma 4 26B-A4B Q6_K (turbo4 KV) — lru_cache

## Output

To implement an LRU (Least Recently Used) cache with TTL (Time To Live), we need two primary data structures working in tandem:
1.  **A Hash Map (Dictionary):** To provide $O(1)$ access to nodes by key.
2.  **A Doubly Linked List:** To maintain the order of usage. The most recently used items are at the head, and the least recently used are at the tail.

To handle TTL, each node will store an `expiry` timestamp. We will use "lazy cleanup," meaning we only check if an item is expired when a user attempts to access it.

Here is the complete implementation.

```python
import time
from typing import Any, Dict, Optional


class Node:
    """A node in the doubly linked list."""
    __slots__ = ("key", "value", "expiry", "prev", "next")

    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) support.
    Complexity: O(1) average for get, put, and delete.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default time-to-live in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail to simplify edge cases in doubly linked list
        self.head = Node(None, None, 0)  # Dummy head
        self.tail = Node(None, None, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (most recent position)."""
        first_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = first_node
        if first_node:
            first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value. Returns None if key is missing or expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy cleanup: Check if expired
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None

        # Move to front (mark as recently used)
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a value.
        
        :param key: The key to store.
        :param value: The value to store.
        :param ttl: Custom TTL for this specific key. If None, uses default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            # Create new
            if len(self.cache) >= self.capacity:
                # Evict least recently used (the node before dummy tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """Remove a key from the cache. Returns True if successful."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Return current number of items in cache (including potentially expired ones)."""
        return len(self.cache)
```

### Unit Tests

The following tests use `unittest.mock.patch` to control `time.monotonic()`, allowing us to simulate the passage of time without using `time.sleep()`.

```python
import pytest
from unittest.mock import patch

def test_put_and_get():
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.size() == 1

def test_lru_eviction():
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")      # "a" is now most recent
        cache.put("c", 3)    # "b" should be evicted
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

def test_ttl_expiration():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        
        # Advance time to 105 (not expired)
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Advance time to 111 (expired)
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=100)
        cache.put("short", "val", ttl=5)
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None

def test_delete():
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("non_existent") is False

def test_update_existing_key():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        cache.put("a", 2) # Update
        assert cache.get("a") == 2
        assert cache.size() == 1
```

### Key Implementation Details:
1.  **Complexity:** `get`, `put`, and `delete` all operate in $O(1)$ time because dictionary lookups and doubly linked list pointer updates are constant time.
2.  **Lazy Cleanup:** We don't run a background thread to clean up expired keys (which would be complex and resource-heavy). Instead, we check if a key is expired only when `get()` is called.
3.  **Memory Management:** We use `__slots__` in the `Node` class to reduce memory overhead, which is important for large caches.
4.  **Dummy Nodes:** The `head` and `tail` dummy nodes prevent `if node.prev is not None` checks, making the linked list logic cleaner and faster.
