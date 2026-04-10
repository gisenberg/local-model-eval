# Gemma 4 26B-A4B Q6_K — lru_cache

## Output

To implement an LRU (Least Recently Used) cache with TTL (Time To Live), we need two primary data structures working in tandem:
1.  **A Hash Map (Dictionary):** For $O(1)$ access to nodes by key.
2.  **A Doubly Linked List:** To maintain the order of usage. The most recently used items are at the head, and the least recently used are at the tail.

To handle TTL, each node will store an `expiry` timestamp. We will use "lazy cleanup," meaning we check if a key is expired only when someone tries to access it.

Here is the complete implementation.

```python
import time
from typing import Any, Optional, Dict


class Node:
    """A node in the doubly linked list."""

    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) support.
    Uses a Doubly Linked List and a Hash Map for O(1) average time complexity.
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
        self.head = Node(None, None, float('inf'))
        self.tail = Node(None, None, float('inf'))
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
        """Add a node right after the dummy head (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_front(self, node: Node) -> None:
        """Move an existing node to the front (mark as recently used)."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache. Returns None if key is missing or expired.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy cleanup: check if expired
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        :param key: The key to store.
        :param value: The value to store.
        :param ttl: Optional specific TTL for this key. Defaults to default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict the least recently used (the node before dummy tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """Remove a key from the cache. Returns True if key existed, False otherwise."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Return the current number of items in the cache."""
        return len(self.cache)
```

### Unit Tests

The following tests use `unittest.mock.patch` to control `time.monotonic()`, allowing us to simulate the passage of time precisely.

```python
import pytest
from unittest.mock import patch

def test_basic_put_get():
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
        # Access 'a' to make 'b' the LRU
        cache.get("a")
        # Put 'c', should evict 'b'
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None
        assert cache.size() == 2

def test_ttl_expiration():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1) # Expires at 110.0
        
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
        cache = TTLCache(capacity=2, default_ttl=100) # Long default
        
        cache.put("short", "val", ttl=5) # Expires at 105
        cache.put("long", "val")        # Expires at 200
        
        mock_time.return_value = 110.0
        assert cache.get("short") is None
        assert cache.get("long") == "val"

def test_delete():
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False
        assert cache.size() == 0

def test_update_existing_key():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1)
        cache.put("a", 2) # Update
        
        assert cache.get("a") == 2
        assert cache.size() == 1
```

### Complexity Analysis
*   **`get(key)`**: $O(1)$ average. Dictionary lookup is $O(1)$, and moving a node in a doubly linked list is $O(1)$.
*   **`put(key, value)`**: $O(1)$ average. Dictionary insertion is $O(1)$, and adding/removing nodes from the linked list is $O(1)$.
*   **`delete(key)`**: $O(1)$ average.
*   **Space Complexity**: $O(N)$ where $N$ is the capacity, as we store each item in both a dictionary and a linked list.
