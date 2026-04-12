# Gemma 4 26B-A4B Q6_K (f16 @ 32K, new base default ub) — lru_cache

**Hypothesis:** Refresh baseline: f16 at 32K works on new base without -ub 256

## Output

To implement an LRU (Least Recently Used) cache with TTL (Time To Live), we need to combine two data structures:
1.  **A Doubly Linked List**: To maintain the order of access for the LRU eviction policy.
2.  **A Hash Map (Dictionary)**: To provide $O(1)$ access to the nodes in the linked list.

To handle TTL, each node will store an expiration timestamp. We will use "lazy cleanup," meaning we check if a key is expired only when it is accessed via `get`.

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
    Evicts least recently used items when capacity is reached.
    Items are also considered 'deleted' if they have expired.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Args:
            capacity: Maximum number of items allowed in the cache.
            default_ttl: Default time-to-live in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy head and tail to simplify linked list operations
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node right after the dummy head (most recent)."""
        first_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = first_node
        if first_node:
            first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        """Moves an existing node to the front (most recent)."""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: Any) -> Optional[Any]:
        """
        Returns the value associated with key if it exists and is not expired.
        Otherwise returns None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        # Lazy cleanup: check if expired
        if time.monotonic() > node.expiry:
            self.delete(key)
            return None

        # Move to front because it was accessed
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates a key-value pair.
        
        Args:
            key: The key to store.
            value: The value to store.
            ttl: Optional specific TTL for this key. If None, uses default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU (the node before dummy tail)
                lru_node = self.tail.prev
                if lru_node != self.head:
                    self.delete(lru_node.key)

            new_node = Node(key, value, expiry)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def delete(self, key: Any) -> bool:
        """Removes the key from the cache. Returns True if key existed."""
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the cache (including expired ones)."""
        return len(self.cache)
```

### Unit Tests

The following tests use `unittest.mock.patch` to control `time.monotonic()`, allowing us to simulate the passage of time precisely.

```python
import pytest
from unittest.mock import patch

def test_lru_eviction():
    """Test that the least recently used item is evicted when capacity is reached."""
    with patch('time.monotonic', return_value=100.0):
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")      # 'a' is now most recent, 'b' is LRU
        cache.put("c", 3)   # Should evict 'b'
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None

def test_ttl_expiration():
    """Test that items expire after their TTL has passed."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=10)
        
        cache.put("a", 1)
        
        # Advance time to 105 (not expired)
        mock_time.return_value = 105.0
        assert cache.get("a") == 1
        
        # Advance time to 111 (expired)
        mock_time.return_value = 111.0
        assert cache.get("a") is None
        assert cache.size() == 0

def test_custom_ttl():
    """Test that specific TTLs provided in put() override the default."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=5, default_ttl=100)
        
        cache.put("short", "val", ttl=5)
        cache.put("long", "val") # uses default 100
        
        mock_time.return_value = 106.0
        assert cache.get("short") is None
        assert cache.get("long") == "val"

def test_update_existing_key():
    """Test that putting an existing key updates value and resets position/TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        cache = TTLCache(capacity=2, default_ttl=10)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10) # Update 'a', making it most recent
        
        cache.put("c", 3)  # Should evict 'b'
        
        assert cache.get("a") == 10
        assert cache.get("b") is None
        assert cache.get("c") == 3

def test_delete():
    """Test the delete functionality."""
    cache = TTLCache(capacity=5, default_ttl=10)
    cache.put("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False
    assert cache.get("a") is None

def test_size_and_capacity():
    """Test that size reflects the number of items in the hash map."""
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3) # Evicts 'a'
    
    assert cache.size() == 2
    assert "a" not in cache.cache
```

### Complexity Analysis
*   **`get(key)`**: $O(1)$ average. Hash map lookup is $O(1)$, and moving a node in a doubly linked list is $O(1)$.
*   **`put(key, value)`**: $O(1)$ average. Hash map insertion is $O(1)$, and adding/removing nodes from the linked list is $O(1)$.
*   **`delete(key)`**: $O(1)$ average.
*   **Space Complexity**: $O(N)$ where $N$ is the capacity, to store the nodes in both the dictionary and the linked list.
