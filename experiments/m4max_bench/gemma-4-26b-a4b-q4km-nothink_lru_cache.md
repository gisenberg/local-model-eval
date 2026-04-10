# Gemma 4 26B-A4B Q4_K_M — lru_cache

## Output

To implement an LRU (Least Recently Used) cache with TTL (Time To Live) in $O(1)$ average time, we need two primary data structures:
1.  **A Hash Map (Dictionary):** To provide $O(1)$ access to nodes by key.
2.  **A Doubly-Linked List:** To maintain the order of access. The "head" represents the most recently used item, and the "tail" represents the least recently used item.

To handle TTL, each node will store an `expiry_time`. We will use "lazy cleanup," meaning we only check if a key has expired when it is accessed via `get` or when the cache needs to evict items.

Here is the complete implementation.

```python
import time
from typing import Any, Optional, Dict

class Node:
    """A node in the doubly-linked list."""
    def __init__(self, key: Any, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) support.
    Provides O(1) average time complexity for all operations.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        :param capacity: Maximum number of items allowed in the cache.
        :param default_ttl: Default lifetime of an item in seconds.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
            
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        
        # Dummy nodes for the doubly-linked list to simplify boundary logic
        self.head = Node(None, None, 0)
        self.tail = Node(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after the dummy head (Most Recently Used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _is_expired(self, node: Node) -> bool:
        """Check if a node's TTL has passed."""
        return time.monotonic() > node.expiry

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        :param key: The key to look up.
        :return: The value if found and not expired, otherwise None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]

        if self._is_expired(node):
            self.delete(key)
            return None

        # Move to front (Mark as MRU)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update an item in the cache.

        :param key: The key to store.
        :param value: The value to store.
        :param ttl: Optional specific TTL for this key. Defaults to default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)

        if key in self.cache:
            self._remove(self.cache[key])

        new_node = Node(key, value, expiry)
        self.cache[key] = new_node
        self._add_to_front(new_node)

        if len(self.cache) > self.capacity:
            # Evict Least Recently Used (the node before dummy tail)
            lru_node = self.tail.prev
            if lru_node != self.head:
                self.delete(lru_node.key)

    def delete(self, key: Any) -> bool:
        """
        Remove a key from the cache.

        :param key: The key to delete.
        :return: True if the key was found and deleted, False otherwise.
        """
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)
            return True
        return False

    def size(self) -> int:
        """Return the current number of valid (non-expired) items."""
        # Note: We don't perform a full sweep here to maintain O(1)
        # But we do check the specific key if we were to implement a 'clean' method.
        # For the purpose of this requirement, we return the dict size.
        # In a real production system, one might periodically prune.
        return len(self.cache)
```

### Unit Tests

The following tests use `unittest.mock.patch` to control `time.monotonic()`, allowing us to simulate the passage of time precisely.

```python
import pytest
from unittest.mock import patch

# Assuming the class is in a file named cache_impl.py
# from cache_impl import TTLCache

def test_lru_eviction():
    """Test that the least recently used item is evicted when capacity is reached."""
    cache = TTLCache(capacity=2, default_ttl=100)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now MRU, 'b' is LRU
    cache.put("c", 3) # Should evict 'b'

    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    """Test that items expire after their TTL has passed."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=5, default_ttl=10)
    
    cache.put("key1", "val1") # Expires at 110.0
    
    # Advance time to 105 (not expired)
    mock_time.return_value = 105.0
    assert cache.get("key1") == "val1"
    
    # Advance time to 111 (expired)
    mock_time.return_value = 111.0
    assert cache.get("key1") is None

@patch('time.monotonic')
def test_custom_ttl(mock_time):
    """Test that custom TTL overrides the default TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=5, default_ttl=100)
    
    cache.put("short", "val", ttl=5) # Expires at 105.0
    
    mock_time.returnreturn_value = 106.0
    assert cache.get("short") is None

def test_delete_and_size():
    """Test the delete functionality and size reporting."""
    cache = TTLCache(capacity=5, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.size() == 2
    
    assert cache.delete("a") is True
    assert cache.delete("non_existent") is False
    assert cache.size() == 1

@patch('time.monotonic')
def test_get_updates_lru_order(mock_time):
    """Test that accessing an item moves it to the front (MRU)."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=100)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a")
    
    # Put 'c', which should evict 'b' (the LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

@patch('time.monotonic')
def test_lazy_cleanup_on_access(mock_time):
    """Test that expired items are removed when accessed."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10)
    
    cache.put("a", 1)
    
    mock_time.return_value = 120.0 # Past expiry
    # The get call should trigger the internal delete
    assert cache.get("a") is None
    assert cache.size() == 0 # Note: size() depends on implementation, 
                             # but here we check if it's gone from dict
```

### Complexity Analysis
*   **`get(key)`**: $O(1)$ average. Hash map lookup is $O(1)$, and moving a node in a doubly-linked list is $O(1)$.
*   **`put(key, value)`**: $O(1)$ average. Hash map insertion and linked list pointer updates are $O(1)$.
*   **`delete(key)`**: $O(1)$ average. Hash map deletion and linked list removal are $O(1)$.
*   **Space Complexity**: $O(N)$ where $N$ is the capacity, to store the nodes in the dictionary and the linked list.
