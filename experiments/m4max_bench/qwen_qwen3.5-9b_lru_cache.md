# qwen/qwen3.5-9b — lru_cache

## Thinking (17420 chars)

```
Here's a thinking process that leads to the suggested solution:

1.  **Understand the Goal:** The objective is to implement an LRU (Least Recently Used) Cache with Time-To-Live (TTL) expiration in Python. It needs specific performance characteristics (O(1) operations), specific implementation details (Doubly-linked list + Hash map, no `OrderedDict`), time tracking (`time.monotonic()`), lazy cleanup, and comprehensive testing using `unittest.mock.patch`.

2.  **Analyze Requirements:**
    *   **Class Name:** `TTLCache`.
    *   **Constructor:** `__init__(self, capacity: int, default_ttl: float)`.
    *   **Methods:**
        *   `get(key)` -> Returns value or None.
        *   `put(key, value, ttl=None)` -> Sets value with optional specific TTL (defaults to `default_ttl`).
        *   `delete(key)` -> Removes key, returns bool.
        *   `size()` -> Returns current number of items.
    *   **Performance:** O(1) average time complexity for all operations.
    *   **Data Structures:** Doubly-linked list + Hash map (dictionary). Explicitly forbidden: `OrderedDict`.
    *   **TTL Logic:** Use `time.monotonic()`. Lazy cleanup means checking expiration only when accessing the item (`get` or `put`), not on every iteration.
    *   **Code Quality:** Type hints, docstrings.
    *   **Testing:** 6 pytest tests using `unittest.mock.patch` on `time.monotonic`.

3.  **Design the Data Structures:**
    *   **Hash Map (`self.cache`):** Maps `key -> Node`. This allows O(1) lookup to find a node.
    *   **Doubly-Linked List (`self.head`, `self.tail`):** Maintains order of usage.
        *   Head: Most Recently Used (MRU).
        *   Tail: Least Recently Used (LRU).
        *   Node class needs `key`, `value`, `next`, `prev`.
    *   **TTL Tracking:** Each node needs to store the `expiry_time` (absolute timestamp) or calculate it based on insertion time. Storing absolute expiry time in the node is cleaner for O(1) checks.

4.  **Implementing Operations:**
    *   **`put(key, value, ttl=None)`:**
        1.  Calculate `expiry_time = current_time + (ttl or default_ttl)`.
        2.  Check if key exists.
            *   If yes: Update value and expiry time. Move node to head.
            *   If no: Create new node. Add to head.
        3.  **Lazy Cleanup:** Before adding, check if the tail is expired? No, requirement says "lazy cleanup on access". However, strictly speaking, `size()` should reflect valid items. If we don't clean up expired items immediately, `size()` might be wrong if we count nodes in the list.
        *   *Correction:* The prompt says "lazy cleanup on access". This usually implies checking expiration when `get` is called. But what about `put`? If I put an item that expires 0 seconds from now, it should probably be treated as expired immediately or handled gracefully.
        *   *Constraint Check:* "Lazy cleanup on access". This means we don't sweep the list periodically. We check expiration when `get` is called.
        *   *Edge Case:* If I call `put`, and the item expires instantly, should it be stored? Usually, yes, but `get` will return None immediately. To keep `size()` accurate regarding valid items, we might need to remove expired nodes from the list when they are accessed or inserted if they are already expired.
        *   *Refinement:* Standard LRU with TTL usually removes expired entries upon access (`get`). If an entry is expired, it's effectively gone. So, `get` checks expiration. If expired, return None and remove node. `put` adds new node.
    *   **`get(key)`:**
        1.  Check if key in map.
        2.  If yes: Check if `current_time > expiry_time`.
            *   If expired: Remove from list, delete from map, return None.
            *   If valid: Move node to head (MRU), return value.
        3.  If no: Return None.
    *   **`delete(key)`:**
        1.  Check if key in map.
        2.  If yes: Remove from list, delete from map, return True.
        3.  If no: Return False.
    *   **`size()`:**
        1.  Count nodes in the linked list? Or maintain a counter `self._count`. Maintaining a counter is O(1). Since we remove on expiration and deletion, the counter stays accurate.

5.  **Handling Capacity:**
    *   When adding a new item (or updating an existing one that exceeds capacity?), wait. LRU eviction happens when `len(cache) > capacity`.
    *   Standard LRU: If size == capacity and key is new -> Evict Tail.
    *   Wait, does TTL affect capacity? Usually, capacity limits the number of *active* entries. Expired entries are removed lazily. So if I have 10 items, one expires, `size()` should ideally reflect valid items. If I use a counter `_count`, I must decrement it when removing expired items during `get`.
    *   Eviction Logic: When `put` is called and `len(cache) >= capacity`:
        *   Remove Tail (LRU).
        *   Add new item to Head.
    *   Wait, if the Tail is already expired? The requirement says "lazy cleanup on access". This implies we don't clean up during `put`. But if we add a new item and the list is full, we must evict the LRU item. If that LRU item was already expired (but not cleaned yet), should we evict it? Yes, because it's taking up space.
    *   Actually, to keep O(1) and simple: `size()` usually tracks valid items. But if we rely on a counter `_count`, we must ensure consistency.
    *   Let's stick to the standard pattern: `self.cache` dict stores nodes. `self.head`, `self.tail`. `self._size` tracks number of items in the map/list.
    *   When `get` finds an expired item, remove it from list and map, decrement `_size`.
    *   When `put` adds a new item:
        *   If key exists: Update value/time, move to head.
        *   If key doesn't exist:
            *   If `_size == capacity`: Remove Tail (evict). Decrement `_size`.
            *   Add new node to Head. Increment `_size`.

6.  **Refining the "Lazy Cleanup" Requirement:**
    *   "Lazy cleanup on access" means we don't run a background thread or sweep function. We check `time.monotonic()` inside `get`.
    *   What about `put`? If I put an item with TTL=0, it expires immediately. `get` will handle it.
    *   What about `size()`? It should return the number of items currently in the map/list. Since we remove expired items on `get`, `size` is accurate for valid items.

7.  **Drafting the Code:**
    *   Need a `Node` class (inner or outer). Inner is cleaner.
    *   Need to import `time`.
    *   Need to handle `Optional` types.
    *   Need to mock `time.monotonic` in tests.

8.  **Drafting the Tests:**
    *   Test 1: Basic Put/Get (No TTL).
    *   Test 2: LRU Eviction (Capacity reached, new item evicts old).
    *   Test 3: TTL Expiration (Item expires between get calls).
    *   Test 4: Update Existing Key (Refreshes TTL).
    *   Test 5: Delete Method.
    *   Test 6: Size Accuracy with Lazy Cleanup (Ensure expired items don't count towards size if removed on access, or ensure `size` reflects valid items). Actually, a better test for lazy cleanup is checking that `get` returns None after expiration and removes the node so `size` drops.

9.  **Addressing Specific Constraints:**
    *   "No OrderedDict": Use manual Doubly Linked List.
    *   "O(1)": Dict lookup + Node manipulation must be O(1).
    *   "unittest.mock.patch": Must patch `time.monotonic`.

10. **Refining Implementation Details:**
    *   `Node` class: `__slots__` for memory efficiency (optional but good practice), `key`, `value`, `prev`, `next`, `expiry`.
    *   `TTLCache`: `head`, `tail`, `cache` (dict), `_size`.
    *   Helper methods: `_add_to_head`, `_remove_node`, `_evict_lru`.

11. **Self-Correction/Edge Case Check:**
    *   *Issue:* If I call `put` on a key that exists, do I update the expiry? Yes.
    *   *Issue:* If I call `put` and capacity is full, I evict tail. What if tail is expired? It doesn't matter, it's still taking up space in the map. We remove it to make room.
    *   *Issue:* `size()`. Should it return `_size` or count nodes? `_size` is O(1). Counting nodes is O(N). Requirement says O(1) average time. So maintain `_size`.
    *   *Mocking:* `time.monotonic()` returns a float. Need to ensure tests set specific times.

12. **Writing the Code:** (Mental Draft)
    ```python
    class Node:
        def __init__(self, key, value, expiry):
            self.key = key
            self.value = value
            self.expiry = expiry
            self.prev = None
            self.next = None

    class TTLCache:
        def __init__(self, capacity, default_ttl):
            self.capacity = capacity
            self.default_ttl = default_ttl
            self.cache = {} # key -> Node
            self.head = None # MRU
            self.tail = None # LRU
            self._size = 0

        def _add_to_head(self, node):
            ...
        def _remove_node(self, node):
            ...
        def _evict_lru(self):
            ...
        def get(self, key):
            if key not in self.cache: return None
            node = self.cache[key]
            now = time.monotonic()
            if now > node.expiry:
                # Expired
                self._remove_node(node)
                del self.cache[key]
                self._size -= 1
                return None
            self._move_to_head(node)
            return node.value
        def put(self, key, value, ttl=None):
            expiry = time.monotonic() + (ttl or self.default_ttl)
            if key in self.cache:
                # Update existing
                node = self.cache[key]
                node.value = value
                node.expiry = expiry
                self._move_to_head(node)
            else:
                new_node = Node(key, value, expiry)
                self._add_to_head(new_node)
                self.cache[key] = new_node
                self._size += 1
                if self._size > self.capacity:
                    self._evict_lru()
        def delete(self, key):
            ...
        def size(self):
            return self._size
    ```

13. **Final Review against Requirements:**
    *   Class `TTLCache`? Yes.
    *   Methods signatures match? Yes.
    *   O(1)? Yes (Dict + Linked List).
    *   Doubly-linked list + Hash map? Yes. No OrderedDict.
    *   `time.monotonic()`? Yes.
    *   Lazy cleanup? Yes (checked in `get`).
    *   Type hints/Docstrings? Yes.
    *   6 pytest tests with mock? Yes.

14. **Constructing the Test File:**
    *   Need to import `unittest.mock`.
    *   Patch `time.monotonic` globally or on the class instance? Usually patch module level function.
    *   Ensure tests cover: Basic Put/Get, LRU Eviction, TTL Expiration, Update TTL, Delete, Size consistency.

15. **Final Polish:** Check imports, indentation, and docstring formatting. Ensure `Optional` is imported from typing.

Let's write the solution.

*Self-Correction on `size()`:*
If I have 3 items, one expires. `get` removes it. `_size` becomes 2. `size()` returns 2. This is correct behavior for "valid" cache size.
However, if I call `put` and the list is full, I evict the tail. If the tail was expired (but not cleaned yet), does it count towards capacity? Yes, because it's in the map. So eviction logic should just remove the tail node regardless of expiry status to make room for new data.

Wait, one nuance: "Lazy cleanup on access".
If I have 10 items, capacity 5.
Item 1 inserted at T=0, TTL=100.
Item 2 inserted at T=0, TTL=100.
...
Item 6 inserted at T=0, TTL=100. (Evicts Item 1).
Now I wait until T=50. Item 1 is gone.
Wait, what if I insert Item 7 at T=0, TTL=100?
If I don't clean up expired items lazily on `put`, the map grows with expired items.
The requirement says "lazy cleanup on access". This implies we only check expiration when `get` is called.
So, if an item expires, it stays in the map until accessed (`get`) or evicted by capacity pressure.
This means `_size` tracks *all* items currently in the map/list, including potentially expired ones that haven't been accessed yet?
No, standard LRU behavior usually implies `size()` returns valid entries. But if we don't clean up on `put`, an expired entry sits there.
If I call `get(key)` and it's expired, I remove it. `_size` decreases.
If I call `put(new_key)`, and capacity is full, I evict tail.
Does the tail need to be checked for expiration before eviction? No, because it occupies a slot.
So `_size` tracks items in the map. When an item expires on `get`, we remove it from map and decrement `_size`.
This satisfies "lazy cleanup".

Okay, let's finalize the code structure.

One detail: `time.monotonic()` returns float seconds since epoch (or similar).
Mocking needs to return specific floats.

Let's assemble the Python file.

```python
import time
from typing import Any, Optional

class Node:
    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        ...
```

Wait, `delete` method signature in requirements: `delete(key) -> bool`.
My logic handles this.

Okay, ready to generate the response. I will combine the implementation and tests into a single block or separate blocks? Usually, providing the implementation and then the test file is better for clarity, but often users want one copy-pasteable solution. I'll provide the implementation class first, then the test script. Actually, standard practice for these prompts is to provide the code that can be run. I will put them in separate sections or a single block if it fits well. Given the instruction "Write 6 pytest tests", I should probably provide the test file content as well. I'll structure it as:
1. Implementation (`ttl_cache.py`)
2. Tests (`test_ttl_cache.py`)

Wait, checking the prompt again: "Implement an LRU cache... Write 6 pytest tests". It implies I should provide the code for both. I will combine them into one response with clear separation.

One detail on `time.monotonic`: The mock needs to be applied correctly.
`with patch('time.monotonic', return_value=100.0):`

Let's double check the `put` logic regarding capacity.
If `key` exists, we update value and expiry. We move to head. Capacity doesn't change (unless we consider updating as adding? No, standard LRU updates don't increase size).
If `key` does not exist:
  If `_size == capacity`: Evict Tail.
  Add new node. Increment `_size`.

Wait, if I update an existing key, do I need to check capacity? No, it's a replacement.
So logic holds.

One edge case: `ttl=None`. The requirement says `put(key, value, ttl=None)`. If `ttl` is None, use `default_ttl`.
My code handles this.

Let's write the tests carefully.
1. Basic Put/Get.
2. LRU Eviction (Capacity).
3. TTL Expiration (Lazy cleanup on get).
4. Update Existing Key (Refresh TTL).
5. Delete Method.
6. Size Consistency (Expired items removed from size count).

Wait, requirement 6 says "Write 6 pytest tests using unittest.mock.patch".
I will write a `test_ttl_cache.py` file content.

Okay, plan confirmed. I will write the implementation first, then the tests. I'll make sure imports are correct.

One detail: `Optional[Any]`. Need `from typing import Optional, Any`.
Also `Node` class needs to be defined inside or outside. Outside is fine.

Let's refine the `delete` method. It should return `True` if deleted, `False` otherwise.
My logic: `if key in self.cache`: remove and return True else False.

One detail on `size()`: The requirement says `size() -> int`.
If I have expired items that haven't been accessed yet, they are still in the map.
Does `size()` count them?
"Lazy cleanup on access". This implies we don't clean up immediately.
So if an item expires at T=100, and current time is T=50, it is valid.
If current time is T=200, and I call `get`, it becomes invalid and removed.
What about `size()`? It should return the number of items currently in the map (including potentially expired ones that haven't been accessed yet).
However, usually "Size" implies valid entries. But with lazy cleanup, we can't know if an entry is expired without checking time.
If I call `size()`, do I check expiration? That would be O(N) or O(1) per item? No, that defeats the purpose of lazy cleanup (which avoids scanning).
So `size()` should return `_size` which tracks items in the map.
Wait, if an item expires, it is still in the map until accessed. So `size()` returns count of items in map.
This is consistent with "lazy cleanup". We don't sweep.
Okay, I will stick to `_size` tracking items in the map.

Wait, one more check: `put(key, value, ttl=None)`.
If I put a key that already exists, does it count as adding? No, update.
So capacity logic only applies when adding new keys.

Okay, ready to code.

One final check on imports: `import time`, `from typing import Any, Optional`.
Mocking: `from unittest.mock import patch`.

Let's construct the response. I'll provide the implementation and then the test file content. I will combine them into a single markdown block for ease of copy-pasting if needed, but clearly separated. Actually, standard is to provide the solution code. I will provide the implementation class and the test suite.

Wait, I should check if `time.monotonic` needs to be patched in the tests specifically. Yes.
I need to make sure the mock returns a float.

Okay, let's write it.

```

## Output



Here is the complete implementation of the `TTLCache` class and a test suite using `unittest.mock.patch`.

### 1. Implementation (`ttl_cache.py`)

```python
import time
from typing import Any, Optional

class Node:
    """Doubly-linked list node to store cache entries."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with Time-To-Live (TTL) expiration.
    
    Uses a doubly-linked list and hash map for O(1) operations.
    Cleanup of expired items is lazy (happens on access).
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of items in the cache.
            default_ttl: Default time-to-live in seconds for new entries.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[str, Node] = {}  # Maps key -> Node
        self.head: Optional[Node] = None  # Most Recently Used (MRU)
        self.tail: Optional[Node] = None  # Least Recently Used (LRU)
        self._size: int = 0

    def _add_to_head(self, node: Node) -> None:
        """Insert a new node at the head of the list."""
        if self.head is not None:
            self.head.prev = node
        node.next = self.head
        node.prev = None
        self.head = node

        if self.tail is None:
            self.tail = node

    def _remove_node(self, node: Node) -> None:
        """Remove a specific node from the list."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _evict_lru(self) -> None:
        """Remove the LRU item from the cache to make room."""
        if self.tail is not None:
            node = self.tail
            del self.cache[node.key]
            self._remove_node(node)
            self._size -= 1

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key. If expired or missing, returns None and removes from cache.

        Args:
            key: The key to retrieve.

        Returns:
            The value if found and not expired, else None.
        """
        now = time.monotonic()
        
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        # Check expiration (Lazy cleanup)
        if now > node.expiry:
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return None
        
        # Move to head (MRU)
        self._move_to_head(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value for key with optional TTL.

        Args:
            key: The key to set.
            value: The value to store.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        
        if key in self.cache:
            # Update existing entry
            node = self.cache[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
        else:
            # Add new entry
            new_node = Node(key, value, expiry)
            self._add_to_head(new_node)
            self.cache[key] = new_node
            self._size += 1
            
            # Evict if capacity exceeded
            if self._size > self.capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if key did not exist.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self._size -= 1
            return True
        return False

    def size(self) -> int:
        """
        Return the current number of items in the cache.

        Returns:
            The count of items currently stored (including potentially expired ones 
            that haven't been accessed yet).
        """
        return self._size
```

### 2. Tests (`test_ttl_cache.py`)

```python
import unittest
from unittest.mock import patch
from typing import Any, Optional

# Import the implementation here to keep it in one file for copy-paste convenience
# In a real project, these would be separate files.
import sys
sys.path.insert(0, '.') 
from ttl_cache import TTLCache, Node

class TestTTLCache(unittest.TestCase):
    
    def setUp(self):
        self.cache = TTLCache(capacity=2, default_ttl=10.0)

    @patch('ttl_cache.time.monotonic')
    def test_basic_put_and_get(self, mock_time):
        """Test basic put and get functionality."""
        mock_time.return_value = 100.0
        
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        
        self.assertEqual(result, "value1")
        self.assertEqual(self.cache.size(), 1)

    @patch('ttl_cache.time.monotonic')
    def test_lru_eviction(self, mock_time):
        """Test that LRU eviction works when capacity is reached."""
        mock_time.return_value = 100.0
        
        # Fill cache
        self.cache.put("a", "val_a")
        self.cache.put("b", "val_b")
        
        # Access 'a' to make it MRU
        self.cache.get("a")
        
        # Add new item, should evict 'b' (LRU)
        self.cache.put("c", "val_c")
        
        self.assertIsNone(self.cache.get("b"))
        self.assertEqual(self.cache.get("a"), "val_a")
        self.assertEqual(self.cache.get("c"), "val_c")

    @patch('ttl_cache.time.monotonic')
    def test_ttl_expiration_lazy_cleanup(self, mock_time):
        """Test that expired items are removed on access (lazy cleanup)."""
        mock_time.return_value = 100.0
        
        # Insert item with TTL=5 seconds
        self.cache.put("key1", "value1", ttl=5.0)
        
        # Wait until it expires
        mock_time.return_value = 106.0
        
        # Access should return None and remove from cache
        result = self.cache.get("key1")
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('ttl_cache.time.monotonic')
    def test_update_existing_key_refreshes_ttl(self, mock_time):
        """Test that updating a key refreshes its TTL."""
        mock_time.return_value = 100.0
        
        # Insert with short TTL
        self.cache.put("key1", "value1", ttl=5.0)
        
        # Wait for expiration
        mock_time.return_value = 106.0
        
        # Access should fail (expired)
        self.assertIsNone(self.cache.get("key1"))
        
        # Update the key with new value and long TTL
        self.cache.put("key1", "value2", ttl=100.0)
        
        # Now it should be valid again
        mock_time.return_value = 107.0
        result = self.cache.get("key1")
        
        self.assertEqual(result, "value2")

    @patch('ttl_cache.time.monotonic')
    def test_delete_method(self, mock_time):
        """Test delete method returns correct boolean."""
        mock_time.return_value = 100.0
        
        self.cache.put("key1", "value1")
        
        # Delete existing
        result = self.cache.delete("key1")
        self.assertTrue(result)
        self.assertIsNone(self.cache.get("key1"))
        
        # Delete non-existing
        result = self.cache.delete("missing_key")
        self.assertFalse(result)

    @patch('ttl_cache.time.monotonic')
    def test_size_consistency_with_lazy_cleanup(self, mock_time):
        """Test that size reflects valid items after lazy cleanup."""
        mock_time.return_value = 100.0
        
        # Fill cache with capacity 2
        self.cache.put("a", "val_a")
        self.cache.put("b", "val_b")
        
        # Set 'a' to expire immediately (TTL=0)
        self.cache.put("a", "val_a", ttl=0.0)
        
        # Size should still be 2 because cleanup is lazy
        self.assertEqual(self.cache.size(), 2)
        
        # Access 'a', it expires and gets removed
        mock_time.return_value = 100.0
        self.assertIsNone(self.cache.get("a"))
        
        # Size should now be 1
        self.assertEqual(self.cache.size(), 1)

if __name__ == '__main__':
    unittest.main()
```
