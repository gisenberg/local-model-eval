from typing import Any, Optional
from time import monotonic

class _Node:
    """Internal doubly-linked list node."""
    __slots__ = ('key', 'value', 'expiry', 'last_access', 'prev', 'next')

    def __init__(self, key: str, value: Any, expiry: float, last_access: float):
        self.key = key
        self.value = value
        self.expiry = expiry
        self.last_access = last_access
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None

class TTLCache:
    """
    An LRU Cache with Time-To-Live (TTL) support.
    
    Uses a doubly-linked list and a hash map for O(1) average time complexity.
    Expired items are lazily removed upon access or during eviction logic.
    """

    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the cache.
        
        :param capacity: Maximum number of items the cache can hold.
        :param default_ttl: Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache_map: dict[str, _Node] = {}
        
        # Dummy head and tail for the doubly-linked list
        self._head = _Node("", "", 0.0, 0.0)
        self._tail = _Node("", "", 0.0, 0.0)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove_node(self, node: _Node) -> None:
        """Remove a node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: _Node) -> None:
        """Add a node immediately after the head (most recently used)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _evict_lru(self) -> None:
        """
        Evict the least recently used item.
        If the LRU item is expired, it is removed. 
        If all items are expired, this loop continues until a valid item is found or list is empty.
        """
        current = self._tail.prev
        
        # If list is empty, nothing to do
        if current == self._head:
            return

        # We need to find the first non-expired item from the LRU end to evict?
        # Actually, standard LRU eviction removes the LRU item regardless of expiry 
        # if we assume "expired" items are effectively "gone". 
        # However, the requirement says: "If all items are expired, clear them all first."
        # And "evict the least-recently-used non-expired item".
        
        # Strategy: Scan from LRU end.
        # 1. If the LRU item is expired, remove it and continue scanning.
        # 2. If we find a non-expired item, that is the one to evict.
        # 3. If we reach the head (all scanned were expired), we clear everything.
        
        current = self._tail.prev
        found_valid_to_evict = False
        node_to_evict = None
        
        while current != self._head:
            if current.expiry > monotonic():
                # Found a valid item. This is the LRU valid item.
                node_to_evict = current
                found_valid_to_evict = True
                break
            else:
                # Expired item, remove it and move to previous
                self._remove_node(current)
                del self._cache_map[current.key]
                current = current.prev
        
        if not found_valid_to_evict:
            # All items were expired. The loop above removed them one by one.
            # But wait, if the list was full of expired items, the loop above 
            # would have emptied the list (current becomes head).
            # We just need to ensure the list is empty.
            # The loop above handles removal of expired items.
            # If we exit the loop because current == self._head, the list is empty.
            pass
        else:
            # Evict the specific node found
            self._remove_node(node_to_evict)
            del self._cache_map[node_to_evict.key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value for the key if it exists and is not expired.
        Accessing a key makes it the most recently used.
        
        :param key: The key to retrieve.
        :return: The value if valid, else None.
        """
        if key not in self._cache_map:
            return None
        
        node = self._cache_map[key]
        current_time = monotonic()
        
        if node.expiry <= current_time:
            # Expired: remove from list and map
            self._remove_node(node)
            del self._cache_map[key]
            return None
        
        # Update last access time and move to front
        node.last_access = current_time
        self._remove_node(node)
        self._add_to_front(node)
        
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update a key-value pair.
        
        :param key: The key to insert/update.
        :param value: The value to store.
        :param ttl: Optional custom TTL in seconds. Overrides default_ttl.
        """
        current_time = monotonic()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = current_time + effective_ttl
        
        if key in self._cache_map:
            node = self._cache_map[key]
            # Update value and expiry
            node.value = value
            node.expiry = expiry_time
            node.last_access = current_time
            
            # Move to front (it's already in the list, just move it)
            self._remove_node(node)
            self._add_to_front(node)
            return

        # New item
        if len(self._cache_map) >= self.capacity:
            self._evict_lru()
        
        new_node = _Node(key, value, expiry_time, current_time)
        self._cache_map[key] = new_node
        self._add_to_front(new_node)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        :param key: The key to remove.
        :return: True if the key existed and was removed, False otherwise.
        """
        if key not in self._cache_map:
            return False
        
        node = self._cache_map[key]
        self._remove_node(node)
        del self._cache_map[key]
        return True

    def size(self) -> int:
        """
        Return the count of non-expired items.
        Performs lazy cleanup of expired items during the count.
        
        :return: Number of valid items currently in the cache.
        """
        current_time = monotonic()
        count = 0
        # We iterate through the map to check expiry. 
        # Note: Iterating a dict while modifying it is unsafe in Python.
        # We must collect keys to delete first.
        keys_to_delete = []
        
        for key, node in self._cache_map.items():
            if node.expiry > current_time:
                count += 1
            else:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            node = self._cache_map[key]
            self._remove_node(node)
            del self._cache_map[key]
            
        return count

import pytest
from unittest.mock import patch, MagicMock
from typing import Any


# Helper to create a list of time values for mocking
def mock_time_sequence(*times):
    """Returns a generator that yields the specified times in order."""
    for t in times:
        yield t

@pytest.fixture
def cache():
    return TTLCache(capacity=3, default_ttl=10.0)

@pytest.fixture
def mock_time():
    """Fixture to patch time.monotonic."""
    with patch('ttl_cache.monotonic') as mock_monotonic:
        # Default return value if not specified in test
        mock_monotonic.return_value = 0.0
        yield mock_monotonic

class TestBasicOperations:
    def test_basic_put_and_get(self, cache, mock_time):
        # Set time to 10.0
        mock_time.return_value = 10.0
        cache.put("key1", "value1")
        
        # Get should work
        assert cache.get("key1") == "value1"
        
        # Non-existent key
        assert cache.get("nonexistent") is None

    def test_put_overwrites_value(self, cache, mock_time):
        mock_time.return_value = 10.0
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        
        assert cache.get("key1") == "value2"

class TestCapacityEviction:
    def test_lru_eviction_order(self, cache, mock_time):
        # Fill cache to capacity (3)
        mock_time.return_value = 10.0
        cache.put("a", "val_a")
        cache.put("b", "val_b")
        cache.put("c", "val_c")
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'd'. Should evict 'b' (LRU)
        mock_time.return_value = 11.0
        cache.put("d", "val_d")
        
        assert cache.get("a") == "val_a"
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == "val_c"
        assert cache.get("d") == "val_d"

class TestTTLExpiry:
    def test_item_expires_after_ttl(self, cache, mock_time):
        # Time 0: Put item with default TTL (10s)
        mock_time.return_value = 0.0
        cache.put("key", "val")
        
        # Time 5: Still valid
        mock_time.return_value = 5.0
        assert cache.get("key") == "val"
        
        # Time 10: Expired (0 + 10 = 10, expiry <= current)
        mock_time.return_value = 10.0
        assert cache.get("key") is None

    def test_custom_ttl(self, cache, mock_time):
        mock_time.return_value = 0.0
        # Custom TTL of 5 seconds
        cache.put("key", "val", ttl=5.0)
        
        mock_time.return_value = 4.0
        assert cache.get("key") == "val"
        
        mock_time.return_value = 5.0
        assert cache.get("key") is None

    def test_eviction_of_expired_items(self, cache, mock_time):
        # Fill cache with items that expire quickly
        mock_time.return_value = 0.0
        cache.put("a", "val_a", ttl=1.0)
        cache.put("b", "val_b", ttl=1.0)
        cache.put("c", "val_c", ttl=1.0)
        
        # Advance time past expiry
        mock_time.return_value = 2.0
        
        # Accessing any key should trigger cleanup or eviction logic
        # Since all are expired, adding a new one should clear them first
        cache.put("d", "val_d")
        
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None
        assert cache.get("d") == "val_d"

class TestDelete:
    def test_delete_existing_key(self, cache, mock_time):
        mock_time.return_value = 0.0
        cache.put("key", "val")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_non_existing_key(self, cache, mock_time):
        mock_time.return_value = 0.0
        assert cache.delete("nonexistent") is False

class TestSize:
    def test_size_with_mixed_items(self, cache, mock_time):
        mock_time.return_value = 0.0
        cache.put("a", "val_a", ttl=10.0)
        cache.put("b", "val_b", ttl=1.0)
        cache.put("c", "val_c", ttl=10.0)
        
        # Size should be 3
        assert cache.size() == 3
        
        # Advance time to expire 'b'
        mock_time.return_value = 2.0
        
        # Size should be 2 (a and c)
        assert cache.size() == 2
        
        # Access 'a' to ensure it stays valid
        cache.get("a")
        assert cache.size() == 2

    def test_size_clears_expired(self, cache, mock_time):
        mock_time.return_value = 0.0
        cache.put("a", "val_a", ttl=1.0)
        cache.put("b", "val_b", ttl=1.0)
        
        mock_time.return_value = 2.0
        
        # Size should be 0 after cleanup
        assert cache.size() == 0