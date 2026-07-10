# ttl_cache.py
import time
from typing import Optional, Any, Dict


class Node:
    """Doubly-linked list node for TTLCache."""
    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: Any, value: Any, expiry: float) -> None:
        self.key = key
        self.value = value
        self.expiry = expiry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class TTLCache:
    """
    LRU Cache with per-entry Time-To-Live (TTL).

    Features:
    - O(1) average-time operations: get, put, delete, size.
    - Lazy expiration: Expired entries are evicted only when accessed.
    - Monotonic clock (time.monotonic()) used for expiration.

    Example:
        cache = TTLCache(capacity=2, default_ttl=10)
        cache.put("a", 1)  # uses default_ttl
        cache.put("b", 2, ttl=5)  # overrides to 5 seconds
        val = cache.get("a")  # returns 1 if not expired
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        """
        Initialize TTLCache.

        Args:
            capacity: Maximum number of entries. Must be ≥ 1.
            default_ttl: Default TTL in seconds for entries without explicit ttl.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        if default_ttl <= 0:
            raise ValueError("Default TTL must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self._map: Dict[Any, Node] = {}
        # Sentinel nodes for doubly-linked list
        self._head = Node(None, None, float('inf'))  # Dummy head (MRU)
        self._tail = Node(None, None, 0.0)          # Dummy tail (LRU)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: Node) -> None:
        """Remove node from list (in O(1))."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node
        node.prev = node.next = None

    def _append(self, node: Node) -> None:
        """Append node right before tail (LRU position)."""
        last = self._tail.prev
        if last:
            last.next = node
            node.prev = last
        self._tail.prev = node
        node.next = self._tail

    def _prepend(self, node: Node) -> None:
        """Prepend node right after head (MRU position)."""
        first = self._head.next
        if first:
            first.prev = node
        self._head.next = node
        node.prev = self._head
        node.next = first

    def _expire_one(self) -> Optional[Node]:
        """
        Evict the *least recently used expired* node (closest to tail).
        Returns evicted node (if any), else None.

        Must be called only when list is non-empty and head != tail.
        """
        node = self._tail.prev
        # Skip over empty/sentinel (though sentinel shouldn't expire)
        while node != self._head:
            if node.expiry <= time.monotonic():
                self._remove(node)
                return node
            node = node.prev
        return None

    def _evict_if_needed(self) -> None:
        """Evict expired nodes until cache size ≤ capacity."""
        while len(self._map) > self.capacity:
            node = self._tail.prev
            if node == self._head:
                break  # should not happen if logic is correct
            # Prefer evicting expired ones first
            if node.expiry <= time.monotonic():
                self._remove(node)
            else:
                # Evict LRU (whether expired or not — fallback)
                self._remove(node)
            # Remove from map
            if node.key in self._map:
                del self._map[node.key]

    def get(self, key: Any) -> Optional[Any]:
        """
        Get value for key. Returns None if key not found or expired.

        Moves accessed node to MRU position.
        Evicts expired nodes if needed.

        Args:
            key: Key to retrieve.

        Returns:
            Value if found and not expired, else None.
        """
        if key not in self._map:
            return None

        node = self._map[key]
        # Check expiry first
        if node.expiry <= time.monotonic():
            # Expired — remove and return None
            self._remove(node)
            del self._map[key]
            return None

        # Move to MRU position (right after head)
        self._remove(node)
        self._prepend(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Insert or update key-value pair.

        If key exists, updates value and TTL, moves to MRU.
        If capacity exceeded, evicts expired or LRU node(s) until fits.

        Args:
            key: Key to insert/update.
            value: Value to store.
            ttl: Optional TTL in seconds. If None, uses default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.monotonic() + ttl

        if key in self._map:
            node = self._map[key]
            # Update value & expiry
            node.value = value
            node.expiry = expiry
            # Move to MRU
            self._remove(node)
            self._prepend(node)
        else:
            node = Node(key, value, expiry)
            self._map[key] = node
            self._prepend(node)

            # Check size & evict expired/LRU if over capacity
            if len(self._map) > self.capacity:
                self._evict_if_needed()

    def delete(self, key: Any) -> bool:
        """
        Delete key. Returns True if key existed, False otherwise.

        Args:
            key: Key to delete.

        Returns:
            True if deleted, False if not found.
        """
        if key not in self._map:
            return False

        node = self._map.pop(key)
        self._remove(node)
        return True

    def size(self) -> int:
        """
        Return current number of non-expired entries.

        Lazy cleanup: removes expired entries before counting.
        """
        # Clean up expired entries (lazy)
        node = self._tail.prev
        while node != self._head:
            if node.expiry <= time.monotonic():
                self._remove(node)
                if node.key in self._map:
                    del self._map[node.key]
                node = self._tail.prev  # restart after removal
            else:
                node = node.prev

        return len(self._map)

# test_ttl_cache.py
import pytest
from unittest.mock import patch


def patch_time(monkeypatch, current_time):
    """Helper to patch time.monotonic() globally."""
    monkeypatch.setattr("ttl_cache.time.monotonic", lambda: current_time)


def test_basic_put_get(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.size() == 1


def test_ttl_expires_and_evicts(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=3, default_ttl=1.0)
    cache.put("a", 1)  # expires at t=1.0
    cache.put("b", 2)  # expires at t=1.0

    # At t=0.5: both alive
    with patch("ttl_cache.time.monotonic", return_value=0.5):
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.size() == 2

    # At t=1.5: both expired
    with patch("ttl_cache.time.monotonic", return_value=1.5):
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size() == 0  # lazy cleanup


def test_put_updates_ttl_and_position(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=2, default_ttl=5.0)
    cache.put("a", 1)
    cache.put("b", 2)

    # At t=1.0: a is LRU (since b just put)
    with patch("ttl_cache.time.monotonic", return_value=1.0):
        # Reinsert a with new TTL=10 → now MRU, expires at t=11
        cache.put("a", 100, ttl=10.0)

    # At t=2.0
    with patch("ttl_cache.time.monotonic", return_value=2.0):
        assert cache.get("a") == 100
        assert cache.get("b") == 2  # still valid (expires at t=6)
        # a is MRU now, b is LRU

    # At t=7.0: b expired (originally set at t=0), a still alive (expires t=11)
    with patch("ttl_cache.time.monotonic", return_value=7.0):
        assert cache.get("a") == 100
        assert cache.get("b") is None  # expired
        assert cache.size() == 1


def test_capacity_evicts_LRU_on_overflow(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=2, default_ttl=100.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" (LRU) before "b"

    with patch("ttl_cache.time.monotonic", return_value=1.0):
        assert cache.get("a") is None  # evicted
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 2


def test_delete_works_and_maintains_size(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=2, default_ttl=10.0)
    cache.put("a", 1)
    cache.put("b", 2)

    # At t=0.5
    with patch("ttl_cache.time.monotonic", return_value=0.5):
        assert cache.delete("a") is True
        assert cache.size() == 1
        assert cache.get("a") is None
        assert cache.delete("a") is False  # not found


def test_ttl_none_uses_default_ttl(monkeypatch):
    patch_time(monkeypatch, 0.0)
    cache = TTLCache(capacity=2, default_ttl=2.0)
    cache.put("a", 1)  # ttl = default = 2.0
    cache.put("b", 2, ttl=1.0)

    # At t=0.5
    with patch("ttl_cache.time.monotonic", return_value=0.5):
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    # At t=1.5: b expired, a alive
    with patch("ttl_cache.time.monotonic", return_value=1.5):
        assert cache.get("b") is None
        assert cache.get("a") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])