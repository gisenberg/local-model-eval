import time
from typing import Any, Dict, Optional, Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class _Node:
    """A node of the doubly‑linked list used by :class:`TTLCache`."""

    __slots__ = ('key', 'value', 'expiry', 'prev', 'next')

    def __init__(self, key: K, value: V, expiry: float) -> None:
        self.key: K = key
        self.value: V = value
        self.expiry: float = expiry          # time.monotonic() timestamp when the entry expires
        self.prev: Optional['_Node'] = None
        self.next: Optional['_Node'] = None


class TTLCache(Generic[K, V]):
    """LRU cache with per‑item TTL.

    * ``capacity`` – maximum number of entries that can be stored.
    * ``default_ttl`` – default time‑to‑live in seconds for items created without an explicit ``ttl``.
    * All public operations (`get`, `put`, `delete`, `size`) run in O(1) average time.
    * Expired entries are removed lazily – only when they are accessed.
    """

    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non‑negative")
        self.capacity: int = capacity
        self.default_ttl: float = default_ttl
        self._map: Dict[K, _Node] = {}
        self._size: int = 0

        # sentinel nodes simplify edge‑case handling
        self._head = _Node(None, None, 0.0)   # most recent
        self._tail = _Node(None, None, 0.0)   # least recent
        self._head.next = self._tail
        self._tail.prev = self._head

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _time(self) -> float:
        """Return the current monotonic timestamp."""
        return time.monotonic()

    def _expire(self, node: _Node) -> bool:
        """Remove *node* if it is expired.

        Returns ``True`` if the node was removed, ``False`` otherwise.
        """
        if node.expiry <= self._time():
            self._remove_node(node)
            del self._map[node.key]
            self._size -= 1
            return True
        return False

    def _add_to_head(self, node: _Node) -> None:
        """Insert *node* right after the head (the most‑recent position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Detach *node* from the linked list."""
        prev = node.prev  # type: ignore[assignment]
        nxt = node.next   # type: ignore[assignment]
        prev.next = nxt
        nxt.prev = prev

    def _move_to_head(self, node: _Node) -> None:
        """Mark *node* as the most recently used entry."""
        self._remove_node(node)
        self._add_to_head(node)

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def get(self, key: K) -> Optional[V]:
        """Return the value for *key* if it exists and is not expired.

        The accessed entry becomes the most recently used.  If the key is
        missing or expired, ``None`` is returned and the entry is removed
        lazily.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if self._expire(node):
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Insert or update *key* with *value*.

        If *ttl* is ``None`` the ``default_ttl`` is used.  The entry is
        considered fresh for the given TTL; if the TTL has already passed
        at insertion time the entry is ignored (it will be removed on the
        next access).
        """
        if self.capacity <= 0:
            return

        expiry = self._time() + (ttl if ttl is not None else self.default_ttl)

        if key in self._map:
            node = self._map[key]
            node.value = value
            node.expiry = expiry
            self._move_to_head(node)
        else:
            if self._size >= self.capacity:
                # evict the least‑recently used entry
                lru = self._tail.prev
                if lru is not self._head:
                    self._remove_node(lru)
                    del self._map[lru.key]
                    self._size -= 1
            node = _Node(key, value, expiry)
            self._map[key] = node
            self._add_to_head(node)
            self._size += 1

    def delete(self, key: K) -> bool:
        """Remove *key* from the cache.

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        node = self._map.pop(key, None)
        if node:
            self._remove_node(node)
            self._size -= 1
            return True
        return False

    def size(self) -> int:
        """Return the number of entries currently stored in the cache.

        The count includes entries that may be expired; they are removed
        lazily when accessed.
        """
        return self._size

import pytest
from unittest.mock import patch



def test_basic_put_get():
    """Basic put/get functionality."""
    with patch('time.monotonic', return_value=10.0):
        cache = TTLCache(capacity=3, default_ttl=100.0)
        cache.put('k1', 'val1')
        assert cache.get('k1') == 'val1'
        assert cache.size() == 1


def test_ttl_expiration():
    """Entry should disappear after its TTL."""
    with patch('time.monotonic', side_effect=[0, 3]):   # 0 for put, 3 for get
        cache = TTLCache(capacity=3, default_ttl=2.0)
        cache.put('k1', 'val1', ttl=2)   # expires at t = 0 + 2 = 2
        # time now 3 → entry is expired
        assert cache.get('k1') is None
        assert cache.size() == 0


def test_lru_eviction():
    """When capacity is exceeded, the least‑recently used entry is evicted."""
    with patch('time.monotonic', side_effect=[0, 0, 1]):
        cache = TTLCache(capacity=2, default_ttl=10.0)
        cache.put('a', 1)   # t=0
        cache.put('b', 2)   # t=0
        # third insertion forces eviction of 'a' (LRU)
        cache.put('c', 3)   # t=1
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.size() == 2


def test_update_key_moves_to_head():
    """Updating a key should make it the most recently used."""
    with patch('time.monotonic', side_effect=[0, 0, 1]):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put('a', 1)          # t=0
        assert cache.get('a') == 1
        cache.put('a', 2)          # update, t=1 → becomes most recent
        assert cache.get('a') == 2
        assert cache.size() == 2


def test_delete():
    """Delete removes a key and updates size."""
    with patch('time.monotonic', return_value=5.0):
        cache = TTLCache(capacity=3, default_ttl=10.0)
        cache.put('x', 'y')
        assert cache.get('x') == 'y'
        assert cache.size() == 1
        assert cache.delete('x') is True
        assert cache.get('x') is None
        assert cache.size() == 0


def test_capacity_zero():
    """Cache with zero capacity never stores items."""
    with patch('time.monotonic', return_value=10.0):
        cache = TTLCache(capacity=0, default_ttl=5.0)
        cache.put('k', 'v')
        assert cache.get('k') is None
        assert cache.size() == 0