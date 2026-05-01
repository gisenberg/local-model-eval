import time
from typing import Any, Optional


class Node:
    def __init__(self, key: Any, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expires_at: Optional[float] = None
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: dict[Any, Node] = {}
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None

    def _update_node(self, node: Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        node.prev = None
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node

    def _remove_node(self, node: Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        node.prev = None
        node.next = None

    def _cleanup_expired(self) -> None:
        current_time = time.monotonic()
        node = self.tail
        while node:
            if node.expires_at is not None and node.expires_at <= current_time:
                self._remove_node(node)
                del self.cache[node.key]
            node = node.prev

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._update_node(node)
        self._cleanup_expired()
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl = ttl
            node.expires_at = None if ttl is None else time.monotonic() + ttl
            self._update_node(node)
            self._cleanup_expired()
            return

        if len(self.cache) >= self.capacity:
            self._remove_node(self.tail)
            del self.cache[self.tail.key]

        node = Node(key, value, ttl)
        node.expires_at = None if ttl is None else time.monotonic() + ttl
        self.cache[key] = node
        self._update_node(node)

    def delete(self, key: Any) -> None:
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def size(self) -> int:
        self._cleanup_expired()
        return len(self.cache)


# pytest tests
import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_get_expired_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    assert cache.get('key') == 'value'
    mock_monotonic.return_value = 15.0
    assert cache.get('key') == -1


@patch('time.monotonic')
def test_put_expired_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    assert cache.get('key') == 'value'
    mock_monotonic.return_value = 15.0
    cache.put('key', 'new_value', 5.0)
    assert cache.get('key') == 'new_value'


@patch('time.monotonic')
def test_delete_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    cache.delete('key')
    assert cache.get('key') == -1


@patch('time.monotonic')
def test_size_and_capacity(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(2, 5.0)
    assert cache.size() == 0
    cache.put('key1', 'value1', 5.0)
    assert cache.size() == 1
    cache.put('key2', 'value2', 5.0)
    assert cache.size() == 2
    cache.put('key3', 'value3', 5.0)
    assert cache.size() == 2


@patch('time.monotonic')
def test_update_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    assert cache.get('key') == 'value'
    mock_monotonic.return_value = 15.0
    cache.put('key', 'new_value', 5.0)
    assert cache.get('key') == 'new_value'


@patch('time.monotonic')
def test_lazy_cleanup(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    assert cache.get('key') == 'value'
    mock_monotonic.return_value = 15.0
    assert cache.get('key') == -1