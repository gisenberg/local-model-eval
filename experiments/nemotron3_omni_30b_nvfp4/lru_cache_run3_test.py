import time
from typing import Any, Optional


class Node:
    def __init__(self, key: Any, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.expires_at = time.monotonic() + ttl if ttl is not None else None
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class TTLCache:
    def __init__(self, capacity: int, default_ttl: Optional[float] = None):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.map: dict[Any, Node] = {}
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None

    def _remove_node(self, node: Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def _add_node(self, node: Node) -> None:
        node.prev = self.tail
        node.next = None

        if self.tail:
            self.tail.next = node
        self.tail = node

        if self.head is None:
            self.head = node

    def _is_expired(self, node: Node) -> bool:
        return node.expires_at is not None and time.monotonic() > node.expires_at

    def _cleanup_expired(self) -> None:
        current = self.head
        while current:
            if self._is_expired(current):
                self._remove_node(current)
                next_node = current.next
                del self.map[current.key]
                current = next_node
            else:
                current = current.next

    def get(self, key: Any) -> Any:
        self._cleanup_expired()
        if key not in self.map:
            return -1

        node = self.map[key]
        self._remove_node(node)
        self._add_node(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        self._cleanup_expired()

        if key in self.map:
            node = self.map[key]
            node.value = value
            node.ttl = ttl
            node.expires_at = time.monotonic() + ttl if ttl is not None else None
            self._remove_node(node)
            self._add_node(node)
            return

        if len(self.map) >= self.capacity:
            self._remove_node(self.head)
            del self.map[self.head.key]
            self.head = self.head.next

        node = Node(key, value, ttl)
        self.map[key] = node
        self._add_node(node)

    def delete(self, key: Any) -> None:
        self._cleanup_expired()
        if key not in self.map:
            return

        node = self.map[key]
        self._remove_node(node)
        del self.map[key]

    def size(self) -> int:
        self._cleanup_expired()
        return len(self.map)


# pytest tests
import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_get_expired_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 1.0)
    cache.put('key', 'value')
    assert cache.get('key') == -1


@patch('time.monotonic')
def test_put_expired_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 1.0)
    cache.put('key', 'value')
    assert cache.get('key') == -1


@patch('time.monotonic')
def test_put_update_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 1.0)
    cache.put('key', 'value')
    cache.put('key', 'new_value')
    assert cache.get('key') == 'new_value'


@patch('time.monotonic')
def test_delete_node(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 1.0)
    cache.put('key', 'value')
    cache.delete('key')
    assert cache.get('key') == -1


@patch('time.monotonic')
def test_size_and_capacity(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(2, 1.0)
    assert cache.size() == 0
    cache.put('key1', 'value1')
    assert cache.size() == 1
    cache.put('key2', 'value2')
    assert cache.size() == 2
    cache.put('key3', 'value3')
    assert cache.size() == 2


@patch('time.monotonic')
def test_ttl_parameter(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 2.0)
    cache.put('key', 'value', ttl=1.0)
    assert cache.get('key') == 'value'
    assert cache.size() == 1
    assert time.monotonic() - 10.0 < 1.0
    assert cache.get('key') == -1