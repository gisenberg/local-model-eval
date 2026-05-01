import time
from typing import Any, Dict, Optional


class Node:
    def __init__(self, key: Any, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None


class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node(None, None)  # Dummy head
        self.tail = Node(None, None)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _add_node(self, node: Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: Node) -> None:
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Node:
        res = self.tail.prev
        self._remove_node(res)
        return res

    def _is_expired(self, node: Node) -> bool:
        return node.ttl is not None and time.monotonic() > node.ttl

    def get(self, key: Any) -> Optional[Any]:
        node = self.cache.get(key)
        if not node:
            return None
        if self._is_expired(node):
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        node = self.cache.get(key)
        if node:
            node.value = value
            node.ttl = ttl
            self._move_to_head(node)
        else:
            new_node = Node(key, value, ttl)
            self.cache[key] = new_node
            self._add_node(new_node)
            self.size += 1
            if self.size > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1

    def delete(self, key: Any) -> None:
        node = self.cache.get(key)
        if node:
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1

    def size(self) -> int:
        return self.size


# pytest tests
import pytest
from unittest.mock import patch


@patch('time.monotonic')
def test_get_expired(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 5.0)
    assert cache.get('key') is None


@patch('time.monotonic')
def test_put_and_get(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value')
    assert cache.get('key') == 'value'


@patch('time.monotonic')
def test_delete(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value')
    cache.delete('key')
    assert cache.get('key') is None


@patch('time.monotonic')
def test_size_and_capacity(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(2, 5.0)
    assert cache.size() == 0
    cache.put('key1', 'value1')
    assert cache.size() == 1
    cache.put('key2', 'value2')
    assert cache.size() == 2
    cache.put('key3', 'value3')
    assert cache.size() == 2


@patch('time.monotonic')
def test_ttl_override(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(1, 5.0)
    cache.put('key', 'value', 3.0)
    assert cache.get('key') == 'value'
    mock_monotonic.side_effect = [10.0, 10.0]
    assert cache.get('key') is None


@patch('time.monotonic')
def test_lru_eviction(mock_monotonic):
    mock_monotonic.return_value = 10.0
    cache = TTLCache(2, 5.0)
    cache.put('key1', 'value1')
    cache.put('key2', 'value2')
    cache.put('key3', 'value3')
    assert cache.get('key1') is None
    assert cache.get('key2') == 'value2'