import time
from typing import Optional, Dict

class Node:
    def __init__(self, key: int, value: int, expire_time: float):
        self.key = key
        self.value = value
        self.expire_time = expire_time
        self.prev = None
        self.next = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[int, Node] = {}
        self.head = Node(0, 0, 0)
        self.tail = Node(0, 0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node: Node) -> None:
        last_node = self.tail.prev
        last_node.next = node
        node.prev = last_node
        node.next = self.tail
        self.tail.prev = node

    def _move_to_head(self, node: Node) -> None:
        self._remove(node)
        self._add(node)

    def _is_expired(self, node: Node) -> bool:
        return time.monotonic() >= node.expire_time

    def get(self, key: int) -> Optional[int]:
        if key not in self.cache:
            return None
        node = self.cache[key]
        if self._is_expired(node):
            del self.cache[key]
            self._remove(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int, ttl: Optional[float] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_time = time.monotonic() + ttl
            self._move_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                lru_node = self.head.next
                if self._is_expired(lru_node):
                    del self.cache[lru_node.key]
                self._remove(lru_node)
                del self.cache[lru_node.key]
            new_node = Node(key, value, time.monotonic() + ttl)
            self.cache[key] = new_node
            self._add(new_node)

    def delete(self, key: int) -> None:
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        return len(self.cache)

# Tests
from unittest.mock import patch
import pytest

@patch('time.monotonic')
def test_get(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 5)
    cache.put(1, 10)
    assert cache.get(1) == 10

@patch('time.monotonic')
def test_ttl_expiration(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 5)
    cache.put(1, 10)
    mock_time.return_value = 106
    assert cache.get(1) is None

@patch('time.monotonic')
def test_lru_eviction(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 10)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.get(1)
    cache.put(3, 30)
    assert cache.get(2) is None

@patch('time.monotonic')
def test_delete(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 5)
    cache.put(1, 10)
    cache.delete(1)
    assert cache.get(1) is None

@patch('time.monotonic')
def test_size(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 5)
    cache.put(1, 10)
    cache.put(2, 20)
    assert cache.size() == 2

@patch('time.monotonic')
def test_update_existing_key(mock_time):
    mock_time.return_value = 100
    cache = TTLCache(2, 5)
    cache.put(1, 10)
    cache.put(1, 20)
    assert cache.get(1) == 20