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

    def _add_node(self, node: Node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: Node):
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Node:
        res = self.tail.prev
        self._remove_node(res)
        return res

    def _cleanup(self):
        current_time = time.monotonic()
        node = self.head.next
        while node != self.tail:
            if node.ttl is not None and node.ttl < current_time:
                self._remove_node(node)
                next_node = node.next
                del self.cache[node.key]
                self.size -= 1
                node = next_node
            else:
                node = node.next

    def get(self, key: Any) -> Any:
        self._cleanup()
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None):
        self._cleanup()
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.ttl = ttl if ttl is not None else self.default_ttl
            self._move_to_head(node)
        else:
            if self.size == self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
            node = Node(key, value, ttl if ttl is not None else self.default_ttl)
            self.cache[key] = node
            self._add_node(node)
            self.size += 1

    def delete(self, key: Any):
        self._cleanup()
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1

    def size(self) -> int:
        self._cleanup()
        return self.size

import pytest
from unittest.mock import patch

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_get_and_put(mock_time):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_delete(mock_time):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    cache.delete('a')
    assert cache.get('a') == -1
    assert cache.size() == 0

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_size_and_capacity(mock_time):
    cache = TTLCache(2, 1)
    assert cache.size() == 0
    cache.put('a', 1)
    assert cache.size() == 1
    cache.put('b', 2)
    assert cache.size() == 2
    cache.put('c', 3)
    assert cache.size() == 2

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_ttl_and_default_ttl(mock_time):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    assert cache.get('a') == 1
    assert cache.size() == 1
    time.sleep(1.1)
    assert cache.get('a') == -1
    assert cache.size() == 0

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_multiple_keys_and_ttl(mock_time):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    cache.put('b', 2)
    assert cache.get('a') == 1
    assert cache.get('b') == 2
    assert cache.size() == 2
    time.sleep(1.1)
    assert cache.get('a') == -1
    assert cache.get('b') == 2
    assert cache.size() == 1

@patch('time.monotonic', side_effect=[1000, 1001, 1002, 1003, 1004, 1005])
def test_ttl_cache_delete_and_size(mock_time):
    cache = TTLCache(2, 1)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.delete('a')
    assert cache.get('a') == -1
    assert cache.size() == 1
    cache.delete('b')
    assert cache.size() == 0