import time
from typing import Optional, Any, Dict, List

class Node:
    def __init__(self, key: Any = None, value: Any = None, expire_at: float = 0):
        self.key = key
        self.value = value
        self.expire_at = expire_at
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Node] = {}
        self.head = Node()  # dummy head
        self.tail = Node()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node: Node) -> None:
        last = self.tail.prev
        last.next = node
        node.prev = last
        node.next = self.tail
        self.tail.prev = node

    def _move_to_head(self, node: Node) -> None:
        self._remove(node)
        self._add(node)

    def _pop_tail(self) -> Node:
        node = self.tail.prev
        self._remove(node)
        return node

    def _is_expired(self, node: Node) -> bool:
        return node.expire_at <= time.monotonic()

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return None
        node = self.cache[key]
        if self._is_expired(node):
            del self.cache[key]
            self._remove(node)
            return None
        self._move_to_head(node)
        return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        expire_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            node.expire_at = expire_at
            self._move_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]
            new_node = Node(key, value, expire_at)
            self.cache[key] = new_node
            self._add(new_node)

    def delete(self, key: Any) -> None:
        if key in self.cache:
            node = self.cache.pop(key)
            self._remove(node)

    def size(self) -> int:
        return len(self.cache)

# Tests
import pytest
from unittest.mock import patch

@pytest.fixture
def cache():
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 100.0
        yield TTLCache(2, 5.0)

def test_get(cache):
    cache.put("key", "value")
    assert cache.get("key") == "value"

def test_get_expired(cache):
    cache.put("key", "value", ttl=1)
    with patch('time.monotonic', return_value=102.0):
        assert cache.get("key") is None

def test_put_eviction(cache):
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_delete(cache):
    cache.put("key", "value")
    cache.delete("key")
    assert cache.get("key") is None

def test_size(cache):
    assert cache.size() == 0
    cache.put("a", 1)
    assert cache.size() == 1
    cache.put("b", 2)
    assert cache.size() == 2
    cache.delete("a")
    assert cache.size() == 1

def test_update_existing_key(cache):
    cache.put("key", "value1")
    cache.put("key", "value2")
    assert cache.get("key") == "value2"