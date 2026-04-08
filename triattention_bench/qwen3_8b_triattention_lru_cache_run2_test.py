import time
from typing import Optional, Any
from unittest.mock import patch
import unittest

class Node:
    def __init__(self, key: str, value: Any, expiration_time: float):
        self.key = key
        self.value = value
        self.expiration_time = expiration_time
        self.prev = None
        self.next = None


class TTLCache:
    def __init__(self, capacity: int, default_ttl: float):
        """
        Initialize the TTLCache with given capacity and default TTL.

        Args:
            capacity (int): Maximum number of items the cache can hold.
            default_ttl (float): Default time-to-live in seconds for items.
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}  # Maps keys to nodes
        self.head = None  # Head of the doubly linked list
        self.tail = None  # Tail of the doubly linked list
        self.size = 0     # Number of non-expired items in the cache

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for the given key if it exists and is not expired.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and is not expired, else None.
        """
        if key not in self.cache:
            return None

        node = self.cache[key]
        if node.expiration_time < time.monotonic():
            # Expired, remove from cache
            self._remove_node(node)
            del self.cache[key]
            self.size -= 1
            return None

        # Move to tail (most recently used)
        self._move_to_tail(node)
        return node.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Insert or update the value for the given key with the specified TTL.

        Args:
            key (str): The key to insert/update.
            value (Any): The value to store.
            ttl (Optional[float]): Time-to-live in seconds. Defaults to default_ttl.
        """
        if ttl is None:
            ttl = self.default_ttl
        expiration_time = time.monotonic() + ttl

        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            node.expiration_time = expiration_time
            self._move_to_tail(node)
            return

        # Add new node
        if self.size >= self.capacity:
            # Evict least recently used non-expired item
            # First, remove all expired nodes from the head
            while self.head and self.head.expiration_time < time.monotonic():
                self._remove_node(self.head)
                self.size -= 1
                if not self.head:
                    break

            # Now, evict the head (least recently used non-expired)
            if self.head:
                self._remove_node(self.head)
                self.size -= 1

        # Add new node
        new_node = Node(key, value, expiration_time)
        self.cache[key] = new_node
        self._add_to_tail(new_node)
        self.size += 1

    def delete(self, key: str) -> bool:
        """
        Delete the key from the cache if it exists.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key existed and was deleted, else False.
        """
        if key not in self.cache:
            return False

        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size -= 1
        return True

    def size(self) -> int:
        """
        Return the number of non-expired items in the cache.

        Returns:
            int: The number of non-expired items.
        """
        return self.size

    def _remove_node(self, node):
        """
        Remove a node from the linked list.

        Args:
            node (Node): The node to remove.
        """
        if node.prev:
            node.prev.next = node.next
        else:
            # Node is the head
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            # Node is the tail
            self.tail = node.prev

    def _move_to_tail(self, node):
        """
        Move a node to the tail of the linked list.

        Args:
            node (Node): The node to move.
        """
        self._remove_node(node)
        self._add_to_tail(node)

    def _add_to_tail(self, node):
        """
        Add a node to the tail of the linked list.

        Args:
            node (Node): The node to add.
        """
        if self.tail:
            self.tail.next = node
            node.prev = self.tail
        else:
            # List is empty
            self.head = node
        self.tail = node


---

### 🧪 **Unit Tests with Mocking**