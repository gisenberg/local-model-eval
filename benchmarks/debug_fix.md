# Benchmark: Debug and Fix

**Difficulty:** Medium
**Expected tests:** 4 (provided, not model-generated)
**Skills tested:** Code reading, bug identification, targeted fixes

## Prompt

```
The following Python code implements a binary search tree with insert, search, and in-order traversal. It has 3 bugs. Find and fix all bugs. Return ONLY the corrected code — do not explain the bugs.

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(self.root, value)

    def _insert(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(node.left, value)
        else:
            if node.left is None:  # Bug 1: should be node.right
                node.right = Node(value)
            else:
                self._insert(node.right, value)

    def search(self, value):
        return self._search(self.root, value)

    def _search(self, node, value):
        if node is None:
            return False
        if value == node.value:
            return True
        if value < node.value:
            return self._search(node.left, value)
        return self._search(node.left, value)  # Bug 2: should be node.right

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node.value)
        self._inorder(node.left, result)  # Bug 3: should be node.right
```

## Provided Test Suite

```python
import pytest

@pytest.fixture
def bst():
    tree = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        tree.insert(v)
    return tree

def test_search_existing(bst):
    assert bst.search(3) is True
    assert bst.search(7) is True
    assert bst.search(1) is True

def test_search_missing(bst):
    assert bst.search(2) is False
    assert bst.search(9) is False

def test_inorder(bst):
    assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]

def test_insert_and_search():
    tree = BST()
    tree.insert(10)
    tree.insert(5)
    tree.insert(15)
    assert tree.search(10) is True
    assert tree.search(5) is True
    assert tree.search(15) is True
    assert tree.search(20) is False
    assert tree.inorder() == [5, 10, 15]
```

## What Makes This a Good Benchmark

- Tests a completely different skill: reading and fixing code vs generating from scratch
- Requires understanding data structure traversal logic
- Three bugs of increasing subtlety (right-pointer check, search direction, traversal direction)
- Fast to generate — model output is short
