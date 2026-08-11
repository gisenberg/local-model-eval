```python
# astar.py
from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values:
        - 0: impassable wall
        - positive int: cost to enter that cell
    Movement is 4-directional (up, down, left, right), no diagonals.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list of integers, where 0 is a wall and positive values are movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic.

        Args:
            a: (row, col) coordinate
            b: (row, col) coordinate

        Returns:
            Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: (row, col) start coordinate
            end: (row, col) end coordinate

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Check bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End out of bounds")

        # Check walls
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Trivial case
        if start == end:
            return [start]

        # A* setup
        open_heap = []  # min-heap of (f, g, row, col)
        g_score = {start: 0}  # cost from start to node (including node cost)
        came_from = {}  # parent pointers for path reconstruction

        start_f = self.heuristic(start, end)
        heapq.heappush(open_heap, (start_f, 0, start[0], start[1]))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)
            current = (r, c)

            # Skip stale entries (we already found a better path to this node)
            if g_score.get(current, float('inf')) < g:
                continue

            # Goal reached
            if current == end:
                # Reconstruct path
                path = []
                node = end
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cost = self.grid[nr][nc]
                    if cost == 0:
                        continue
                    neighbor = (nr, nc)
                    tentative_g = g + cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, end)
                        heapq.heappush(open_heap, (f, tentative_g, nr, nc))

        return None
```

```python
# test_astar.py
import pytest
from astar import AStarGrid


def path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Total cost of a path (excluding the start cell)."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, path, start, end):
    """Check that the path is valid: adjacent moves, no walls, correct endpoints."""
    assert path[0] == start
    assert path[-1] == end
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] > 0


def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert path_cost(grid, path) == 4  # Manhattan distance = 4 steps, each cost 1


def test_path_around_obstacles():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert path_cost(grid, path) == 4


def test_weighted_grid():
    grid = [
        [1, 5, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert path_cost(grid, path) == 4
    # Optimal path avoids the high-cost cells
    for r, c in path[1:]:
        assert grid[r][c] == 1


def test_no_path():
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None


def test_start_equals_end():
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]


def test_invalid_coordinates():
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 1))
```