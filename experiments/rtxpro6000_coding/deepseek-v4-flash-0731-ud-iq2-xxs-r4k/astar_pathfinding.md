```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    Grid values: 0 = impassable wall, positive int = cost to enter that cell.
    Movement is 4-directional (up, down, left, right), no diagonals.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.

        Args:
            grid: 2D list of integers. 0 = wall, positive int = movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is a wall (cost 0)."""
        r, c = pos
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid, non-wall neighbors (4-directional)."""
        r, c = pos
        candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        neighbors = []
        for nr, nc in candidates:
            if self._is_in_bounds((nr, nc)) and not self._is_wall((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: (row, col) start coordinate.
            end: (row, col) end coordinate.

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Validate bounds
        if not self._is_in_bounds(start):
            raise ValueError(f"Start {start} is out of bounds")
        if not self._is_in_bounds(end):
            raise ValueError(f"End {end} is out of bounds")

        # Wall check
        if self._is_wall(start) or self._is_wall(end):
            return None

        # Start equals end
        if start == end:
            return [start]

        # A* initialization
        open_heap = []  # (f_score, tie_breaker, node)
        tie_breaker = 0
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        start_h = self._heuristic(start, end)
        heapq.heappush(open_heap, (start_h, tie_breaker, start))
        tie_breaker += 1

        while open_heap:
            f, _, current = heapq.heappop(open_heap)

            # Goal reached
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._neighbors(current):
                # tentative_g = g[current] + cost to enter neighbor
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                # If this path is better than any previous
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tie_breaker, neighbor))
                    tie_breaker += 1

        # No path found
        return None
```

```python
# tests/test_astar.py
import pytest
from astar import AStarGrid

def test_simple_path_uniform_grid():
    """Simple path on a uniform grid (all costs 1)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Verify path validity (each step moves 4-directionally)
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i-1][0])
        dc = abs(path[i][1] - path[i-1][1])
        assert dr + dc == 1
    # Optimal cost: Manhattan distance = 4 steps, each cost 1 → total 4
    total_cost = sum(grid[r][c] for r, c in path[1:])  # exclude start
    assert total_cost == 4

def test_path_around_obstacles():
    """Path must go around a wall."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (0, 3)
    path = astar.find_path(start, end)
    assert path is not None
    # Path should not include wall cells (1,1) or (1,2)
    for r, c in path:
        assert grid[r][c] != 0
    # Optimal cost: must go down and around, e.g., (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)->(1,3)->(0,3)
    # That's 7 steps, cost = 7 (all 1s)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 7

def test_weighted_grid_prefers_lower_cost():
    """Path should prefer lower-cost cells even if longer."""
    grid = [
        [1, 10, 10, 10],
        [1, 1, 1, 10],
        [10, 10, 1, 10],
        [10, 10, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (3, 3)
    path = astar.find_path(start, end)
    assert path is not None
    # Compute total cost (excluding start)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    # The optimal path should go through the cheap 1-cells:
    # (0,0)->(1,0)->(1,1)->(1,2)->(2,2)->(3,2)->(3,3)
    # Cost = 1+1+1+1+1+1 = 6
    assert total_cost == 6
    # Verify path is valid
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i-1][0])
        dc = abs(path[i][1] - path[i-1][1])
        assert dr + dc == 1

def test_no_path_exists_fully_blocked():
    """No path when start is enclosed by walls."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    # The wall at (1,1) blocks all paths? Actually there is a path around: (0,0)->(0,1)->(0,2)->(1,2)->(2,2)
    # So let's make it fully blocked:
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    assert path is None

def test_start_equals_end():
    """Start equals end returns [start]."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    start = (1, 1)
    end = (1, 1)
    path = astar.find_path(start, end)
    assert path == [start]

def test_invalid_coordinates():
    """Out-of-bounds raises ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
```