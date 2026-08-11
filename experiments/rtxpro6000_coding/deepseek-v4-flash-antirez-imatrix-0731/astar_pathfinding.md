```python
from typing import List, Tuple, Optional, Dict
import heapq

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    Grid values: 0 = impassable wall, positive int = cost to enter that cell.
    Movement is 4-directional (up, down, left, right), no diagonals.
    Heuristic: Manhattan distance.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list of integers. 0 represents a wall, positive integers
                  represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if (row, col) is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, row: int, col: int) -> bool:
        """Check if the cell is a wall (cost 0)."""
        return self.grid[row][col] == 0

    def _heuristic(self, row: int, col: int, end: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        end_row, end_col = end
        return abs(row - end_row) + abs(col - end_col)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: (row, col) of starting cell.
            end: (row, col) of target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        start_row, start_col = start
        end_row, end_col = end

        # Out-of-bounds check
        if not self._is_valid(start_row, start_col) or not self._is_valid(end_row, end_col):
            raise ValueError("Start or end is out of bounds")

        # Wall check
        if self._is_wall(start_row, start_col) or self._is_wall(end_row, end_col):
            return None

        # Start equals end
        if start == end:
            return [start]

        # A* algorithm
        open_heap = []  # (f_score, g_score, row, col)
        heapq.heappush(open_heap, (self._heuristic(start_row, start_col, end), 0, start_row, start_col))

        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap:
            f, g, row, col = heapq.heappop(open_heap)
            current = (row, col)

            # Skip if we already found a better path to this node
            if g > g_score.get(current, float('inf')):
                continue

            # Check if we reached the end
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                neighbor = (nr, nc)

                if not self._is_valid(nr, nc):
                    continue
                if self._is_wall(nr, nc):
                    continue

                # Movement cost = cost of entering the neighbor cell
                tentative_g = g + self.grid[nr][nc]

                # If this path is better than previously recorded
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(nr, nc, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, nr, nc))

        # No path found
        return None


# ---------- Tests ----------

import pytest

def path_cost(grid, path):
    """Compute total cost of a path (excluding start cell)."""
    if not path:
        return None
    total = 0
    for r, c in path[1:]:  # skip start
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Optimal cost: Manhattan distance = 4 steps, each cost 1 → total 4
    assert path_cost(grid, path) == 4
    # Path should be valid (all cells traversable)
    for r, c in path:
        assert grid[r][c] != 0

def test_path_around_obstacles():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Must go around the wall: e.g., (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # Cost = 1+1+1+1+1+1 = 6 (excluding start)
    assert path_cost(grid, path) == 6
    # Verify no wall cells in path
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid_prefers_lower_cost():
    grid = [
        [1, 10, 10, 10],
        [1, 1, 1, 10],
        [10, 10, 1, 10],
        [10, 10, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (3, 3)
    path = astar.find_path(start, end)
    assert path is not None
    # Optimal path should go through low-cost cells (1s) rather than high-cost (10s)
    # Expected path: (0,0)->(1,0)->(1,1)->(1,2)->(2,2)->(3,2)->(3,3)
    # Cost = 1+1+1+1+1+1 = 6 (excluding start)
    assert path_cost(grid, path) == 6
    # Verify path uses only cost-1 cells (except maybe start)
    for r, c in path[1:]:
        assert grid[r][c] == 1

def test_no_path_fully_blocked():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is None

def test_start_equals_end():
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    start = end = (1, 1)
    path = astar.find_path(start, end)
    assert path == [start]

def test_invalid_coordinates():
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))
    # Wall start/end
    grid_wall = [
        [0, 1],
        [1, 1]
    ]
    astar_wall = AStarGrid(grid_wall)
    assert astar_wall.find_path((0, 0), (1, 1)) is None
    assert astar_wall.find_path((1, 1), (0, 0)) is None
```