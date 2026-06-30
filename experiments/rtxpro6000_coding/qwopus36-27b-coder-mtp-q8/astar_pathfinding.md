```python
import heapq
from typing import List, Optional, Tuple

class AStarGrid:
    """A* pathfinding on a weighted 2D grid with 4-directional movement."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list where each value represents the movement cost to enter
                  that cell. 0 means impassable wall, positive int means cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a cell is within bounds and not a wall.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if the cell is valid (in bounds and not a wall).
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] > 0

    def _heuristic(self, row: int, col: int, end_row: int, end_col: int) -> int:
        """Manhattan distance heuristic.

        Args:
            row: Current row.
            col: Current column.
            end_row: Target row.
            end_col: Target column.

        Returns:
            Manhattan distance to the target.
        """
        return abs(row - end_row) + abs(col - end_col)

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A*.

        Args:
            start: Starting (row, col) coordinate.
            end: Ending (row, col) coordinate.

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        start_row, start_col = start
        end_row, end_col = end

        # Check bounds
        if (
            start_row < 0
            or start_row >= self.rows
            or start_col < 0
            or start_col >= self.cols
        ):
            raise ValueError(f"Start coordinate {start} is out of bounds")
        if (
            end_row < 0
            or end_row >= self.rows
            or end_col < 0
            or end_col >= self.cols
        ):
            raise ValueError(f"End coordinate {end} is out of bounds")

        # Check if start or end is a wall
        if self.grid[start_row][start_col] == 0:
            return None
        if self.grid[end_row][end_col] == 0:
            return None

        # Start equals end
        if start == end:
            return [start]

        # A* algorithm
        # Priority queue: (f_score, counter, row, col)
        # counter is used to break ties and avoid comparing tuples
        counter = 0
        open_set: List[Tuple[int, int, int, int]] = []
        heapq.heappush(
            open_set,
            (
                self._heuristic(start_row, start_col, end_row, end_col),
                counter,
                start_row,
                start_col,
            ),
        )
        counter += 1

        # g_score: cost from start to current cell
        g_score: dict = {start: 0}

        # came_from: for reconstructing the path
        came_from: dict = {}

        # closed set
        closed_set: set = set()

        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            f, _, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)

            if current in closed_set:
                continue

            # Found the end
            if current == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)

            for dr, dc in directions:
                neighbor_row = current_row + dr
                neighbor_col = current_col + dc
                neighbor = (neighbor_row, neighbor_col)

                if not self._is_valid(neighbor_row, neighbor_col):
                    continue

                if neighbor in closed_set:
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor_row][neighbor_col]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(
                        neighbor_row, neighbor_col, end_row, end_col
                    )
                    heapq.heappush(
                        open_set,
                        (f_score, counter, neighbor_row, neighbor_col),
                    )
                    counter += 1

        return None
```

```python
import pytest
from typing import List, Tuple

from a_star_grid import AStarGrid

def _total_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate the total cost of a path (cost to enter each cell except start)."""
    total = 0
    for i in range(1, len(path)):
        row, col = path[i]
        total += grid[row][col]
    return total

def _is_valid_path(grid: List[List[int]], path: List[Tuple[int, int]]) -> bool:
    """Check that the path is valid (adjacent cells, no walls, correct start/end)."""
    if not path:
        return False
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    for i, (r, c) in enumerate(path):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if grid[r][c] == 0:
            return False
        if i > 0:
            prev_r, prev_c = path[i - 1]
            if abs(r - prev_r) + abs(c - prev_c) != 1:
                return False
    return True

def test_simple_path_uniform_grid():
    """Test a simple path on a uniform grid with no obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    assert _is_valid_path(grid, path)
    # Optimal cost: 6 steps, each costing 1
    assert _total_cost(grid, path) == 6

def test_path_around_obstacles():
    """Test pathfinding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    assert _is_valid_path(grid, path)
    # Must go around the wall: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3)
    # or similar, cost = 6
    assert _total_cost(grid, path) == 6

def test_weighted_grid_prefers_lower_cost():
    """Test that A* prefers lower-cost cells when available."""
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    assert _is_valid_path(grid, path)
    # Optimal path goes around the high-cost cells:
    # (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3) cost = 6
    # or (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3) cost = 6
    # Not through the 10-cost cells
    assert _total_cost(grid, path) == 6

def test_no_path_exists():
    """Test that None is returned when no path exists."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test that start == end returns a single-element path."""
    grid = [
        [1, 1],
        [1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that ValueError is raised for out-of-bounds coordinates."""
    grid = [
        [1, 1],
        [1, 1],
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))
```