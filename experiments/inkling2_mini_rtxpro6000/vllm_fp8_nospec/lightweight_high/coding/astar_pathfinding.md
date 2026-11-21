Here's a complete implementation plus the test suite. Save the first block as `a_star.py` and the second as `test_a_star.py`.

**`a_star.py`**

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted 2D grid with 4-directional movement.

    Grid values are movement costs: 0 = impassable wall, positive int =
    cost to enter that cell. Path cost = sum of cell costs along the path,
    excluding the starting cell (you are already there).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Validate and store the grid.

        Raises:
            ValueError: if the grid is empty, ragged, or contains invalid values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must contain at least one row and one column")
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("All rows must have the same length")
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("Grid values must be non-negative integers")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = width
        self._directions: Tuple[Tuple[int, int], ...] = (
            (-1, 0), (1, 0), (0, -1), (0, 1)
        )

    def find_path(self, start: Cell, end: Cell) -> Optional[List[Cell]]:
        """Find the minimum-cost path from start to end.

        Returns the list of (row, col) coordinates from start to end
        inclusive, or None if no path exists. Wall endpoints return None
        (even if start == end). Out-of-bounds endpoints raise ValueError.
        """
        start = self._parse_cell(start)
        end = self._parse_cell(end)
        self._check_bounds(start)
        self._check_bounds(end)

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, Cell]] = [
            (self._heuristic(start, end), 0, start)
        ]
        g_score: Dict[Cell, int] = {start: 0}
        came_from: Dict[Cell, Cell] = {}

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if g > g_score[current]:
                continue  # stale heap entry (better path already found)
            if current == end:
                return self._reconstruct_path(came_from, start, end)
            for dr, dc in self._directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(neighbor):
                    continue
                cost = self.grid[neighbor[0]][neighbor[1]]
                if cost == 0:
                    continue
                tentative_g = g + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
        return None

    def _heuristic(self, a: Cell, b: Cell) -> int:
        """Manhattan distance between two cells.

        Admissible here because every move costs at least 1, so the true
        remaining cost can never be less than the step count.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: Dict[Cell, Cell], start: Cell, end: Cell) -> List[Cell]:
        """Walk back from end to start via came_from and return the path."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def _check_bounds(self, cell: Cell) -> None:
        """Raise ValueError if the cell is outside the grid."""
        if not self._in_bounds(cell):
            raise ValueError(
                f"Coordinate {cell} is out of bounds for a "
                f"{self.rows}x{self.cols} grid"
            )

    def _in_bounds(self, cell: Cell) -> bool:
        """Return True if the cell is inside the grid."""
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    @staticmethod
    def _parse_cell(cell: Cell) -> Cell:
        """Normalize and validate a coordinate, raising ValueError if malformed."""
        try:
            r, c = cell
        except (TypeError, ValueError):
            raise ValueError(f"Cell must be a (row, col) pair, got {cell!r}")
        if not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"Cell coordinates must be ints, got {cell!r}")
        return (r, c)
```

**`test_a_star.py`**

```python
"""Tests for AStarGrid."""
import pytest
from a_star import AStarGrid


def path_cost(grid, path):
    """Total cost of a path, excluding the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Check path endpoints, adjacency, and passability."""
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert grid[r][c] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional steps only


def test_simple_path_uniform_grid():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, (0, 0), (2, 2), path)
    assert path_cost(grid, path) == 4  # 4 moves on a uniform grid


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (4, 4))
    assert path is not None
    assert_valid_path(grid, (0, 0), (4, 4), path)
    # Only route: along the top row, then down the rightmost column
    assert path_cost(grid, path) == 8


def test_weighted_grid_prefers_low_cost_cells():
    grid = [
        [1, 10, 10, 10, 1],
        [1, 10, 1, 10, 1],
        [1, 1, 1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 4))
    assert path is not None
    assert_valid_path(grid, (0, 0), (0, 4), path)
    # Straight across the top would cost 31; the cheap detour costs 8
    assert path_cost(grid, path) == 8
    assert path != [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_no_path_exists_fully_blocked():
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None


def test_start_equals_end():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    solver = AStarGrid([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
    with pytest.raises(ValueError):
        solver.find_path((3, 0), (0, 0))   # row out of bounds
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, 3))   # col out of bounds
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (0, 0))  # negative row
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, -1))  # negative col


def test_wall_endpoint_returns_none():
    solver = AStarGrid([[1, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1]])
    assert solver.find_path((0, 1), (0, 2)) is None  # start is a wall
    assert solver.find_path((0, 0), (0, 1)) is None  # end is a wall
    assert solver.find_path((0, 1), (0, 1)) is None  # wall, even start == end


def test_invalid_grid_raises_value_error():
    with pytest.raises(ValueError):
        AStarGrid([])                    # empty grid
    with pytest.raises(ValueError):
        AStarGrid([[1, 1], [1]])         # ragged rows
    with pytest.raises(ValueError):
        AStarGrid([[1, -1]])             # negative cost
```

Run with `pytest test_a_star.py -v`.

A few design notes:

- **Optimality**: Manhattan distance is admissible here (every step costs ≥ 1, so no path can cost less than its length in moves) and consistent, so the first time `end` is popped from the heap the path is optimal. The `g > g_score[current]` stale-entry skip makes the result correct even with duplicate heap entries.
- **Cost definition**: total cost is the sum of cell costs along the path, excluding the start cell (you don't pay to "enter" it). `start == end` therefore has cost 0.
- **Edge-case precedence**: bounds are checked first (`ValueError`), then walls (`None`), then the `start == end` shortcut. So `start == end` on a wall returns `None` — documented in the `find_path` docstring.
- Two extra tests beyond the six requested cover wall endpoints and grid validation; the two requested tests are `test_invalid_coordinates_raise_value_error` and the others exactly as you specified.