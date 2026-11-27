Here's a complete implementation. The key design point: since every passable cell costs ≥ 1, plain Manhattan distance is an admissible (and consistent) heuristic, so A* with a min-heap returns the minimum-cost path.

**`astar.py`**

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """Finds the cheapest 4-directional path between cells of a weighted grid.

    Grid values are movement costs: 0 means impassable (wall); any positive
    integer is the cost of entering that cell. The total cost of a path is
    the sum of the entry costs of the cells entered after the start cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store the grid and validate its shape.

        Args:
            grid: Row-major 2D list of non-negative ints (0 = wall).

        Raises:
            ValueError: If the grid is empty, ragged, or contains an
                invalid (negative or non-integer) cell value.
        """
        if not isinstance(grid, list) or not grid:
            raise ValueError("grid must be a non-empty list of rows")
        width = len(grid[0])
        if width == 0:
            raise ValueError("grid must contain at least one column")
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("grid must be rectangular")
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("cell values must be non-negative ints (0 = wall)")
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = width

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the cheapest path from start to end.

        Args:
            start: (row, col) start cell.
            end: (row, col) target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists (e.g. a wall start/end or a blocked
            grid). If start == end, returns [start].

        Raises:
            ValueError: If start or end is out of bounds.

        Note:
            Wall check takes precedence over the start == end case: a wall
            start (even with start == end) returns None.
        """
        self._validate_point(start)
        self._validate_point(end)

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, Point]] = [
            (self._heuristic(start, end), 0, start)
        ]
        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}
        closed: Set[Point] = set()

        while open_heap:
            f_cost, g_cost, current = heapq.heappop(open_heap)
            if current in closed:
                continue  # stale heap entry
            if g_cost > g_score.get(current, float("inf")):
                continue  # a better path was found later
            if current == end:
                return self._reconstruct(came_from, start, end)

            closed.add(current)
            for neighbor in self._neighbors(current):
                if neighbor in closed:
                    continue
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                if move_cost == 0:
                    continue
                tentative_g = g_cost + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_new = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_new, tentative_g, neighbor))

        return None

    def _validate_point(self, point: Tuple[int, int]) -> None:
        """Raise ValueError if point is malformed or outside the grid."""
        try:
            r, c = point
        except (TypeError, ValueError):
            raise ValueError(f"Invalid coordinate {point!r}: expected (row, col)") from None
        if not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"Invalid coordinate {point!r}: row and col must be ints")
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"Coordinate {point!r} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    def _heuristic(self, a: Point, b: Point) -> int:
        """Manhattan distance between two points (admissible: min cell cost is 1)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, point: Point) -> List[Point]:
        """Return the in-bounds 4-directional neighbors of a cell."""
        r, c = point
        result = []
        if r > 0:
            result.append((r - 1, c))
        if r < self.rows - 1:
            result.append((r + 1, c))
        if c > 0:
            result.append((r, c - 1))
        if c < self.cols - 1:
            result.append((r, c + 1))
        return result

    def _reconstruct(
        self, came_from: Dict[Point, Point], start: Point, end: Point
    ) -> List[Point]:
        """Walk the came_from chain from end back to start and return the path."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

**`test_astar.py`**

```python
"""Pytest tests for AStarGrid."""
import pytest

from astar import AStarGrid


def total_cost(grid, path):
    """Sum of entry costs of cells entered after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, path, start, end):
    """Assert path is a legal 4-directional route from start to end."""
    assert path[0] == start
    assert path[-1] == end
    rows, cols = len(grid), len(grid[0])
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0  # every cell on the path is passable
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # single 4-directional step


def test_simple_path_on_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert len(path) == 5  # 4 moves + start cell
    assert total_cost(grid, path) == 4  # Manhattan distance; all costs 1


def test_path_around_obstacles():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 2))
    assert total_cost(grid, path) == 6  # forced detour down and back up


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [
        [1, 5, 5],
        [1, 1, 5],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert total_cost(grid, path) == 4  # optimal: hug the cheap cells
    assert all(grid[r][c] <= 1 for r, c in path[1:])  # never pays 5-cost cells


def test_no_path_exists_fully_blocked():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    solver = AStarGrid([[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (1, 1))  # start above grid
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (2, 0))   # end below grid
    with pytest.raises(ValueError):
        solver.find_path((0, 5), (0, 0))   # start right of grid


def test_wall_start_or_end_returns_none():  # bonus coverage
    grid = [[1, 1], [1, 0]]
    solver = AStarGrid(grid)
    assert solver.find_path((1, 1), (0, 0)) is None  # wall start
    assert solver.find_path((0, 0), (1, 1)) is None  # wall end
```

Run with `pytest test_astar.py -v`.

Notes on the design:

- **Optimality:** Manhattan distance never overestimates the cost to the end when all cell costs are ≥ 1 (the cheapest possible step costs 1, and each step reduces Manhattan distance by at most 1). It's also *consistent*, so a node is finalized when first popped — no re-opening needed, though stale heap entries are still guarded against with the `closed` set and a `g` check.
- **Edge-case precedence:** out-of-bounds → `ValueError`; then wall start/end → `None`; then `start == end` → `[start]`.
- **Cost convention:** the path cost is the sum of entry costs of cells after the start (start cell cost excluded), which is exactly what A* accumulates in `g_score`.
- Each test asserts both path legality (bounds, passability, adjacency, endpoints) and the hand-computed optimal total cost, which is the optimality check.