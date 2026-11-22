# A* Pathfinding Implementation

## `a_star_grid.py`

```python
"""A* pathfinding on a weighted 2D grid with 4-directional movement."""

from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """A* pathfinding solver over a weighted 2D grid.

    Grid semantics:
        - 0           : impassable wall
        - positive int: cost to enter that cell

    Coordinates are (row, col). Manhattan distance is used as the heuristic.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the solver.

        Args:
            grid: Non-empty, rectangular list of rows of non-negative ints.

        Raises:
            ValueError: If the grid is empty, ragged, or contains invalid values.
        """
        self._validate_grid(grid)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find the minimum-cost path from start to end.

        Args:
            start: Starting (row, col) cell.
            end: Target (row, col) cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or None
            if no path exists (e.g., start or end is a wall).

        Raises:
            ValueError: If start or end is malformed or out of bounds.
        """
        self._validate_coord(start, "start")
        self._validate_coord(end, "end")

        # Wall endpoints: no valid path can begin or end in an impassable cell.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Trivial path.
        if start == end:
            return [start]

        # Heap entries: (f_score, tie_breaker, g_score, row, col)
        open_heap: List[Tuple[int, int, int, int, int]] = []
        counter = 0
        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}

        heappush(open_heap, (self._heuristic(start[0], start[1], end), counter, 0, start[0], start[1]))

        while open_heap:
            _, _, g, r, c = heappop(open_heap)
            current = (r, c)

            # Skip stale heap entries (a better path was found since push).
            if g > g_score.get(current, float("inf")):
                continue

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                cell_cost = self.grid[nr][nc]
                if cell_cost == 0:
                    continue

                new_g = g + cell_cost
                if new_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = new_g
                    came_from[(nr, nc)] = current
                    counter += 1
                    f = new_g + self._heuristic(nr, nc, end)
                    heappush(open_heap, (f, counter, new_g, nr, nc))

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic(self, r: int, c: int, end: Point) -> int:
        """Manhattan distance from (r, c) to end.

        Admissible (never overestimates) because every step costs at least 1.
        """
        return abs(r - end[0]) + abs(c - end[1])

    def _validate_grid(self, grid: List[List[int]]) -> None:
        """Validate grid structure and cell values."""
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError("grid must be a non-empty list of rows")
        cols: Optional[int] = None
        for row in grid:
            if not isinstance(row, list) or len(row) == 0:
                raise ValueError("grid rows must be non-empty lists")
            if cols is None:
                cols = len(row)
            elif len(row) != cols:
                raise ValueError("grid must be rectangular (all rows same length)")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("grid values must be non-negative integers")

    def _validate_coord(self, coord: Point, name: str) -> None:
        """Validate a coordinate; raises ValueError if malformed or out of bounds."""
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise ValueError(f"{name} must be a (row, col) tuple, got {coord!r}")
        r, c = coord
        if not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"{name} coordinates must be ints, got {coord!r}")
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"{name} {coord!r} is out of bounds (grid is {self.rows}x{self.cols})")

    def _reconstruct_path(self, came_from: Dict[Point, Point], start: Point, end: Point) -> List[Point]:
        """Walk came_from backward from end to start and return the path."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

## `test_a_star_grid.py`

```python
"""Tests for AStarGrid. Run with: pytest test_a_star_grid.py -v"""

import pytest
from a_star_grid import AStarGrid


def total_cost(grid, path):
    """Total cost of a path: sum of cell costs entered after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, path, start, end):
    """Assert a path is well-formed, within bounds, and crosses no walls."""
    rows, cols = len(grid), len(grid[0])
    assert path[0] == start
    assert path[-1] == end
    assert all(0 <= r < rows and 0 <= c < cols for r, c in path)
    assert all(grid[r][c] > 0 for r, c in path)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional only


def test_simple_path_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    path = AStarGrid(grid).find_path((0, 0), (4, 4))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (4, 4))
    # Any valid path is >= 8 steps (Manhattan), each step costs 1 -> optimal cost is 8.
    assert total_cost(grid, path) == 8


def test_path_around_obstacles():
    grid = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 4))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 4))
    # Must detour to row 3 (only free row crossing the col-2 wall):
    # 3 down + 4 right + 3 up = 10 moves, all cost 1.
    assert total_cost(grid, path) == 10


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [10, 10, 10, 10, 10],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 10],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 4))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 4))
    # Straight row-0 shortcut costs 51; optimal is down to the cheap corridor:
    # 1 (row1 col0) + 1 (row1 col1) + 1 + 1 + 1 + 10 (cell (2,4)) = 15.
    assert total_cost(grid, path) == 15


def test_no_path_exists_fully_blocked():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None


def test_start_equals_end():
    grid = [[1] * 3 for _ in range(3)]
    assert AStarGrid(grid).find_path((1, 1), (1, 1)) == [(1, 1)]


@pytest.mark.parametrize("start,end", [
    ((-1, 0), (0, 0)),   # negative row (start)
    ((0, -1), (0, 0)),   # negative col (start)
    ((5, 0), (0, 0)),    # row beyond grid (start)
    ((0, 5), (0, 0)),    # col beyond grid (start)
    ((0, 0), (-1, 0)),   # negative row (end)
    ((0, 0), (0, 99)),   # col beyond grid (end)
])
def test_invalid_coordinates_raise_value_error(start, end):
    grid = [[1] * 3 for _ in range(3)]
    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)


def test_wall_start_or_end_returns_none():
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    solver = AStarGrid(grid)
    assert solver.find_path((0, 1), (2, 2)) is None   # start is a wall
    assert solver.find_path((2, 2), (0, 1)) is None   # end is a wall
```

## Design notes

- **Optimality:** Manhattan distance is admissible here because every enterable cell costs ≥ 1, so the true remaining cost is always ≥ the step count. It's also consistent (h changes by ≤ 1 per step, edge costs ≥ 1), so the first time `end` is popped from the heap, the path is provably minimal.
- **Min-heap:** heap entries are `(f_score, counter, g_score, r, c)`. The monotonically increasing `counter` breaks ties deterministically (avoiding tuple comparisons), and stale entries (superseded by a better `g_score`) are skipped lazily.
- **Edge-case ordering:** bounds → `ValueError`, then wall endpoints → `None`, then `start == end` → `[start]`. So a wall start/end returns `None` even if `start == end`, since a path may not begin in an impassable cell.
- **Cost accounting:** the cost of a path is the sum of cell costs for every cell *entered* after the start cell (standard interpretation of "cost to enter that cell").
- **Complexity:** `O((N + E) log N)` where N is cells and E is relaxed edges — fine for typical grids.

The extra test (`test_wall_start_or_end_returns_none`) is included beyond the 6 required tests to cover the wall-endpoint edge case explicitly.