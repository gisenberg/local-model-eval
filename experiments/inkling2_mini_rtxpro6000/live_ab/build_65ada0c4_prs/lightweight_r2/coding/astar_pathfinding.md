Here's a complete implementation plus a test suite. I split it into two files: `a_star_grid.py` (the implementation) and `test_a_star_grid.py` (the tests).

## `a_star_grid.py`

```python
"""A* pathfinding on a weighted 2D grid.

The grid stores per-cell movement costs: 0 marks an impassable wall and any
positive integer is the cost of entering that cell. The cost of a path is the
sum of the costs of the cells entered after the start cell (the start cell's
own cost is not paid).
"""

import heapq
from typing import Dict, List, Optional, Tuple


class AStarGrid:
    """A* pathfinding on a weighted 2D grid.

    Attributes:
        grid: A validated copy of the 2D grid of cell costs.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    _DIRECTIONS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    )

    def __init__(self, grid: List[List[int]]) -> None:
        """Store a validated copy of the grid.

        Args:
            grid: 2D list of cell costs. 0 = impassable wall; a positive
                int = cost to enter that cell.

        Raises:
            ValueError: If the grid is empty, contains empty rows, is not
                rectangular, or contains a negative cost.
        """
        if not grid:
            raise ValueError("Grid must not be empty")
        cols = len(grid[0])
        if cols == 0:
            raise ValueError("Grid must not contain empty rows")
        for row in grid:
            if len(row) != cols:
                raise ValueError("Grid must be rectangular (all rows equal length)")
            for cost in row:
                if cost < 0:
                    raise ValueError("Grid costs must be non-negative integers")
        self.grid: List[List[int]] = [row[:] for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = cols

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the cheapest path from ``start`` to ``end``.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the end cell.

        Returns:
            A list of (row, col) coordinates from start to end inclusive,
            or ``None`` if no path exists or either endpoint is a wall.
            If ``start == end``, returns ``[start]`` (unless it is a wall).

        Raises:
            ValueError: If either endpoint is out of bounds.
        """
        self._check_bounds(start)
        self._check_bounds(end)

        # Walls are impassable.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Trivial path.
        if start == end:
            return [start]

        # (f_score, g_score, cell) — g is the tiebreaker so heap entries are
        # always comparable (avoids comparing cells directly).
        open_heap: List[Tuple[int, int, Tuple[int, int]]] = [
            (self._heuristic(start, end), 0, start)
        ]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)

            # Skip stale entries superseded by a better g-score.
            if current_g > g_score.get(current, float("inf")):
                continue

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for dr, dc in self._DIRECTIONS:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                cell_cost = self.grid[nr][nc]
                if cell_cost == 0:
                    continue
                neighbor = (nr, nc)
                tentative_g = current_g + cell_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        return None

    def _check_bounds(self, coord: Tuple[int, int]) -> None:
        """Raise ``ValueError`` if ``coord`` lies outside the grid."""
        row, col = coord
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"Coordinate {coord} is out of bounds for a grid with "
                f"{self.rows} rows and {self.cols} columns"
            )

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance between two cells.

        Admissible because every cell costs at least 1, so no path can be
        cheaper than the minimum number of steps.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Walk back from ``end`` to ``start`` and return the path inclusive."""
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
"""Pytest tests for AStarGrid (see a_star_grid.py)."""

import pytest

from a_star_grid import AStarGrid


# ---------- helpers ----------

def total_cost(grid, path):
    """Sum of grid costs of cells entered after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(path, grid, start, end):
    """Assert the path connects start to end, is wall-free, and is contiguous."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert all(grid[r][c] > 0 for r, c in path)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


# ---------- tests ----------

def test_simple_path_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal: Manhattan distance is 4 moves, each cell costs 1.
    assert total_cost(grid, path) == 4
    assert len(path) == 5


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    start, end = (0, 0), (4, 4)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal: 8 cells must be entered to route around the wall ring;
    # the uniform cost-1 grid means 8 is the lower bound and is achieved.
    assert total_cost(grid, path) == 8


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [1, 5, 5, 1],
        [1, 1, 5, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (2, 3)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal: Manhattan distance = 5 cells entered (lower bound 5 since every
    # cell costs >= 1). The cheap bottom/left route achieves exactly 5; any
    # route through the 5-cost cells costs 9 or more.
    assert total_cost(grid, path) == 5


def test_no_path_fully_blocked():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    path = AStarGrid(grid).find_path((1, 1), (1, 1))

    assert path == [(1, 1)]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    obj = AStarGrid([[1, 1], [1, 1]])

    with pytest.raises(ValueError):
        obj.find_path((-1, 0), (0, 0))  # start row out of bounds
    with pytest.raises(ValueError):
        obj.find_path((0, 0), (0, 2))   # end column out of bounds
    with pytest.raises(ValueError):
        obj.find_path((2, 0), (0, 0))   # start row too large
    with pytest.raises(ValueError):
        obj.find_path((0, -1), (1, 1))  # start column out of bounds


def test_wall_start_or_end_returns_none():
    grid = [
        [1, 0],
        [1, 1],
    ]
    obj = AStarGrid(grid)

    assert obj.find_path((0, 1), (1, 1)) is None  # start is a wall
    assert obj.find_path((0, 0), (0, 1)) is None  # end is a wall
```

Run with:

```bash
pytest test_a_star_grid.py -v
```

## Design notes

- **Cost convention**: a path's cost is the sum of grid costs of the cells entered after the start (the start cell is free). This is documented in the class docstring and used consistently in the tests.
- **Heuristic admissibility**: Manhattan distance is admissible because every non-wall cell costs ≥ 1, so the cheapest path needs at least that many steps. It is also consistent, so the first time `end` is popped from the heap the path is optimal — that's what makes the "return immediately" step safe.
- **Edge-case precedence** in `find_path`: out-of-bounds → `ValueError`, then wall endpoints → `None`, then `start == end` → `[start]`, then the search. (So `start == end` on a wall cell returns `None` — documented behavior; the trivial-path case is for valid cells.)
- **Heap entries** are `(f, g, cell)` with `g` as a tiebreaker, so heap comparisons never touch the cell tuples. Stale entries are skipped via lazy deletion (`current_g > g_score[current]`), which avoids needing a closed set; correctness is also independent of heuristic consistency.
- **`__init__` validation** is light but useful: it rejects empty grids, non-rectangular grids, and negative costs.

The 6 required tests are all present (plus one bonus wall-endpoint test). Each asserts path validity (endpoints, contiguity, wall-freeness) and optimality via the total cost, justified by a hand-verified lower bound (e.g., Manhattan distance × minimum cell cost) for each grid.