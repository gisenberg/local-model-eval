# A* Pathfinding Implementation

Here's a complete implementation. I'll give you two files: `astar.py` (the solver) and `test_astar.py` (the tests).

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid.

Grid convention:
    0          -> impassable wall
    positive   -> cost to enter that cell

Movement is 4-directional (no diagonals). The heuristic is Manhattan distance,
which is admissible here because every enter-cost is >= 1.
"""

from __future__ import annotations

import heapq
from typing import Iterator, List, Optional, Set, Tuple

Position = Tuple[int, int]


class AStarGrid:
    """A* solver for a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Store and validate the grid.

        Args:
            grid: 2D list of non-negative ints. 0 = wall, >0 = enter cost.

        Raises:
            ValueError: if the grid is empty, non-rectangular, or contains a
                negative or non-integer value.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must contain at least one row and column")
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("Grid must be rectangular")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("Grid values must be integers")
                if value < 0:
                    raise ValueError("Grid values must be non-negative")
        self.grid = grid
        self.rows = len(grid)
        self.cols = width

    def find_path(self, start: Position, end: Position) -> Optional[List[Position]]:
        """Find the lowest-cost path from start to end.

        Args:
            start: (row, col) starting cell.
            end: (row, col) goal cell.

        Returns:
            List of (row, col) cells from start to end inclusive, or None if
            either cell is a wall or the goal is unreachable. Note that the
            wall check takes precedence over the start == end case.

        Raises:
            ValueError: if start or end is malformed or out of bounds.
        """
        self._validate_position(start, "start")
        self._validate_position(end, "end")

        # Walls are never entered (checked before start == end).
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, Position]] = [(self._heuristic(start, end), 0, start)]
        counter = 0  # unique tiebreaker so the heap never compares positions
        g_score: dict[Position, int] = {start: 0}
        came_from: dict[Position, Position] = {}
        closed: Set[Position] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue  # stale heap entry

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            closed.add(current)

            for neighbor in self._neighbors(current):
                enter_cost = self.grid[neighbor[0]][neighbor[1]]
                if enter_cost == 0:
                    continue
                tentative_g = g_score[current] + enter_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return None

    def _neighbors(self, position: Position) -> Iterator[Position]:
        """Yield in-bounds 4-directional neighbors of a cell."""
        row, col = position
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield (nr, nc)

    def _heuristic(self, a: Position, b: Position) -> int:
        """Manhattan distance between two cells (admissible for cost >= 1)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _validate_position(self, position: Position, name: str) -> None:
        """Raise ValueError if a position is malformed or out of bounds."""
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError(f"{name} must be a (row, col) pair, got {position!r}")
        row, col = position
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError(f"{name} row must be an int, got {row!r}")
        if isinstance(col, bool) or not isinstance(col, int):
            raise ValueError(f"{name} column must be an int, got {col!r}")
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"{name} {position!r} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    def _reconstruct_path(
        self, came_from: dict[Position, Position], start: Position, end: Position
    ) -> List[Position]:
        """Rebuild the start-to-end path by walking the came_from chain."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Tests for AStarGrid."""

import pytest

from astar import AStarGrid

Grid = list[list[int]]
Path = list[tuple[int, int]]


def total_cost(grid: Grid, path: Path) -> int:
    """Cost of entering each cell after the start."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_path_valid(grid: Grid, path: Path, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Check endpoints, 4-directional adjacency, and no wall crossings."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "moves must be 4-directional"
        assert grid[r2][c2] > 0, "path must not enter walls"


def test_simple_path_uniform_grid() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)
    start, end = (0, 0), (2, 2)

    path = solver.find_path(start, end)

    assert_path_valid(grid, path, start, end)
    assert total_cost(grid, path) == 4  # Manhattan distance, unit costs


def test_path_around_obstacles() -> None:
    grid = [
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    solver = AStarGrid(grid)
    start, end = (0, 0), (0, 3)

    path = solver.find_path(start, end)

    assert_path_valid(grid, path, start, end)
    # Only gap between the left block and the right column is row 2,
    # forcing a detour: 2 down + 3 across + 2 up = 7 steps.
    assert total_cost(grid, path) == 7


def test_weighted_grid_prefers_lower_cost_cells() -> None:
    grid = [
        [1, 2, 2],
        [1, 2, 2],
        [2, 2, 1],
    ]
    solver = AStarGrid(grid)
    start, end = (0, 0), (2, 2)

    path = solver.find_path(start, end)

    assert_path_valid(grid, path, start, end)
    # Cheapest route is cost 1+2+2+1 = 6 (via row 1/2); the shorter-looking
    # top route costs 2+2+2+1 = 7, so 6 proves the solver prefers cheap cells.
    assert total_cost(grid, path) == 6


def test_no_path_when_fully_blocked() -> None:
    grid = [
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
    ]
    solver = AStarGrid(grid)

    assert solver.find_path((0, 0), (0, 4)) is None


def test_start_equals_end() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)

    path = solver.find_path((1, 1), (1, 1))

    assert path == [(1, 1)]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)

    for bad in [(3, 0), (0, 3), (-1, 0), (0, -1), (9, 9)]:
        with pytest.raises(ValueError):
            solver.find_path(bad, (0, 0))
        with pytest.raises(ValueError):
            solver.find_path((0, 0), bad)


def test_wall_start_or_end_returns_none() -> None:
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    solver = AStarGrid(grid)

    assert solver.find_path((1, 1), (2, 2)) is None   # wall start
    assert solver.find_path((0, 0), (1, 1)) is None   # wall end
    assert solver.find_path((1, 1), (1, 1)) is None   # wall start == end
```

## Design notes

- **Optimality**: Manhattan distance is admissible for any grid where all enter-costs are ≥ 1 (each step costs at least 1), so A* returns a minimum-cost path. In `test_weighted_grid_prefers_lower_cost_cells`, the Manhattan-shortest-looking route costs 7, while the optimal one costs 6 — a genuine optimality check, not just a validity check.
- **Heap correctness**: entries are `(f_score, counter, position)` where `counter` is a monotonically increasing tiebreaker, so the heap never has to compare positions. Stale entries (superseded by better g-scores) are discarded via the `closed` set.
- **Edge-case precedence** (documented in `find_path`): out-of-bounds → `ValueError`; wall → `None`; `start == end` → `[start]`. So a `start == end` on a wall cell returns `None` (wall check wins), while an out-of-bounds `start == end` raises `ValueError`.
- **Cost of a path**: the sum of enter-costs of cells after the start (start costs 0 by convention).

## Running the tests

```bash
pip install pytest
python -m pytest test_astar.py -v
```

Note the 7th test (`test_wall_start_or_end_returns_none`) is an extra beyond the 6 required cases — it covers the wall-handling edge case, which the requirement calls out but the test list doesn't explicitly name. If you'd prefer wall-vs-`start == end` precedence to be the other way around, it's a one-line reorder in `find_path`.