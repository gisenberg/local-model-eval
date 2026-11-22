Here's a complete implementation split into two files: `astar.py` (the solver) and `test_astar.py` (pytest tests).

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
import math
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted grid.

    Grid semantics:
        - 0 means an impassable wall.
        - Any positive int means the cost of *entering* that cell.

    Movement is 4-directional (up/down/left/right); diagonals are not allowed.
    The heuristic is Manhattan distance to the goal, which is admissible
    (and consistent) here because every passable cell costs at least 1 to
    enter — so the returned path is optimal.
    """

    _DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, grid: List[List[int]]) -> None:
        """Store the grid after validating it.

        Args:
            grid: 2D list of non-negative ints; 0 = wall, >0 = entry cost.

        Raises:
            ValueError: if the grid is empty, ragged, or holds invalid values.
        """
        if not isinstance(grid, list) or not grid:
            raise ValueError("grid must be a non-empty list of rows")
        if not all(isinstance(row, list) for row in grid):
            raise ValueError("each grid row must be a list")
        width = len(grid[0])
        if width == 0:
            raise ValueError("grid rows must not be empty")
        for row in grid:
            if len(row) != width:
                raise ValueError("grid must be rectangular (all rows same length)")
            for cell in row:
                if not isinstance(cell, int) or isinstance(cell, bool) or cell < 0:
                    raise ValueError("grid values must be non-negative ints")
        # Defensive copy so external mutation cannot affect the solver.
        self._grid: List[List[int]] = [row[:] for row in grid]
        self._rows: int = len(self._grid)
        self._cols: int = len(self._grid[0])

    def find_path(
        self, start: Coord, end: Coord
    ) -> Optional[List[Coord]]:
        """Find the minimum-total-cost path from start to end.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if no path exists (e.g. a wall blocks the way, or start/end
            is a wall).

        Raises:
            ValueError: if start or end is out of bounds.
        """
        # Edge case order: bounds -> wall -> start == end -> A* search.
        self._check_bounds(start)
        self._check_bounds(end)

        if self._grid[start[0]][start[1]] == 0 or self._grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, Coord]] = [(self._heuristic(start, end), start)]
        g_score: Dict[Coord, int] = {start: 0}
        parent: Dict[Coord, Coord] = {}

        while open_heap:
            f, current = heapq.heappop(open_heap)

            # Skip stale heap entries (a better g was found after this was pushed).
            if f > g_score[current] + self._heuristic(current, end):
                continue

            if current == end:
                return self._reconstruct_path(parent, start, end)

            for dr, dc in self._DIRECTIONS:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(neighbor):
                    continue
                step_cost = self._grid[neighbor[0]][neighbor[1]]
                if step_cost == 0:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return None  # open set exhausted: no path exists

    def _check_bounds(self, point: Coord) -> None:
        """Raise ValueError if (row, col) lies outside the grid."""
        row, col = point
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"coordinate {point!r} is out of bounds "
                f"for a {self._rows}x{self._cols} grid"
            )

    def _in_bounds(self, point: Coord) -> bool:
        """Return True if (row, col) lies inside the grid."""
        row, col = point
        return 0 <= row < self._rows and 0 <= col < self._cols

    @staticmethod
    def _heuristic(a: Coord, b: Coord) -> int:
        """Manhattan distance between two cells (admissible heuristic)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self, parent: Dict[Coord, Coord], start: Coord, end: Coord
    ) -> List[Coord]:
        """Walk parent pointers from end back to start and reverse."""
        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path


if __name__ == "__main__":
    demo = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    path = AStarGrid(demo).find_path((0, 0), (0, 4))
    print("path:", path)
    if path:
        print("cost:", sum(demo[r][c] for r, c in path[1:]))
```

## `test_astar.py`

```python
"""Tests for AStarGrid (run with: pytest test_astar.py -v)."""

import heapq
from typing import List, Optional, Tuple

import pytest

from astar import AStarGrid

Coord = Tuple[int, int]


# ---------- helpers ----------

def path_cost(grid: List[List[int]], path: List[Coord]) -> int:
    """Total entry cost of the cells after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_path_valid(grid: List[List[int]], start: Coord, end: Coord,
                      path: List[Coord]) -> None:
    """Assert the path starts at start, ends at end, and is 4-connected
    through passable cells."""
    assert path[0] == start
    assert path[-1] == end
    rows, cols = len(grid), len(grid[0])
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0, f"path enters impassable cell {(r, c)}"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "cells must be 4-connected"


def dijkstra_optimal_cost(grid: List[List[int]], start: Coord,
                          end: Coord) -> float:
    """Independent optimal-cost oracle (Dijkstra over the same grid)."""
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist.get((r, c), float("inf")):
            continue
        if (r, c) == end:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                nd = d + grid[nr][nc]
                if nd < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return dist.get(end, float("inf"))


# ---------- tests ----------

def test_simple_path_uniform_grid() -> None:
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    assert path is not None
    assert_path_valid(grid, start, end, path)
    assert path_cost(grid, path) == 4  # Manhattan distance, uniform cost
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, start, end)


def test_path_around_obstacles() -> None:
    grid = [[1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]]
    start, end = (0, 0), (0, 4)
    path = AStarGrid(grid).find_path(start, end)
    assert path is not None
    assert_path_valid(grid, start, end, path)
    assert path_cost(grid, path) == 6  # forced detour below the wall row
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, start, end)


def test_weighted_grid_prefers_low_cost_cells() -> None:
    grid = [[1, 5, 5],
            [1, 1, 1]]
    start, end = (0, 0), (0, 2)
    path = AStarGrid(grid).find_path(start, end)
    assert path is not None
    assert_path_valid(grid, start, end, path)
    assert path_cost(grid, path) == 4          # bottom route beats top (10)
    assert (1, 1) in path                        # takes the low-cost route
    assert (0, 1) not in path
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, start, end)


def test_no_path_exists_fully_blocked() -> None:
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    start, end = (0, 0), (0, 2)
    assert AStarGrid(grid).find_path(start, end) is None


def test_start_equals_end() -> None:
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    start = (1, 1)
    path = AStarGrid(grid).find_path(start, start)
    assert path == [start]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error() -> None:
    grid = [[1, 1],
            [1, 1]]
    solver = AStarGrid(grid)
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (0, 0))  # start row out of bounds (negative)
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (2, 0))   # end row out of bounds
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, 2))   # end column out of bounds


def test_wall_start_or_end_returns_none() -> None:
    grid = [[1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]]
    solver = AStarGrid(grid)
    assert solver.find_path((0, 1), (2, 2)) is None  # start on wall
    assert solver.find_path((0, 0), (2, 1)) is None  # end on wall
```

Run with:

```bash
pytest test_astar.py -v
```

**Notes on the design:**

- **Optimality**: every passable cell costs ≥ 1, so Manhattan distance never overestimates the remaining cost and is consistent (`h(a) ≤ cost(a→b) + h(b)`), which guarantees the first time `end` is popped from the heap, its `g` value is optimal. The tests double-check this against an independent Dijkstra oracle.
- **Edge-case precedence**: bounds check → wall check → `start == end` → search. So `start == end` on a wall returns `None` (walls are impassable, so a zero-cost "path" through one is not valid). `start == end` on a passable cell returns `[start]` with cost 0.
- **Stale entries**: because a better `g` may be found after a node is pushed, the heap holds lazy entries; stale ones are discarded via the `f > g_score + h` check, which is why a `closed` set isn't needed.
- **`total cost`** = sum of entry costs of cells *after* the start (you don't pay to enter the cell you start in); this is the convention used consistently in the implementation and the tests.
- The 6 required test scenarios are all present, plus a 7th test covering the wall-start/end edge case from the spec.