Here's a complete implementation. Two files: `a_star.py` (the implementation) and `test_a_star.py` (the tests).

**`a_star.py`**

```python
"""A* pathfinding on a weighted 2D grid.

Grid values: 0 = impassable wall, positive int = cost to enter that cell.
Movement is 4-directional (up/down/left/right). Path cost is the sum of
the entering-costs of all cells *after* the start cell.
"""

import heapq
from typing import List, Optional, Tuple

Grid = List[List[int]]
Point = Tuple[int, int]


class AStarGrid:
    """A* pathfinder over a weighted 2D grid."""

    def __init__(self, grid: Grid) -> None:
        """Initialize with a grid of movement costs.

        Args:
            grid: Non-empty rectangular list of rows; each value is an
                  int >= 0 (0 = wall, positive = cost to enter).

        Raises:
            ValueError: If the grid is empty, ragged, or contains
                        invalid values.
        """
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError("grid must be a non-empty list of rows")
        if not isinstance(grid[0], list) or len(grid[0]) == 0:
            raise ValueError("grid must contain at least one column")
        cols = len(grid[0])
        for row in grid:
            if len(row) != cols:
                raise ValueError("grid must be rectangular")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("grid values must be non-negative integers")
        self.grid: Grid = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = cols

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find the shortest path from start to end.

        Args:
            start: (row, col) start cell.
            end: (row, col) target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if the start or end cell is a wall or no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        self._check_bounds(start, "start")
        self._check_bounds(end, "end")

        # Wall endpoints: no valid path can begin or end on a wall.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, Point]] = [(self._heuristic(start, end), start)]
        g_scores: dict = {start: 0}
        came_from: dict = {}
        closed: set = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue  # stale heap entry
            closed.add(current)

            if current == end:
                return self._reconstruct(came_from, start, end)

            for nb in self._neighbors(current):
                tentative = g_scores[current] + self.grid[nb[0]][nb[1]]
                if tentative < g_scores.get(nb, float("inf")):
                    g_scores[nb] = tentative
                    came_from[nb] = current
                    f = tentative + self._heuristic(nb, end)
                    heapq.heappush(open_heap, (f, nb))

        return None  # open set exhausted: no path exists

    def _neighbors(self, point: Point):
        """Yield in-bounds, passable 4-directional neighbors of point."""
        r, c = point
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                yield (nr, nc)

    def _heuristic(self, a: Point, b: Point) -> int:
        """Manhattan distance between a and b.

        Admissible (never overestimates) because every cell costs >= 1
        to enter, so the true remaining cost is always >= the step count.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _check_bounds(self, point: Point, name: str) -> None:
        """Raise ValueError if point is outside the grid bounds."""
        r, c = point
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"{name} coordinate {point} is out of bounds for a "
                f"{self.rows}x{self.cols} grid"
            )

    def _reconstruct(self, came_from: dict, start: Point, end: Point) -> List[Point]:
        """Walk came_from from end back to start and return the path."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

**`test_a_star.py`**

```python
"""Tests for AStarGrid (A* pathfinding on a weighted 2D grid)."""

import heapq

import pytest

from a_star import AStarGrid


# ---------- helpers ----------

def path_cost(grid, path):
    """Total cost of entering cells in path, excluding the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Assert the path is a contiguous, passable route from start to end."""
    rows, cols = len(grid), len(grid[0])
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # exactly one step, 4-dir


def dijkstra_optimal_cost(grid, start, end):
    """Independent Dijkstra oracle used to verify A* optimality."""
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == end:
            return d
        if d > dist.get((r, c), float("inf")):
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                nd = d + grid[nr][nc]
                if nd < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return None


# ---------- tests ----------

def test_simple_path_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))

    assert_valid_path(grid, (0, 0), (2, 2), path)
    assert len(path) == 5  # Manhattan distance + 1 cell
    assert path_cost(grid, path) == 4
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, (0, 0), (2, 2))


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],   # wall blocks the straight line
        [1, 1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 3))

    assert_valid_path(grid, (0, 0), (2, 3), path)
    assert path_cost(grid, path) == 5
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, (0, 0), (2, 3))


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [1, 100, 100],
        [1, 100, 1],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))

    assert_valid_path(grid, (0, 0), (2, 2), path)
    assert all(grid[r][c] < 100 for r, c in path)  # avoids the expensive cells
    assert path_cost(grid, path) == 4
    assert path_cost(grid, path) == dijkstra_optimal_cost(grid, (0, 0), (2, 2))


def test_no_path_exists_fully_blocked():
    grid = [
        [1, 1, 1],
        [0, 0, 0],   # complete wall barrier
        [1, 1, 1],
    ]
    a = AStarGrid(grid)

    assert a.find_path((0, 0), (2, 2)) is None
    assert a.find_path((0, 0), (2, 0)) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))

    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    a = AStarGrid(grid)

    with pytest.raises(ValueError):
        a.find_path((-1, 0), (2, 2))   # start row out of bounds (negative)
    with pytest.raises(ValueError):
        a.find_path((0, 0), (0, 3))    # end column out of bounds
    with pytest.raises(ValueError):
        a.find_path((3, 0), (0, 0))    # start row out of bounds (too large)
    with pytest.raises(ValueError):
        a.find_path((0, 1), (2, -1))   # end column out of bounds (negative)


def test_wall_endpoints_return_none():
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    a = AStarGrid(grid)

    assert a.find_path((0, 1), (2, 2)) is None   # start is a wall
    assert a.find_path((0, 0), (0, 1)) is None   # end is a wall
```

Run with:

```bash
pytest test_a_star.py -v
```

**Design notes**

- **Admissibility → optimality:** Manhattan distance counts steps, and every cell costs ≥ 1, so the heuristic never overestimates the true remaining cost. With positive costs and this heuristic, A* returns a minimum-cost path. Optimality is also cross-checked in the tests against an independent Dijkstra implementation.
- **Path cost convention:** total cost = sum of entering-costs of cells after the start (start is "free"), matching the `g_scores` bookkeeping.
- **Stale entries:** the heap is lazily cleaned — a cell popped after being added to `closed` is skipped, which is safe because the Manhattan heuristic is consistent.
- **Edge-case ordering in `find_path`:** bounds check (raises `ValueError`) → wall check (returns `None`) → start == end (returns `[start]`). So a wall start/end returns `None` even if start == end.
- The wall-endpoint test is a 7th test beyond the 6 requested — the spec's 6 are the first six, plus one to pin down the wall-endpoint behavior.