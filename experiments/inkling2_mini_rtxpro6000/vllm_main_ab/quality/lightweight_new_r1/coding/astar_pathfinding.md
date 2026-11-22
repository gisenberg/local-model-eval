Here's a complete implementation, split into the solver and a test module.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted grid with 4-directional movement.

    Grid semantics:
        - grid[r][c] == 0  -> impassable wall
        - grid[r][c] > 0   -> cost to enter cell (r, c)
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Validate and store the grid.

        Args:
            grid: Rectangular 2D list of non-negative ints.

        Raises:
            ValueError: If the grid is empty, not rectangular, or contains
                a negative value.
        """
        if not isinstance(grid, list) or not grid:
            raise ValueError("grid must be a non-empty list of rows")
        if not isinstance(grid[0], list) or not grid[0]:
            raise ValueError("grid must be a non-empty list of rows")
        width = len(grid[0])
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("grid must be rectangular (all rows the same length)")
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("grid values must be non-negative ints (0 = wall)")
        self.grid = grid
        self.rows = len(grid)
        self.cols = width

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find the lowest-cost path from start to end.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the goal cell.

        Returns:
            The path as a list of (row, col) cells from start to end
            inclusive, or None if no path exists. Total cost = sum of the
            grid values of every cell entered after start.

        Raises:
            ValueError: If start or end is outside the grid.

        Edge-case precedence: bounds are checked first (ValueError), then
        wall cells (None), then the trivial start == end case.
        """
        self._require_in_bounds(start, "start")
        self._require_in_bounds(end, "end")

        sr, sc = start
        er, ec = end

        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None

        if start == end:
            return [start]

        # Heap entries: (f_score, tiebreaker, g_score, cell)
        open_heap: List[Tuple[int, int, int, Point]] = []
        counter = 0
        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}
        heappush(open_heap, (self._heuristic(start, end), counter, 0, start))
        counter += 1

        while open_heap:
            _, _, g, current = heappop(open_heap)
            if g > g_score.get(current, float("inf")):
                continue  # stale heap entry; a better g was found later
            if current == end:
                return self._reconstruct(came_from, start, end)
            r, c = current
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                enter_cost = self.grid[nr][nc]
                if enter_cost == 0:
                    continue
                tentative_g = g + enter_cost
                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    came_from[(nr, nc)] = current
                    counter += 1
                    heappush(open_heap, (
                        tentative_g + self._heuristic((nr, nc), end),
                        counter,
                        tentative_g,
                        (nr, nc),
                    ))
        return None

    def _heuristic(self, cell: Point, goal: Point) -> int:
        """Manhattan distance from cell to goal (admissible since each step costs >= 1)."""
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    def _reconstruct(self, came_from: Dict[Point, Point], start: Point, end: Point) -> List[Point]:
        """Walk came_from back from end to start; return the path (start first)."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _require_in_bounds(self, point: Point, name: str) -> None:
        """Raise ValueError if point lies outside the grid."""
        r, c = point
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"{name} {point} is out of bounds for a {self.rows}x{self.cols} grid"
            )
```

## `test_astar.py`

```python
"""Tests for AStarGrid. Run with: pytest test_astar.py -v"""

import pytest

from astar import AStarGrid


def _total_cost(grid, path):
    """Total cost of a path = sum of cell costs entered after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def _assert_valid_path(grid, path, start, end):
    """Check endpoints, bounds, non-wall cells, and 4-directional adjacency."""
    rows, cols = len(grid), len(grid[0])
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert abs(r1 - r0) + abs(c1 - c0) == 1


def test_simple_path_on_uniform_grid():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    _assert_valid_path(grid, path, start, end)
    # Manhattan distance is 4; every cell costs 1, so optimal cost = 4
    assert _total_cost(grid, path) == 4


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (3, 3)
    path = AStarGrid(grid).find_path(start, end)
    _assert_valid_path(grid, path, start, end)
    # Manhattan distance is 6 and every cell costs 1, so optimal cost = 6
    assert _total_cost(grid, path) == 6


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [
        [1, 8, 8, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 3)
    path = AStarGrid(grid).find_path(start, end)
    _assert_valid_path(grid, path, start, end)
    # Straight along row 0 costs 17; detouring through row 1 costs 5
    assert _total_cost(grid, path) == 5


def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    start, end = (0, 0), (0, 2)
    assert AStarGrid(grid).find_path(start, end) is None


def test_start_equals_end():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    start = (1, 1)
    path = AStarGrid(grid).find_path(start, start)
    assert path == [start]
    _assert_valid_path(grid, path, start, start)
    assert _total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    cases = [
        ((-1, 0), (0, 0)),  # negative row in start
        ((0, 0), (3, 0)),   # row beyond the bottom
        ((0, 0), (0, 3)),   # col beyond the right
        ((2, 2), (2, -1)),  # negative col in end
    ]
    for start, end in cases:
        with pytest.raises(ValueError):
            AStarGrid(grid).find_path(start, end)
```

## Design notes

- **Optimality**: Every non-wall cell costs ≥ 1, so Manhattan distance never overestimates the true remaining cost (admissible), and it's also consistent (moving one step changes Manhattan distance by exactly 1, and each step costs at least 1). Together with returning the goal as soon as it's popped from the heap, this guarantees a minimum-cost path.
- **Cost convention**: The start cell is entered for free; total path cost is the sum of `grid[r][c]` over all cells entered after the start, matching the A* `g`-scores.
- **Open set**: `heapq` min-heap keyed on `(f, tiebreaker, g, cell)`. A `counter` tiebreaker keeps entries comparable, and stale entries are skipped by comparing the popped `g` against the best-known `g`.
- **Edge-case precedence**: out-of-bounds → `ValueError`; wall start/end → `None`; then `start == end` → `[start]`. So `start == end` on a wall cell returns `None` (wall check wins).
- **No re-open problem**: the consistent heuristic makes a simple "skip stale entries" scheme sufficient — no closed set is required.

Expected result: `pytest test_astar.py -v` → **6 passed**. One caveat to be aware of: the optimality assertions in the tests are hardcoded against the expected costs for those specific grids (e.g., the weighted grid's optimal detour costs 5 vs. 17 direct), so if you modify the test grids you'll want to re-derive the expected costs.