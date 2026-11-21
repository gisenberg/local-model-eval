Here's a complete implementation with tests. Since all movement costs are ≥ 1, Manhattan distance is an admissible (and consistent) heuristic, so A* returns a guaranteed-optimal path.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]

# 4-directional offsets: up, down, left, right (no diagonals)
_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class AStarGrid:
    """A* solver for a weighted 2D grid.

    Grid values are movement costs: 0 marks an impassable wall, any positive
    int is the cost of entering that cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store a defensive copy of `grid`.

        Raises:
            ValueError: if the grid is empty, ragged, or contains values
                that are not non-negative ints.
        """
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
            raise ValueError("grid must be a non-empty rectangular list of lists")
        width = len(grid[0])
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("grid must be rectangular")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("grid cells must be non-negative ints (0 = wall)")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Return a lowest-cost path from `start` to `end`.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if no path exists (e.g. start/end is a wall, or the target
            is unreachable). If start == end, returns [start].

        Raises:
            ValueError: if either coordinate is malformed or out of bounds.
        """
        self._validate_coord(start, "start")
        self._validate_coord(end, "end")
        start = (start[0], start[1])
        end = (end[0], end[1])

        if start == end:
            return [start]

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # heap entries: (f_score, g_score, row, col)
        open_heap: List[Tuple[int, int, int, int]] = [
            (self._heuristic(start, end), 0, start[0], start[1])
        ]
        came_from: Dict[Point, Point] = {}
        g_score: Dict[Point, int] = {start: 0}

        while open_heap:
            _, g, row, col = heapq.heappop(open_heap)
            current = (row, col)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            best_known = g_score.get(current)
            if best_known is None or g > best_known:
                continue  # stale heap entry

            for d_row, d_col in _DIRECTIONS:
                n_row, n_col = row + d_row, col + d_col
                if not (0 <= n_row < self.rows and 0 <= n_col < self.cols):
                    continue
                move_cost = self.grid[n_row][n_col]
                if move_cost == 0:
                    continue
                neighbor = (n_row, n_col)
                tentative_g = g + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, n_row, n_col))

        return None

    def _validate_coord(self, coord: Tuple[int, int], name: str) -> None:
        """Raise ValueError unless `coord` is an int (row, col) pair inside the grid."""
        if (
            not isinstance(coord, (tuple, list))
            or len(coord) != 2
            or not all(isinstance(v, int) for v in coord)
        ):
            raise ValueError(f"{name} must be a (row, col) pair of ints, got {coord!r}")
        row, col = coord
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"{name} {coord!r} is out of bounds for a {self.rows} x {self.cols} grid"
            )

    @staticmethod
    def _heuristic(a: Point, b: Point) -> int:
        """Manhattan distance between two cells (admissible since all costs >= 1)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self, came_from: Dict[Point, Point], start: Point, end: Point
    ) -> List[Point]:
        """Walk `came_from` from `end` back to `start` and reverse the result."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Pytest tests for AStarGrid. Run with: pytest -v"""

import pytest

from astar import AStarGrid


# ---------- helpers ----------

def path_cost(grid, path):
    """Total cost of entering the cells of `path` after the first cell."""
    return sum(grid[r][c] for r, c in path[1:])


def brute_force_min_cost(grid, start, end):
    """Exhaustively find the lowest-cost simple path (None if unreachable)."""
    rows, cols = len(grid), len(grid[0])
    start, end = tuple(start), tuple(end)
    best = None
    if grid[start[0]][start[1]] == 0 or grid[end[0]][end[1]] == 0:
        return None

    def search(r, c, total, seen):
        nonlocal best
        if (r, c) == end:
            if best is None or total < best:
                best = total
            return
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = r + d_row, c + d_col
            if (0 <= n_row < rows and 0 <= n_col < cols
                    and grid[n_row][n_col] > 0 and (n_row, n_col) not in seen):
                seen.add((n_row, n_col))
                search(n_row, n_col, total + grid[n_row][n_col], seen)
                seen.remove((n_row, n_col))

    search(start[0], start[1], 0, {start})
    return best


def assert_valid_path(grid, path, start, end):
    """Assert `path` is a 4-connected, passable, non-repeating route start -> end."""
    rows, cols = len(grid), len(grid[0])
    start, end = tuple(start), tuple(end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    seen = set()
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0
        assert (r, c) not in seen
        seen.add((r, c))
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional step


def assert_optimal(grid, path, start, end, expected=None):
    """Assert the path is valid and its cost equals the brute-force optimum."""
    assert_valid_path(grid, path, start, end)
    cost = path_cost(grid, path)
    assert cost == brute_force_min_cost(grid, start, end)
    if expected is not None:
        assert cost == expected


# ---------- tests ----------

def test_simple_path_on_uniform_grid():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    # Optimal: Manhattan distance = 4 steps at cost 1 each
    assert_optimal(grid, path, (0, 0), (2, 2), expected=4)


def test_path_around_obstacles():
    grid = [[1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (3, 3))
    # Must detour around the blocked cells; still 6 steps (Manhattan bound)
    assert_optimal(grid, path, (0, 0), (3, 3), expected=6)


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [[1, 4, 1],
            [1, 4, 1],
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    # Optimal: along the bottom row, avoiding the expensive middle column
    assert_optimal(grid, path, (0, 0), (2, 2), expected=4)
    assert all(grid[r][c] != 4 for r, c in path)


def test_no_path_exists_returns_none():
    grid = [[1, 1, 1],
            [0, 0, 0],  # full horizontal barrier
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is None
    assert brute_force_min_cost(grid, (0, 0), (2, 2)) is None


def test_start_equals_end_returns_start_only():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    solver = AStarGrid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (2, 2))   # negative start row
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, 3))    # end column out of range
    with pytest.raises(ValueError):
        solver.find_path((3, 0), (0, 0))    # start row out of range
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, -1))   # negative end column


def test_wall_start_or_end_returns_none():
    grid = [[1, 0, 1],
            [1, 1, 0],
            [0, 1, 1]]
    solver = AStarGrid(grid)
    assert solver.find_path((0, 1), (2, 2)) is None  # start is a wall
    assert solver.find_path((2, 0), (0, 2)) is None  # start is a wall
    assert solver.find_path((2, 1), (1, 2)) is None  # end is a wall
    assert_optimal(grid, solver.find_path((0, 0), (2, 2)), (0, 0), (2, 2))
```

## Notes

- **Optimality**: Manhattan distance is admissible (and consistent) here because every passable cell costs ≥ 1 — moving one cell changes the Manhattan distance by at most 1, never more than the step cost. A* with a consistent heuristic returns a minimum-cost path, and the tests independently confirm this via brute-force enumeration of simple paths.
- **Edge-case order** in `find_path`: bounds validation (`ValueError`) → `start == end` (`[start]`) → wall check (`None`) → A*.
- **Heap**: entries are `(f, g, row, col)` tuples of ints, so comparisons are safe without a tiebreaker counter. Stale entries (pushed before a better `g` was found) are skipped via the `g > best_known` check.
- **Heuristic cost is uninformed by weights**, so on strongly weighted grids A\* may expand more cells than necessary — but it remains optimal. If you later need more guidance, the admissible upgrade is `manhattan * min_cost_in_grid`, but the spec calls for Manhattan, so it's kept as-is.

Run the tests with `pytest -v`.