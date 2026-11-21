# A* Pathfinding on a Weighted Grid

Below is a complete implementation (`a_star.py`) plus its test suite (`test_a_star.py`).

## `a_star.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement).

Grid values:
    0         -> impassable wall
    positive  -> cost to enter the cell
"""

import heapq
from typing import Dict, List, Optional, Tuple, Union


class AStarGrid:
    """A* pathfinding over a weighted 2D grid.

    Attributes:
        grid: Copy of the (rectangular) grid of cell costs.
        rows: Number of rows.
        cols: Number of columns.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: Non-empty, rectangular grid of non-negative ints
                  (0 = impassable wall, positive = cost to enter a cell).

        Raises:
            ValueError: If the grid is empty, ragged, or contains invalid values.
        """
        if not isinstance(grid, (list, tuple)) or len(grid) == 0:
            raise ValueError("grid must be a non-empty list of rows")
        cols = len(grid[0])
        if cols == 0:
            raise ValueError("grid must have at least one column")
        for row in grid:
            if not isinstance(row, (list, tuple)) or len(row) != cols:
                raise ValueError("grid must be rectangular (every row same length)")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("grid values must be integers")
                if value < 0:
                    raise ValueError("grid values must be >= 0 (0 is a wall)")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the minimum-cost path from start to end.

        Path cost = sum of entry costs of all cells AFTER the start cell.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the destination cell.

        Returns:
            Coordinates from start to end inclusive, or None if no path exists
            (wall at either endpoint, or no route through the grid).

        Raises:
            ValueError: If either coordinate is malformed or out of bounds.

        Note:
            Edge-case precedence: bounds (ValueError) -> wall (None) ->
            start == end ([start]). So a wall start == end returns None.
        """
        start = self._normalize_point(start, "start")
        end = self._normalize_point(end, "end")

        for name, (r, c) in (("start", start), ("end", end)):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(
                    f"{name} {(r, c)!r} is out of bounds for a {self.rows}x{self.cols} grid"
                )

        # An endpoint on a wall cannot be part of a path.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        end_r, end_c = end

        def heuristic(r: int, c: int) -> int:
            return abs(r - end_r) + abs(c - end_c)  # Manhattan distance

        g_best: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # heap entries: (f = g + h, g, cell)
        heap: List[Tuple[int, int, Tuple[int, int]]] = [
            (heuristic(start[0], start[1]), 0, start)
        ]

        while heap:
            _, g, current = heapq.heappop(heap)

            # Skip stale entries superseded by a better g-score.
            if g > g_best.get(current, float("inf")):
                continue

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            r, c = current
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # 4-directional
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                step_cost = self.grid[nr][nc]
                if step_cost == 0:
                    continue
                new_g = g + step_cost
                if new_g < g_best.get((nr, nc), float("inf")):
                    g_best[(nr, nc)] = new_g
                    came_from[(nr, nc)] = current
                    heapq.heappush(heap, (new_g + heuristic(nr, nc), new_g, (nr, nc)))

        return None

    @staticmethod
    def _normalize_point(
        point: Union[List[int], Tuple[int, int]], name: str
    ) -> Tuple[int, int]:
        """Validate a coordinate and return it as a (row, col) tuple."""
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} must be a (row, col) pair, got {point!r}")
        r, c = point
        if (
            isinstance(r, bool) or isinstance(c, bool)
            or not isinstance(r, int) or not isinstance(c, int)
        ):
            raise ValueError(f"{name} coordinates must be integers, got {point!r}")
        return r, c

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Rebuild the start-to-end path by following the came_from chain."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_a_star.py`

```python
"""Tests for AStarGrid."""

import pytest

from a_star import AStarGrid


def path_cost(grid, path):
    """Total entry cost of a path, excluding the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, path, start, end):
    """Check endpoints, adjacency (4-directional), and non-wall cells."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert grid[start[0]][start[1]] > 0
    assert grid[end[0]][end[1]] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, f"non-adjacent step {(r1, c1)} -> {(r2, c2)}"
        assert grid[r2][c2] > 0


def test_simple_path_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    # Manhattan distance is 4 steps; every cell costs 1 -> cost 4 is optimal.
    assert path_cost(grid, path) == 4


def test_path_around_obstacles():
    grid = [
        [1, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 3))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 3))
    # Wall column forces a detour: 2 down + 3 right + 2 up = 7 steps, cost 7.
    assert path_cost(grid, path) == 7


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [9, 9, 9, 9],
        [1, 1, 1, 1],
        [9, 9, 9, 9],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 3))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 3))
    # Straight row-0 path costs 27; the cheap-row detour costs 13.
    assert path_cost(grid, path) == 13


def test_no_path_exists_fully_blocked():
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
    grid_obj = AStarGrid(grid)
    assert grid_obj.find_path((0, 0), (0, 2)) is None  # wall column blocks everything
    assert grid_obj.find_path((0, 0), (1, 1)) is None  # end is a wall
    assert grid_obj.find_path((1, 1), (0, 2)) is None  # start is a wall


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert AStarGrid(grid).find_path((1, 1), (1, 1)) == [(1, 1)]


def test_invalid_coordinates():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    with pytest.raises(ValueError):
        grid_obj.find_path((-1, 0), (2, 2))   # start row out of bounds
    with pytest.raises(ValueError):
        grid_obj.find_path((0, 0), (3, 0))    # end row out of bounds
    with pytest.raises(ValueError):
        grid_obj.find_path((0, 3), (0, 0))    # start col out of bounds
    with pytest.raises(ValueError):
        grid_obj.find_path((0, 0, 0), (1, 1))  # malformed point
```

Run with:

```bash
python -m pytest test_a_star.py -v
```

## Design notes

- **Optimality**: Manhattan distance is admissible (every cell costs ≥ 1, and any cell-to-goal route needs at least Manhattan-many steps) and consistent, so A* with a `(f = g + h)` min-heap returns a minimum-cost path. Stale heap entries are skipped via the `g_best` comparison rather than requiring a closed set.
- **Cost convention**: the path cost is the sum of entry costs of all cells *after* the start (you're already at the start). This makes `start == end` cost 0.
- **Edge-case precedence**: bounds (`ValueError`) → wall endpoint (`None`) → `start == end` (`[start]`). Note this means a wall `start == end` returns `None`, since the cell is impassable — flip the two checks in `find_path` if you'd rather return `[start]` there.
- **Tests assert optimality** by pinning exact total costs (hand-derived lower bounds), and assert validity (endpoints, 4-directional adjacency, non-wall cells) via `assert_valid_path`.