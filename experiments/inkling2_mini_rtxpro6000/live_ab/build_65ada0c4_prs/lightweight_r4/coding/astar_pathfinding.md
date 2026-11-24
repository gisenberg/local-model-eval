Here's a complete implementation split into two files: `astar_grid.py` (the implementation) and `test_astar_grid.py` (the tests).

## `astar_grid.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, List, Optional, Tuple


class AStarGrid:
    """Find the cheapest 4-directional path across a weighted grid.

    Cells are indexed ``grid[row][col]``:
        0             -> impassable wall
        positive int  -> cost to enter that cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Validate and store the grid.

        Args:
            grid: 2-D list of ints; 0 = wall, positive int = cost to enter.

        Raises:
            ValueError: if the grid is empty, ragged, non-integer, or negative.
        """
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError("grid must be a non-empty list of rows")
        if not isinstance(grid[0], list) or len(grid[0]) == 0:
            raise ValueError("grid must contain at least one column")
        for row in grid:
            if not isinstance(row, list) or len(row) != len(grid[0]):
                raise ValueError("grid must be rectangular (all rows equal length)")
            for value in row:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("grid values must be integers")
                if value < 0:
                    raise ValueError("grid values must be >= 0 (0 = wall)")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the lowest-cost 4-directional path from start to end.

        Args:
            start: (row, col) of the start cell (must be in bounds).
            end: (row, col) of the end cell (must be in bounds).

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if either endpoint is a wall or no path exists.

        Raises:
            ValueError: if either endpoint is out of bounds or not a (row, col) pair.
        """
        try:
            start = tuple(start)
            end = tuple(end)
        except (TypeError, ValueError):
            raise ValueError("start and end must be (row, col) pairs") from None
        if len(start) != 2 or len(end) != 2:
            raise ValueError("start and end must be (row, col) pairs")
        self._check_bounds(start)
        self._check_bounds(end)

        # Wall check comes first: a wall endpoint is invalid even if start == end.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        def heuristic(r: int, c: int) -> int:
            # Manhattan distance; admissible because every move costs >= 1.
            return abs(r - end[0]) + abs(c - end[1])

        counter = 0  # unique tie-breaker so heap entries never compare coordinates
        open_heap: List[Tuple[int, int, int, int, int]] = []
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        heapq.heappush(open_heap, (heuristic(*start), counter, 0, start[0], start[1]))

        while open_heap:
            _, _, g, r, c = heapq.heappop(open_heap)
            if g > g_score.get((r, c), float("inf")):
                continue  # stale heap entry; a cheaper path was already found
            if (r, c) == end:
                return self._reconstruct_path(came_from, start, end)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                move_cost = self.grid[nr][nc]
                if move_cost == 0:
                    continue  # wall
                tentative_g = g + move_cost
                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    came_from[(nr, nc)] = (r, c)
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative_g + heuristic(nr, nc), counter, tentative_g, nr, nc),
                    )

        return None

    def _check_bounds(self, point: Tuple[int, int]) -> None:
        """Raise ValueError if point is not an in-bounds (row, col) pair."""
        row, col = point
        if (
            not isinstance(row, int) or isinstance(row, bool)
            or not isinstance(col, int) or isinstance(col, bool)
        ):
            raise ValueError(f"coordinates must be integers, got {point!r}")
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"coordinate {point} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Walk came_from backwards from end to start; return start -> end."""
        path = [end]
        node = end
        while node != start:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path
```

## `test_astar_grid.py`

The tests include a small independent reference Dijkstra implementation so the optimality assertion is a cross-check, not just a hard-coded number.

```python
"""Tests for A* pathfinding on a weighted 2D grid."""

import heapq
from math import inf
from typing import List, Optional, Tuple

import pytest

from astar_grid import AStarGrid

Point = Tuple[int, int]


def cell_cost(grid: List[List[int]], path: List[Point]) -> int:
    """Total cost of walking path: sum of enter-costs of cells after start."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_path_valid(grid: List[List[int]], path: Optional[List[Point]],
                      start: Point, end: Point) -> None:
    """Assert the path is a contiguous, non-repeating, in-bounds walk."""
    rows, cols = len(grid), len(grid[0])
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert len(set(path)) == len(path)
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # exactly one 4-directional step


def reference_dijkstra_cost(grid: List[List[int]], start: Point,
                            end: Point) -> Optional[int]:
    """Independent reference implementation (Dijkstra) for optimality checks."""
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist.get((r, c), inf):
            continue
        if (r, c) == end:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                nd = d + grid[nr][nc]
                if nd < dist.get((nr, nc), inf):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return None


def test_simple_path_on_uniform_grid() -> None:
    grid = [[1] * 4 for _ in range(4)]
    start, end = (0, 0), (3, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_path_valid(grid, path, start, end)
    assert path is not None
    assert cell_cost(grid, path) == reference_dijkstra_cost(grid, start, end)
    assert cell_cost(grid, path) == 6  # 3 rights + 3 downs, unit cost each


def test_path_around_obstacles() -> None:
    grid = [
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (3, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_path_valid(grid, path, start, end)
    assert path is not None
    assert cell_cost(grid, path) == reference_dijkstra_cost(grid, start, end)
    assert cell_cost(grid, path) == 6  # 6 unit-cost moves around the walls


def test_weighted_grid_prefers_lower_cost_cells() -> None:
    grid = [
        [1, 20, 20, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_path_valid(grid, path, start, end)
    assert path is not None
    assert cell_cost(grid, path) == reference_dijkstra_cost(grid, start, end)
    assert cell_cost(grid, path) == 5  # cheap bottom detour beats direct route (41)


def test_no_path_exists_fully_blocked() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    start, end = (0, 0), (0, 2)
    assert AStarGrid(grid).find_path(start, end) is None
    assert reference_dijkstra_cost(grid, start, end) is None


def test_start_equals_end() -> None:
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    start = end = (0, 0)
    path = AStarGrid(grid).find_path(start, end)
    assert path == [start]
    assert_path_valid(grid, path, start, end)
    assert cell_cost(grid, path) == reference_dijkstra_cost(grid, start, end) == 0


def test_invalid_coordinates_raise_value_error() -> None:
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))   # end row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))  # start row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 5), (1, 1))   # start col out of bounds
    with pytest.raises(ValueError):
        astar.find_path(("0", 0), (1, 1))  # non-integer coordinate


def test_wall_endpoint_returns_none() -> None:
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 1), (0, 0)) is None
    assert astar.find_path((0, 0), (0, 0)) is None  # wall start == end
```

Run with:

```bash
python -m pytest test_astar_grid.py -v
```

## Design notes

- **Cost semantics**: the path cost is the sum of enter-costs of every cell *after* the start (the start cell's own cost is never paid, matching standard A*/Dijkstra convention).
- **Heuristic admissibility**: Manhattan distance is admissible here because every move costs ≥ 1, so the true remaining cost is at least `|dr| + |dc|`. It's also consistent, which guarantees the first pop of the end node yields the optimal path.
- **Stale-entry handling**: heap entries carry `g`, and pops with a worse `g` than the recorded best are skipped — this keeps the open set simple and correct without a separate closed set.
- **Ordering of edge-case checks**: bounds are validated first (→ `ValueError`), then wall endpoints (→ `None`), then `start == end` (→ `[start]`). So a wall cell with `start == end` returns `None` rather than `[start]` — the wall rule is treated as a hard validity constraint. The `start == end` shortcut avoids running A* on a degenerate case.
- **Bonus coverage**: the 7th test covers the wall-endpoint rule explicitly.