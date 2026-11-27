# A* Pathfinding Implementation

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class AStarGrid:
    """Finds the lowest-cost path between two cells on a weighted grid.

    Attributes:
        grid: 2D list where grid[r][c] is the cost to enter cell (r, c).
              0 means impassable (wall); positive ints are costs >= 1.
        rows: Number of rows in the grid.
        cols: Number of columns in the first row of the grid.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list of cell costs (0 = wall, positive int = cost to enter).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0

    def find_path(self, start: Cell, end: Cell) -> Optional[List[Cell]]:
        """Find the lowest-cost path from start to end.

        The Manhattan heuristic is admissible and consistent for this grid
        (all cell costs >= 1), so the returned path is guaranteed optimal.

        Args:
            start: (row, col) starting cell.
            end: (row, col) target cell.

        Returns:
            List of (row, col) cells from start to end inclusive, or None if
            no path exists (e.g. an endpoint is a wall or unreachable).

        Raises:
            ValueError: If start or end is outside the grid.

        Note on precedence: bounds errors raise first, then wall endpoints
        return None, then start == end returns [start].
        """
        sr, sc = start
        er, ec = end
        if not self._in_bounds(sr, sc) or not self._in_bounds(er, ec):
            raise ValueError(
                f"start {start} and end {end} must be within grid bounds "
                f"(grid is {self.rows} x {self.cols})"
            )

        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, Cell]] = [(0 + self._heuristic(start, end), start)]
        g_score: Dict[Cell, int] = {start: 0}
        parent: Dict[Cell, Cell] = {}
        closed: Set[Cell] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:  # stale heap entry
                continue
            if current == end:
                return self._reconstruct_path(parent, start, end)
            closed.add(current)

            cr, cc = current
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if not self._in_bounds(nr, nc):
                    continue
                step_cost = self.grid[nr][nc]
                if step_cost == 0:
                    continue
                neighbor = (nr, nc)
                new_g = g_score[current] + step_cost
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    parent[neighbor] = current
                    f_score = new_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return None

    def _in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid."""
        return 0 <= row < self.rows and 0 <= col < len(self.grid[row])

    def _heuristic(self, cell: Cell, end: Cell) -> int:
        """Return the Manhattan distance from cell to end."""
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    def _reconstruct_path(self, parent: Dict[Cell, Cell], start: Cell, end: Cell) -> List[Cell]:
        """Reconstruct the path from end back to start using the parent map."""
        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path
```

**Why this is optimal:** all cell costs are ≥ 1, so the cheapest possible continuation from any cell is at least the number of steps remaining — exactly the Manhattan distance. The heuristic is admissible and consistent, which guarantees A* returns a minimum-cost path.

## `test_astar.py`

```python
"""Tests for AStarGrid."""

import heapq
import pytest

from astar import AStarGrid


def total_cost(grid, path):
    """Sum of cell costs along a path, excluding the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, path, start, end):
    """Assert path is adjacent, in-bounds, wall-free, and connects start to end."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    rows, cols = len(grid), len(grid[0])
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def brute_force_optimal_cost(grid, start, end):
    """Independent check: Dijkstra over the grid (no heuristic)."""
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
    return None


def test_simple_path_uniform_grid():
    grid = [[1] * 4 for _ in range(4)]
    start, end = (0, 0), (3, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    assert total_cost(grid, path) == 6
    assert total_cost(grid, path) == brute_force_optimal_cost(grid, start, end)


def test_path_around_obstacles():
    grid = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 4)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    assert total_cost(grid, path) == 10
    assert total_cost(grid, path) == brute_force_optimal_cost(grid, start, end)


def test_weighted_grid_prefers_low_cost_cells():
    grid = [
        [1, 5, 1],
        [1, 5, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    assert total_cost(grid, path) == 4
    assert total_cost(grid, path) == brute_force_optimal_cost(grid, start, end)
    assert all(grid[r][c] < 5 for r, c in path)  # routed around the expensive column


def test_no_path_exists():
    grid = [
        [1, 1, 1],
        [0, 0, 0],  # fully blocked middle row
        [1, 1, 1],
    ]
    assert AStarGrid(grid).find_path((0, 0), (2, 2)) is None


def test_start_equals_end():
    grid = [[1] * 3 for _ in range(3)]
    start = (1, 1)
    path = AStarGrid(grid).find_path(start, start)
    assert path == [start]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates():
    astar = AStarGrid([[1] * 3 for _ in range(3)])
    bad_cases = [
        ((-1, 0), (2, 2)),   # start row out of bounds
        ((0, -1), (2, 2)),   # start col out of bounds
        ((0, 0), (3, 0)),    # end row out of bounds
        ((0, 0), (0, 3)),    # end col out of bounds
        ((5, 5), (5, 5)),    # both out of bounds (start == end, but still invalid)
    ]
    for bad_start, bad_end in bad_cases:
        with pytest.raises(ValueError):
            astar.find_path(bad_start, bad_end)


def test_wall_endpoints_return_none():
    grid = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 1)) is None   # wall start
    assert astar.find_path((1, 1), (2, 2)) is None   # wall end
    assert astar.find_path((0, 0), (2, 2)) is None   # both walls
```

## Run the tests

```bash
python -m pytest test_astar.py -v
```

**Design notes:**

- **Heap**: open set is a `heapq` min-heap of `(f_score, cell)`; stale entries are discarded via a `closed` set (lazy deletion).
- **Optimality check**: each test also compares A*'s total cost against an independent Dijkstra-based `brute_force_optimal_cost`, so the cost assertion is verified two ways.
- **Edge-case precedence**: `ValueError` for out-of-bounds (can't even index the cell) → `None` for wall endpoints → `[start]` for `start == end` (only reached for valid, non-wall cells).
- The weighted-grid test deliberately places the direct route through high-cost cells (5s) so the optimal path (cost 4, via the bottom row) differs from the Manhattan-shortest route.