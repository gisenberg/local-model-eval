# A* Pathfinding on a Weighted 2D Grid

Here's a complete implementation split into two files.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
from typing import Dict, Iterator, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """A* solver for a weighted 2D grid.

    Grid semantics:
        0        -> impassable wall
        positive -> cost to ENTER that cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store the grid and validate its shape.

        Args:
            grid: 2D list of non-negative ints (0 = wall, > 0 = cost).

        Raises:
            ValueError: If the grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be a non-empty rectangular 2D list")
        cols = len(grid[0])
        for row in grid:
            if len(row) != cols:
                raise ValueError("Grid must be rectangular: all rows must have equal length")
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = cols

    def _heuristic(self, a: Point, b: Point) -> int:
        """Return the Manhattan distance between two cells.

        Admissible (and consistent) because every enterable cell costs at
        least 1, so no real path can cost less than the step count.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _walkable_neighbors(self, cell: Point) -> Iterator[Point]:
        """Yield the in-bounds, non-wall 4-directional neighbors of `cell`."""
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                yield (nr, nc)

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find a minimum-total-cost path from `start` to `end`.

        The total cost of a path is the sum of the grid values of every cell
        AFTER the first (you start on the start cell for free).

        Args:
            start: (row, col) start cell.
            end: (row, col) target cell.

        Returns:
            List of (row, col) cells from `start` to `end` inclusive, or None
            if no path exists or either endpoint is a wall.

        Raises:
            ValueError: If `start` or `end` is out of bounds.

        Note:
            Check order is bounds -> wall -> start == end, so a wall cell
            never yields a "valid" single-cell path.
        """
        # Bounds check first — coordinates must be valid before anything else.
        for label, (r, c) in (("start", start), ("end", end)):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(
                    f"{label} {(r, c)} is out of bounds for grid of size "
                    f"{self.rows}x{self.cols}"
                )

        # Wall endpoints -> no path (checked before start == end shortcut).
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}
        closed: Set[Point] = set()
        counter = 0  # tie-breaker so heap entries are never compared by node
        heap: List[Tuple[int, int, Point]] = [(self._heuristic(start, end), counter, start)]
        counter += 1

        while heap:
            _, _, current = heapq.heappop(heap)
            if current in closed:
                continue
            if current == end:
                path = [current]
                node = current
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                return path
            closed.add(current)

            for neighbor in self._walkable_neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(heap, (f_score, counter, neighbor))
                    counter += 1

        return None
```

## `test_astar.py`

```python
"""Tests for AStarGrid."""

from typing import List, Optional, Tuple

import pytest

from astar import AStarGrid

Point = Tuple[int, int]


def path_cost(grid: List[List[int]], path: List[Point]) -> int:
    """Total cost of a path: sum of grid values of all cells except the start."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid: List[List[int]], start: Point, end: Point,
                      path: Optional[List[Point]]) -> None:
    """Assert the path is a walkable, 4-connected route from start to end."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    rows, cols = len(grid), len(grid[0])
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0, f"cell {(r, c)} is impassable"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "non-adjacent step"


def test_simple_uniform_grid() -> None:
    grid = [[1] * 3 for _ in range(3)]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert_valid_path(grid, (0, 0), (2, 2), path)
    assert len(path) == 5                       # 4 steps + start cell
    assert path_cost(grid, path) == 4           # = Manhattan distance, optimal


def test_path_around_obstacles() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert_valid_path(grid, (0, 0), (0, 2), path)
    assert path_cost(grid, path) == 6           # forced detour around wall column


def test_weighted_grid_prefers_cheaper_cells() -> None:
    grid = [
        [1, 5, 5],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert_valid_path(grid, (0, 0), (0, 2), path)
    # Top row costs 5+5=10; bottom row costs 1+1+1+5=8 -> must pick bottom.
    assert path_cost(grid, path) == 8
    assert (1, 1) in path


def test_no_path_exists() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert AStarGrid(grid).find_path((0, 0), (2, 2)) is None


def test_start_equals_end() -> None:
    grid = [[1] * 3 for _ in range(3)]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


@pytest.mark.parametrize("start,end", [
    ((-1, 0), (2, 2)),   # start row negative
    ((3, 0), (2, 2)),    # start row out of bounds
    ((0, 0), (-1, 0)),   # end row negative
    ((0, 0), (2, 3)),    # end column out of bounds
])
def test_invalid_coordinates_raise(start: Point, end: Point) -> None:
    grid = [[1] * 3 for _ in range(3)]
    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)


def test_wall_endpoints_return_none() -> None:
    grid = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ]
    assert AStarGrid(grid).find_path((0, 0), (1, 1)) is None  # wall start
    assert AStarGrid(grid).find_path((1, 1), (0, 0)) is None  # wall end
    assert AStarGrid(grid).find_path((1, 1), (2, 2)) is None  # wall end
```

## Design notes

- **Optimality:** Manhattan distance is both admissible and consistent here because every enterable cell costs ≥ 1 (positive ints), so the closed set is safe and the first time `end` is popped from the heap, the path is provably minimal.
- **Heap tie-breaking:** entries are `(f_score, counter, node)`; the counter guarantees uniqueness so nodes are never compared, and equal-f ties are broken arbitrarily.
- **Cost definition:** entering the start cell is free; the path cost is the sum of grid values of cells after the first. Tests assert exact total costs (e.g. 8 for the weighted grid), not just validity.
- **Edge-case ordering:** bounds → wall → start == end, so a wall cell never yields a bogus one-cell "path".
- **Complexity:** roughly O((V + E) log V) with a binary heap, where V = number of walkable cells.

Run with:

```bash
pytest test_astar.py -v
```

Expected: 10 tests passing (6 required coverage areas, with the invalid-coordinates test parametrized into 4 cases and one extra wall-endpoint test).