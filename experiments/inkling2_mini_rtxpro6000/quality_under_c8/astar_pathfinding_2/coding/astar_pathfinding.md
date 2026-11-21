Here's a complete implementation with tests.

## Implementation: `astar.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted 2D grid.

    Grid semantics:
        0            -> impassable wall
        positive int -> cost to enter that cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the A* solver with a 2D grid.

        Args:
            grid: Rectangular 2D list of non-negative ints (0 = wall).

        Raises:
            ValueError: If the grid is empty, non-rectangular, or contains
                negative or non-integer values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty")
        cols = len(grid[0])
        for row in grid:
            if len(row) != cols:
                raise ValueError("Grid must be rectangular")
            for cell in row:
                if not isinstance(cell, int):
                    raise ValueError("Grid values must be integers")
                if cell < 0:
                    raise ValueError("Grid values must be >= 0")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(grid)
        self.cols: int = cols

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find a shortest path from start to end.

        Args:
            start: (row, col) starting cell.
            end: (row, col) goal cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive
            (minimal total cost), or None if no path exists.
            Returns [start] if start == end.
            Returns None if start or end is a wall.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        self._validate_point(start)
        self._validate_point(end)

        # Wall start/end are invalid even if start == end.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        counter = 0  # tie-breaker so heap never compares positions
        open_heap: List[Tuple[int, int, int, Point]] = [
            (self._heuristic(start, end), 0, counter, start)
        ]
        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}
        visited: Set[Point] = set()

        while open_heap:
            _, g, _, current = heapq.heappop(open_heap)

            if g > g_score.get(current, float("inf")):
                continue  # stale heap entry (a better g was found later)
            if current in visited:
                continue
            visited.add(current)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for neighbor in self._neighbors(current):
                cell_cost = self.grid[neighbor[0]][neighbor[1]]
                if cell_cost == 0 or neighbor in visited:
                    continue
                new_g = g + cell_cost
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = current
                    counter += 1
                    f = new_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f, new_g, counter, neighbor))

        return None

    def total_cost(self, path: List[Point]) -> int:
        """Return the total movement cost of a path.

        Args:
            path: List of (row, col) coordinates.

        Returns:
            Sum of grid costs for each cell entered after the start.
        """
        return sum(self.grid[r][c] for r, c in path[1:])

    def _neighbors(self, point: Point) -> List[Point]:
        """Return in-bounds 4-directional neighbors of a point."""
        r, c = point
        neighbors: List[Point] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Point, b: Point) -> int:
        """Manhattan distance between two cells.

        Admissible (and consistent) here because every cell costs >= 1
        per step.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _validate_point(self, point: Point) -> None:
        """Raise ValueError if point is not a valid in-bounds (row, col) pair."""
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or not isinstance(point[0], int)
            or not isinstance(point[1], int)
        ):
            raise ValueError(f"Point must be a (row, col) pair of ints, got {point!r}")
        row, col = point
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"Point {(row, col)} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    def _reconstruct_path(
        self,
        came_from: Dict[Point, Point],
        start: Point,
        end: Point,
    ) -> List[Point]:
        """Walk came_from back from end to start and return the path."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

## Tests: `test_astar.py`

```python
"""Pytest suite for AStarGrid."""

import pytest

from astar import AStarGrid


def path_cost(grid, path):
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Check start/end, 4-directional adjacency, and no wall cells."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert grid[start[0]][start[1]] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert 0 <= r2 < len(grid) and 0 <= c2 < len(grid[0])
        assert grid[r2][c2] > 0


def test_simple_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    start, end = (0, 0), (4, 4)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, start, end, path)
    assert len(path) == 9
    assert path_cost(grid, path) == 8  # optimal == Manhattan distance


def test_path_around_obstacles():
    grid = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 4)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, start, end, path)
    # forced detour around the wall: 4 down + 4 right + 4 up
    assert path_cost(grid, path) == 12


def test_weighted_grid_prefers_low_cost():
    grid = [
        [1, 1, 1, 1],
        [4, 4, 4, 1],
        [4, 4, 4, 1],
    ]
    start, end = (0, 0), (2, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, start, end, path)
    # only cells costing 1 can form the cheapest route (row 0, then column 3)
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    assert path_cost(grid, path) == 5


def test_no_path_fully_blocked():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    start, end = (0, 0), (0, 2)
    assert AStarGrid(grid).find_path(start, end) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start = end = (1, 1)
    path = AStarGrid(grid).find_path(start, end)
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


@pytest.mark.parametrize(
    "start,end",
    [
        ((-1, 0), (1, 1)),   # start row out of bounds
        ((0, 3), (1, 1)),    # start col out of bounds
        ((0, 0), (3, 0)),    # end row out of bounds
        ((0, 0), (1, -1)),   # end col out of bounds
    ],
)
def test_invalid_coordinates_raise(start, end):
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)


def test_wall_start_or_end_returns_none():
    grid = [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)
    assert solver.find_path((0, 0), (2, 2)) is None   # wall start
    assert solver.find_path((2, 2), (0, 0)) is None   # wall end
    assert solver.find_path((0, 0), (0, 0)) is None   # wall start == end
```

Run with:

```bash
pytest test_astar.py -v
```

## Design notes

- **Heuristic**: plain Manhattan distance, as requested. It's admissible (never overestimates) since every step costs ≥ 1, which guarantees optimality. It's also *consistent*, so the first time `end` is popped from the heap, the path is guaranteed minimal.
- **Open set**: `heapq` with entries `(f, g, counter, position)`. The counter guarantees uniqueness, so the heap never tries to compare `(row, col)` tuples. Stale entries (pushed with a worse `g`) are detected and skipped via the stored `g` vs. the best recorded `g_score`.
- **Edge-case precedence** in `find_path`: bounds check (`ValueError`) → wall check (`None`) → `start == end` (`[start]`). So `start == end` on a wall returns `None`, since a wall cell can't be a valid start.
- **Complexity**: each cell's best `g` is settled once; the heap grows with re-pushes, so worst-case is roughly `O(N log N)` for a `N`-cell grid, which is fine for typical grids.
- **Tests** assert path validity (start/end correct, 4-directional adjacency, no wall cells) and optimality (exact total cost), plus one extra test for wall start/end behavior beyond the six required cases.