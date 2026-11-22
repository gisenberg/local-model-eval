Here's a complete implementation. I split it into two files: `astar.py` (implementation) and `test_astar.py` (tests).

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid with 4-directional movement."""

from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """Find minimum-cost paths on a weighted grid.

    Grid values are movement costs: 0 means impassable wall, any positive
    integer is the cost to enter that cell. Movement is 4-directional
    (up/down/left/right, no diagonals).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store and validate the grid.

        Args:
            grid: 2D list where each cell is 0 (wall) or a positive int (cost).

        Raises:
            ValueError: If the grid is empty, ragged, or contains invalid cells.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty.")

        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("Grid must be rectangular: all rows must have the same length.")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("Each cell must be a non-negative int (0 = wall, >0 = cost).")

        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = width

    def _validate_point(self, name: str, point) -> Point:
        """Return `point` as a validated (row, col) tuple.

        Raises:
            ValueError: If the point is not a (row, col) pair or is out of bounds.
        """
        try:
            row, col = point
        except Exception:
            raise ValueError(f"{name} must be a (row, col) pair, got {point!r}.") from None
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError(f"{name} coordinates must be integers, got {point!r}.")
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"{name} {(row, col)} is out of bounds for a {self.rows}x{self.cols} grid."
            )
        return (row, col)

    def _heuristic(self, a: Point, b: Point) -> int:
        """Manhattan distance between cells a and b (admissible lower bound)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, cell: Point) -> List[Point]:
        """Return the passable, in-bounds 4-directional neighbors of `cell`."""
        r, c = cell
        neighbors: List[Point] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Point]]:
        """Find the minimum-cost path from start to end.

        Args:
            start: (row, col) start cell.
            end: (row, col) goal cell.

        Returns:
            The path as a list of (row, col) coordinates from start to end,
            inclusive, or None if start or end is a wall, or no path exists.

        Raises:
            ValueError: If start or end is out of bounds or malformed.

        Notes:
            Path cost is the sum of grid values of the cells entered after the
            start cell. Manhattan distance is an admissible (and consistent)
            heuristic for all positive integer costs, so the returned path is
            optimal. If start == end and that cell is passable, [start] is returned.
        """
        start = self._validate_point("start", start)
        end = self._validate_point("end", end)

        # Walls are never passable (even if start == end).
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Trivial case.
        if start == end:
            return [start]

        # A* search.
        counter = 0
        open_heap: List[Tuple[float, int, Point]] = [
            (self._heuristic(start, end), counter, start)
        ]
        g_score: Dict[Point, int] = {start: 0}
        came_from: Dict[Point, Point] = {}
        closed: Set[Point] = set()

        while open_heap:
            _, _, current = heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == end:
                path = [current]
                while current != start:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heappush(open_heap, (f_score, counter, neighbor))

        return None
```

## `test_astar.py`

```python
"""Pytest tests for astar.AStarGrid."""

import heapq
from typing import Dict, List, Optional, Tuple

import pytest

from astar import AStarGrid

Point = Tuple[int, int]


def path_cost(grid: List[List[int]], path: List[Point]) -> int:
    """Total cost of a path (sum of costs of cells entered after the start)."""
    return sum(grid[r][c] for r, c in path[1:])


def is_valid_path(grid: List[List[int]], path: List[Point], start: Point, end: Point) -> bool:
    """Check endpoints, bounds, passability, and 4-connectivity of the path."""
    if not path or path[0] != start or path[-1] != end:
        return False
    for r, c in path:
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False
        if grid[r][c] <= 0:
            return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    return True


def dijkstra_cost(grid: List[List[int]], start: Point, end: Point) -> Optional[int]:
    """Independent Dijkstra reference implementation used to verify optimality."""
    rows, cols = len(grid), len(grid[0])
    dist: Dict[Point, int] = {start: 0}
    heap: List[Tuple[int, Point]] = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        if node == end:
            return d
        r, c = node
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                nd = d + grid[nr][nc]
                if nd < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    return None


def test_simple_path_uniform_grid() -> None:
    grid = [[1] * 5 for _ in range(5)]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 4))
    assert path is not None
    assert is_valid_path(grid, path, (0, 0), (4, 4))
    assert len(path) == 9  # Manhattan distance (8) + 1
    assert path_cost(grid, path) == 8
    assert path_cost(grid, path) == dijkstra_cost(grid, (0, 0), (4, 4))


def test_path_around_obstacles() -> None:
    grid = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 4)
    astar = AStarGrid(grid)
    path = astar.find_path(start, end)
    assert path is not None
    assert is_valid_path(grid, path, start, end)
    assert len(path) > 5  # longer than the straight-line path (Manhattan + 1)
    assert path_cost(grid, path) == 12
    assert path_cost(grid, path) == dijkstra_cost(grid, start, end)


def test_weighted_grid_prefers_low_cost_cells() -> None:
    grid = [
        [1, 100, 100, 1],
        [1, 100, 100, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 3)
    astar = AStarGrid(grid)
    path = astar.find_path(start, end)
    assert path is not None
    assert is_valid_path(grid, path, start, end)
    # Going straight across the top costs 100 + 100 + 1 = 201; the cheap
    # detour along the bottom row costs 7.
    assert path_cost(grid, path) == 7
    assert path_cost(grid, path) == dijkstra_cost(grid, start, end)
    assert all(grid[r][c] < 100 for r, c in path)


def test_no_path_exists_fully_blocked() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None  # fully blocked
    assert astar.find_path((0, 1), (0, 2)) is None  # start is a wall
    assert astar.find_path((0, 0), (0, 1)) is None  # end is a wall


def test_start_equals_end() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))   # negative start row
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))    # end row out of range
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 3))    # end col out of range
    with pytest.raises(ValueError):
        astar.find_path((2, 2), (5, 5))    # end fully out of range
```

## How to run

```bash
python -m pytest test_astar.py -v
```

## Design notes

- **Optimality**: Manhattan distance is admissible (and consistent) whenever all cell costs are ≥ 1, so A* with `heapq` and a `closed` set returns a minimum-total-cost path.
- **Path cost convention**: sum of grid values of the cells *entered* after the start cell (start costs nothing). The tests use this same convention, plus an independent Dijkstra implementation as a cross-check for optimality.
- **Edge case precedence** in `find_path`: bounds validation (raises `ValueError`) → wall check (returns `None`) → `start == end` (returns `[start]`) → A* search. Note this means `start == end` on a wall returns `None`, since the wall check runs first.
- **Open set**: entries are `(f_score, counter, node)` tuples; the incrementing `counter` guarantees uniqueness so nodes are never compared directly.
- `__init__` validates that the grid is non-empty, rectangular, and contains only non-negative ints, and copies the grid so external mutation can't corrupt the search.

I haven't executed the code here, so treat the tests as written-but-unrun; they're straightforward enough that they should pass, but do run pytest to confirm. If you'd like, I can also add a `path_cost()` helper to the class itself, or a bonus `__main__` demo block.