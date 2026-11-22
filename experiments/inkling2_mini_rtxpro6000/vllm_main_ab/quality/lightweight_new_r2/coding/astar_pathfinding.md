Here's a complete implementation plus the test suite.

## `a_star_grid.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from itertools import count
from typing import Dict, Iterator, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class AStarGrid:
    """A* pathfinder over a weighted 2D grid.

    Grid values:
        0            -> impassable wall
        positive int -> cost to enter that cell

    Movement is 4-directional (no diagonals). The Manhattan heuristic is
    admissible (and consistent) under this convention, since every passable
    cell costs at least 1, so A* returns a minimum-cost path.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the pathfinder from a 2D grid.

        Args:
            grid: 2D list where 0 means impassable wall and a positive int
                means the cost to enter that cell.

        Raises:
            ValueError: If the grid is empty, not rectangular, or contains a
                non-integer or negative value.
        """
        if not grid or not grid[0]:
            raise ValueError("grid must be non-empty")
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("grid must be rectangular")
            for value in row:
                if not isinstance(value, int):
                    raise ValueError("grid values must be integers")
                if value < 0:
                    raise ValueError("grid values must be non-negative")

        self._grid: List[List[int]] = [list(row) for row in grid]
        self._rows: int = len(self._grid)
        self._cols: int = width

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return self._cols

    def find_path(self, start: Cell, end: Cell) -> Optional[List[Cell]]:
        """Find a minimum-cost path from start to end.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the goal cell.

        Returns:
            The list of (row, col) cells from start to end inclusive along a
            minimum-total-cost path, or None if no path exists or either
            endpoint is a wall cell (even when start == end).

        Raises:
            ValueError: If start or end is malformed or out of bounds.
        """
        self._validate_cell(start, "start")
        self._validate_cell(end, "end")

        if self._grid[start[0]][start[1]] == 0 or self._grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        counter = count()
        open_heap: List[Tuple[int, int, Cell]] = [
            (self._heuristic(start, end), next(counter), start)
        ]
        g_score: Dict[Cell, int] = {start: 0}
        came_from: Dict[Cell, Cell] = {}
        closed: Set[Cell] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for neighbor in self._neighbors(current):
                if neighbor in closed:
                    continue
                cost = self._grid[neighbor[0]][neighbor[1]]
                if cost == 0:
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, next(counter), neighbor))

        return None

    def _validate_cell(self, cell: Cell, name: str) -> None:
        """Validate that a coordinate is well-formed and inside the grid.

        Args:
            cell: (row, col) coordinate to validate.
            name: Parameter name used in error messages.

        Raises:
            ValueError: If the coordinate is malformed or out of bounds.
        """
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            raise ValueError(f"{name} must be a (row, col) pair, got {cell!r}")
        row, col = cell
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError(f"{name} coordinates must be integers, got {cell!r}")
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"{name} {cell!r} is out of bounds for a {self._rows}x{self._cols} grid"
            )

    def _neighbors(self, cell: Cell) -> Iterator[Cell]:
        """Yield the in-bounds 4-directional neighbors of a cell.

        Args:
            cell: (row, col) cell to look at.

        Yields:
            Adjacent in-bounds (row, col) cells.
        """
        row, col = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self._rows and 0 <= nc < self._cols:
                yield (nr, nc)

    @staticmethod
    def _heuristic(cell: Cell, end: Cell) -> int:
        """Manhattan distance from cell to end.

        Args:
            cell: (row, col) cell.
            end: (row, col) goal cell.

        Returns:
            Manhattan distance between cell and end.
        """
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Cell, Cell], start: Cell, end: Cell
    ) -> List[Cell]:
        """Walk came_from backwards to build the path.

        Args:
            came_from: Map from cell to its predecessor.
            start: Start cell.
            end: Goal cell.

        Returns:
            Path from start to end inclusive.
        """
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_a_star_grid.py`

```python
"""Tests for AStarGrid."""

import pytest

from a_star_grid import AStarGrid


def total_cost(grid, path):
    """Total movement cost along a path (sum of costs of cells entered, excluding start)."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Assert a path connects start to end, steps are 4-directional, cells are in bounds and passable."""
    assert path[0] == start
    assert path[-1] == end
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    for r, c in path:
        assert 0 <= r < len(grid)
        assert 0 <= c < len(grid[0])
        assert grid[r][c] > 0


def test_simple_path_on_uniform_grid():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    # Optimal: Manhattan distance (4 moves), each costing 1
    assert total_cost(grid, path) == 4


def test_path_around_obstacles():
    grid = [[1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]]
    start, end = (0, 0), (0, 3)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    # Optimal detour: down column 0, across the bottom row, back up column 3 (9 cells entered)
    assert total_cost(grid, path) == 9


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [[1, 10, 10, 1],
            [1, 1, 1, 1]]
    start, end = (0, 0), (0, 3)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    # Shorter top route costs 10 + 10 + 1 = 21; the cheaper bottom detour costs 5
    assert total_cost(grid, path) == 5


def test_no_path_exists_when_fully_blocked():
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    astar = AStarGrid(grid)

    # Wall column 1 completely separates start from end
    assert astar.find_path((0, 0), (0, 2)) is None
    # Wall endpoints are rejected regardless of blocking
    assert astar.find_path((0, 1), (0, 2)) is None
    assert astar.find_path((0, 0), (0, 1)) is None


def test_start_equals_end():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

    path = AStarGrid(grid).find_path((1, 1), (1, 1))

    assert path == [(1, 1)]
    assert total_cost(grid, path) == 0


@pytest.mark.parametrize("start, end", [
    ((-1, 0), (1, 1)),   # negative row
    ((0, 3), (1, 1)),    # column out of bounds
    ((0, 0), (3, 0)),    # row out of bounds
    ((2, 2), (2, 2)),    # sanity check helper case is NOT used here
])
def test_invalid_coordinates_raise_value_error(start, end):
    # The fourth case in the parametrize above is intentionally invalid:
    # (2, 2) is valid for a 3x3 grid, so it is removed below; keep only out-of-bounds cases.
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)
```

Wait — I made a mistake in the last test above. Let me fix it to keep only genuinely invalid coordinates:

```python
@pytest.mark.parametrize("start, end", [
    ((-1, 0), (1, 1)),
    ((0, 3), (1, 1)),
    ((0, 0), (3, 0)),
    ((2, 2), (-1, 2)),
])
def test_invalid_coordinates_raise_value_error(start, end):
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)
```

## Design notes

- **Optimality**: Manhattan distance is admissible because every passable cell costs ≥ 1 (each step can't be cheaper than 1), and it's also consistent, so the first time `end` is popped from the heap, its path is guaranteed minimum-cost.
- **Edge-case precedence**: bounds check (`ValueError`) → wall check (`None`) → `start == end` (`[start]`). Note that `start == end` on a wall returns `None`, since walls are impassable.
- **Heap entries** use `(f_score, counter, cell)` so ties never require comparing cells.
- **Expected costs** in the tests are hand-verified optima: test 1 matches the Manhattan lower bound (4); test 2 requires the bottom-row detour (9); test 3 is strictly cheaper via the bottom row (5 vs. 21 for the straight route).

Run with:

```bash
pip install pytest
pytest test_a_star_grid.py -v
```

All 6 tests should pass. One small thing to note: `find_path` accepts lists as well as tuples for coordinates, but the declared type is `Tuple[int, int]`.