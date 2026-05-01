"""A* path‑finding implementation for a weighted 2‑D grid.

The grid is represented as a list of list of integers:

* ``0`` – wall (impassable)
* any positive integer – traversal cost of the cell

The algorithm uses a 4‑directional move set, a Manhattan distance
heuristic, and ``heapq`` for the open set.  The returned path is a list of
coordinates ``(row, col)`` from *start* to *end* (inclusive).  If no path
exists ``None`` is returned.

The implementation raises ``ValueError`` for:
* start or end outside the grid,
* start or end being a wall,
* start == end (the trivial path ``[start]`` is returned instead).
"""

from __future__ import annotations

from heapq import heappush, heappop
from typing import List, Tuple, Optional, Dict, Set


class AStarGrid:
    """A* path‑finder for a weighted 2‑D grid."""

    # 4‑directional movement (up, down, left, right)
    _DIRS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Parameters
        ----------
        grid : List[List[int]]
            2‑D grid where ``0`` denotes a wall and positive integers denote
            the cost to step onto the cell.
        """
        if not grid or not grid[0]:
            raise ValueError("grid must be non‑empty")
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
        """Manhattan distance between two cells."""
        return abs(r1 - r2) + abs(c1 - c2)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal (lowest‑cost) path on the grid.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting cell coordinates ``(row, col)``.
        end : Tuple[int, int]
            Destination cell coordinates ``(row, col)``.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from *start* to *end* (inclusive) if a path
            exists, otherwise ``None``.

        Raises
        ------
        ValueError
            * ``start`` or ``end`` is out of bounds,
            * ``start`` or ``end`` is a wall (value ``0``).
        """
        # ---- validation --------------------------------------------------- #
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("start cell is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("end cell is out of bounds")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("start cell is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("end cell is a wall")

        if start == end:
            # Trivial case – the path consists of the single start cell.
            return [start]

        # ---- A* main loop ------------------------------------------------ #
        open_set: List[Tuple[int, int, int, int]] = []  # (f, g, row, col)
        heappush(open_set, (self._manhattan(*start, *end), 0, start[0], start[1]))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        closed: Set[Tuple[int, int]] = set()

        while open_set:
            f, g, r, c = heappop(open_set)

            if (r, c) == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = []
                current = (r, c)
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            if (r, c) in closed:
                continue
            closed.add((r, c))

            for dr, dc in self._DIRS:
                nr, nc = r + dr, c + dc
                # bounds & wall check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:  # wall
                    continue

                tentative_g = g + self.grid[nr][nc]  # cost to enter neighbor
                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + self._manhattan(nr, nc, *end)
                    came_from[(nr, nc)] = (r, c)
                    heappush(open_set, (f, tentative_g, nr, nc))

        # No path found
        return None

"""pytest test suite for ``AStarGrid``."""

from typing import List, Tuple

import pytest



def test_start_equal_end():
    """The trivial case where start and end are the same cell."""
    grid = [
        [1, 2],
        [3, 4],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_simple_grid():
    """A 3×3 grid with uniform cost – any shortest Manhattan path is optimal."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))

    assert path is not None
    # path must contain start and end, and have exactly 5 cells (4 moves)
    assert len(path) == 5
    assert path[-1] == (2, 2)

    # each step must be to a neighboring cell
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    # no wall cells may appear
    for r, c in path:
        assert grid[r][c] != 0

    # total cost (sum of cell weights, excluding start) must be 4
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4


def test_with_wall():
    """A wall blocks the direct route; the algorithm must detour."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))

    assert path is not None
    assert len(path) == 5
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    for r, c in path:
        assert grid[r][c] != 0
    assert path[-1] == (2, 2)

    # cost = 4 (Manhattan distance) because all traversed cells cost 1
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4


def test_out_of_bounds_start():
    """Start coordinate outside the grid raises ValueError."""
    grid = [
        [1, 2],
        [3, 4],
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="start cell is out of bounds"):
        astar.find_path((-1, 0), (0, 1))


def test_start_is_wall():
    """Start cell being a wall raises ValueError."""
    grid = [
        [0, 2],
        [3, 4],
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="start cell is a wall"):
        astar.find_path((0, 0), (1, 1))


def test_weighted_grid():
    """Grid with varying cell weights – algorithm must pick the cheapest route."""
    grid = [
        [1, 2, 1],
        [10, 1, 2],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))

    assert path is not None
    assert len(path) == 5
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    for r, c in path:
        assert grid[r][c] != 0

    # The optimal path is (0,0)->(0,1)->(1,1)->(2,1)->(2,2) with cost 5
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 5