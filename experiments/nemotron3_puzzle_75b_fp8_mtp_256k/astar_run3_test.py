from __future__ import annotations

import heapq
from typing import List, Optional, Tuple, Set

class AStarGrid:
    """
    A* path‑finding on a weighted 2‑D grid.

    Parameters
    ----------
    grid : List[List[int]]
        2‑D array where each cell holds the movement cost to *enter* that cell.
        A value of ``0`` denotes an impassable wall.

    Notes
    -----
    * Movement is allowed in the four cardinal directions (up, down, left, right).
    * The heuristic used is Manhattan distance.
    * ``find_path`` returns the optimal path as a list of ``(row, col)`` tuples
      inclusive of start and end, or ``None`` when no path exists.
    * ``ValueError`` is raised if ``start`` or ``end`` are out of bounds or lie
      on a wall.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not all(grid):
            raise ValueError("grid must be a non‑empty rectangular 2‑D list")
        self._grid = grid
        self._rows = len(grid)
        self._cols = len(grid[0])

        # ensure rectangular shape
        for r in self._grid:
            if len(r) != self._cols:
                raise ValueError("all rows in grid must have the same length")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the optimal path from ``start`` to ``end`` using A*.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates (row, col) from start to end inclusive,
            or ``None`` if no path exists.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` is outside the grid or on a wall.
        """
        self._validate_point(start, "start")
        self._validate_point(end, "end")

        if start == end:
            return [start]

        # Priority queue entries: (f_score, g_score, position)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                # cost to step onto the neighbor cell
                tentative_g = current_g + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # open set exhausted → no path
        return None

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _validate_point(self, pt: Tuple[int, int], name: str) -> None:
        """Raise ValueError if ``pt`` is out of bounds or on a wall."""
        if not (0 <= pt[0] < self._rows and 0 <= pt[1] < self._cols):
            raise ValueError(f"{name} {pt} is out of grid bounds")
        if self._grid[pt[0]][pt[1]] == 0:
            raise ValueError(f"{name} {pt} is a wall (value 0)")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pt: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid 4‑directional neighbours that are inside the grid."""
        r, c = pt
        candidates = [
            (r - 1, c),  # up
            (r + 1, c),  # down
            (r, c - 1),  # left
            (r, c + 1),  # right
        ]
        return [
            (nr, nc)
            for nr, nc in candidates
            if 0 <= nr < self._rows and 0 <= nc < self._cols
        ]

    def _reconstruct_path(
        self,
        came_from: dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Walk backwards from ``current`` to the start using ``came_from``."""
        path: List[Tuple[int, int]] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import List, Optional, Tuple


def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_simple_unweighted_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


def test_weighted_prefers_cheaper_route():
    # Cheaper route goes around the high‑cost cell (5)
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    # Optimal path avoids the 5 (cost 1+1+1+1+1 = 5) vs going through 5 (cost 7)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


def test_walls_block_path():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    # No way to cross the vertical wall of zeros
    assert astar.find_path((0, 0), (0, 2)) is None


def test_out_of_bounds_raises():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))


def test_start_or_end_on_wall_raises():
    grid = [
        [1, 0],
        [0, 1],
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (1, 1))  # start is a wall
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 0))  # end is a wall