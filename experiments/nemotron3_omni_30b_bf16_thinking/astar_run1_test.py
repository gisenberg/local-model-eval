from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import heapq


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    The grid is a list of lists where a value of ``0`` denotes a wall
    (impassable cell) and any positive integer denotes a walkable cell with
    that traversal cost.  Movement is allowed only in the four cardinal
    directions (up, down, left, right).

    The heuristic used is the Manhattan distance, which is admissible because
    the cost of moving from one cell to an adjacent cell is at least ``1``.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Parameters
        ----------
        grid : List[List[int]]
            2‑dimensional grid. ``0`` indicates a wall, any positive integer
            indicates a walkable cell with that traversal cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0

    # --------------------------------------------------------------------- #
    #  Private helpers
    # --------------------------------------------------------------------- #
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if ``pos`` lies inside the grid."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if ``pos`` is out of bounds or a wall (value ``0``)."""
        r, c = pos
        return not self._in_bounds(pos) or self.grid[r][c] == 0

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from ``start`` to ``end`` using A*.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting cell coordinates (row, column).
        end : Tuple[int, int]
            Destination cell coordinates (row, column).

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from ``start`` to ``end`` (inclusive) if a
            path exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` are out of bounds or correspond to a wall.
        """
        # ---- validation -------------------------------------------------
        for pos, name in ((start, "start"), (end, "end")):
            if not self._in_bounds(pos):
                raise ValueError(f"{name} is out of bounds: {pos}")
            if self._is_wall(pos):
                raise ValueError(f"{name} is a wall or out of bounds: {pos}")

        if start == end:
            return [start]

        # ---- A* algorithm ------------------------------------------------
        # 4‑directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(pos: Tuple[int, int]) -> int:
            """Manhattan distance to ``end``."""
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

        open_heap: List[Tuple[int, int, int, int]] = []          # (f, g, row, col)
        heapq.heappush(open_heap, (heuristic(start), 0, start[0], start[1]))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)
            current = (r, c)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in directions:
                nbr = (r + dr, c + dc)
                if self._is_wall(nbr):
                    continue

                tentative_g = g + self.grid[nbr[0]][nbr[1]]
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f = tentative_g + heuristic(nbr)
                    heapq.heappush(open_heap, (f, tentative_g, nbr[0], nbr[1]))

        # no path found
        return None

import pytest


def test_start_end_same():
    """Start and end are the same cell – should return a list containing that cell."""
    grid = [[1, 2], [3, 4]]
    finder = AStarGrid(grid)
    assert finder.find_path((1, 1), (1, 1)) == [(1, 1)]


def test_invalid_start_wall():
    """Start cell is a wall – must raise ValueError."""
    grid = [[0, 1], [1, 1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))


def test_invalid_end_wall():
    """End cell is a wall – must raise ValueError."""
    grid = [[1, 1], [1, 0]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))


def test_out_of_bounds_start():
    """Start coordinates outside the grid – must raise ValueError."""
    grid = [[1, 2], [3, 4]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((-1, 0), (0, 0))


def test_simple_grid():
    """Simple 3×3 grid with uniform cost – path should be the Manhattan shortest."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    # shortest Manhattan distance is 4 steps, so path length (including start) is 5
    assert isinstance(path, list)
    assert len(path) == 5
    # verify that the path is strictly moving right/down
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert path == expected


def test_weighted_grid_and_no_path():
    """
    Grid with a high‑cost cell and a wall that blocks a direct route.
    The algorithm must pick the cheaper detour and return ``None`` when no
    path exists.
    """
    grid = [
        [1, 5, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    finder = AStarGrid(grid)

    # Weighted optimal path: (0,0) → (1,0) → (2,0) → (2,1) → (2,2)
    expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert finder.find_path((0, 0), (2, 2)) == expected_path

    # Grid where start and end are separated by walls – no path should be found.
    no_path_grid = [
        [1, 0],
        [0, 1],
    ]
    finder2 = AStarGrid(no_path_grid)
    assert finder2.find_path((0, 0), (1, 1)) is None