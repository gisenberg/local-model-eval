from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import heapq


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    The grid is a list of lists where a cell value of ``0`` means *wall* (impassable)
    and any positive integer is the cost of stepping onto that cell.
    The algorithm uses a 4‑directional move set (up, down, left, right) and a
    Manhattan distance heuristic (admissible because the smallest possible step
    cost is 1).

    Parameters
    ----------
    grid :
        2‑D list of integers.  All rows must have the same length.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid:
            raise ValueError("Grid must not be empty")

        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        # Verify rectangular shape
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("All rows in the grid must have the same length")

        # Minimum traversable cost (positive).  Used for an admissible heuristic.
        self.min_cost: int = min(
            (cell for row in grid for cell in row if cell > 0), default=1
        )

        # 4‑directional movement vectors (x, y)
        self.directions: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if *pos* is inside the grid and not a wall."""
        x, y = pos
        return (
            0 <= x < self.cols
            and 0 <= y < self.rows
            and self.grid[y][x] != 0
        )

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance multiplied by the smallest traversable cost."""
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * self.min_cost

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the optimal (lowest‑cost) path from *start* to *end*.

        Returns
        -------
        list[tuple[int, int]] | None
            The path as a list of coordinates, ordered from *start* to *end*,
            or ``None`` if no path exists.

        Raises
        ------
        ValueError
            If *start* or *end* are out of bounds or are walls (value ``0``).
        """
        # ---- validation --------------------------------------------------- #
        if not self._is_valid(start):
            raise ValueError("Start position out of bounds or is a wall")
        if not self._is_valid(end):
            raise ValueError("End position out of bounds or is a wall")
        if start == end:
            return [start]

        # ---- A* main loop ------------------------------------------------ #
        open_set: List[Tuple[int, int, int, int]] = []  # (f, g, x, y)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start[0], start[1]))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, g, x, y = heapq.heappop(open_set)
            current = (x, y)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [(x, y)]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if not self._is_valid(neighbor):
                    continue

                tentative_g = g + self.grid[ny][nx]

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f, tentative_g, nx, ny))
                    came_from[neighbor] = (x, y)

        # No path found
        return None

import pytest


def test_start_equal_end():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_start_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))


def test_end_is_wall():
    grid = [[1, 1], [1, 0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))


def test_out_of_bounds_start():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (0, 1))


def test_out_of_bounds_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))


def test_weighted_path():
    """
    Grid where the central cell has a higher cost (2). The optimal path
    avoids that cell.
    """
    grid = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert astar.find_path((0, 0), (2, 2)) == expected


def test_no_path():
    """
    Walls completely separate start from end, so no route exists.
    """
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None