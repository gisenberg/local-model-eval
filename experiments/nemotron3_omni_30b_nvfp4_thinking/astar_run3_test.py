"""A* path‑finding implementation for a weighted 2‑D grid.

The grid is represented as a list of lists where a cell value of ``0`` means
*wall* (impassable) and any positive integer is the cost of traversing that
cell.  The algorithm uses a 4‑directional move set, a Manhattan distance
heuristic (scaled by the smallest traversable cost) and ``heapq`` for the
open set.

The public API consists of the ``AStarGrid`` class with a single method:

    find_path(start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]

which returns the optimal path (including start and end) or ``None`` if no
path exists.
"""

from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    Parameters
    ----------
    grid : List[List[int]]
        2‑D grid where ``0`` denotes a wall and any positive integer denotes
        the traversal cost of the cell.

    Notes
    -----
    * Only the four cardinal directions (up, down, left, right) are allowed.
    * The heuristic is the Manhattan distance multiplied by the smallest
      positive cell cost in the grid, which makes it admissible for any
      positive cell weight.
    * ``start`` and ``end`` must be inside the grid and must not be walls.
    * If ``start`` equals ``end`` the method returns ``[start]``.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Create a new ``AStarGrid`` instance."""
        if not grid or not grid[0]:
            raise ValueError("Grid must be non‑empty.")
        self.grid: List[List[int]] = grid
        self.height: int = len(grid)
        self.width: int = len(grid[0])

        # smallest positive cost (used for a consistent heuristic)
        self.min_cost: int = min(
            (cell for row in grid for cell in row if cell > 0), default=1
        )

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _is_valid(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` lies inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_wall(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is a wall (value == 0)."""
        return self.grid[y][x] == 0

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --------------------------------------------------------------------- #
    # Core A* algorithm
    # --------------------------------------------------------------------- #
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the optimal path from ``start`` to ``end`` using A*.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting cell coordinates ``(x, y)``.
        end : Tuple[int, int]
            Destination cell coordinates ``(x, y)``.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates describing the optimal path, or ``None`` if no
            path exists.

        Raises
        ------
        ValueError
            * If ``start`` or ``end`` are out of bounds.
            * If ``start`` or ``end`` are walls.
        """
        # ---- validation --------------------------------------------------- #
        if not self._is_valid(*start):
            raise ValueError(f"Start {start} is out of bounds.")
        if not self._is_valid(*end):
            raise ValueError(f"End {end} is out of bounds.")
        if self._is_wall(*start):
            raise ValueError(f"Start {start} is a wall.")
        if self._is_wall(*end):
            raise ValueError(f"End {end} is a wall.")

        # Trivial case
        if start == end:
            return [start]

        # ---- A* initialisation ------------------------------------------ #
        open_set: List[Tuple[int, int, int]] = []  # (f_score, g_score, node)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        closed_set: set[Tuple[int, int]] = set()

        # 4‑directional moves
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_set:
            f_current, g_current, current = heapq.heappop(open_set)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)

            cx, cy = current
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if not self._is_valid(nx, ny):
                    continue
                if self._is_wall(nx, ny):
                    continue

                neighbor = (nx, ny)
                tentative_g = g_current + self.grid[ny][nx]

                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                # If this path to neighbor is better, record it and push to heap
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current

        # No path found
        return None

"""Pytest suite for the AStarGrid implementation."""

import pytest


def test_start_end_same():
    """Start and end are the same cell – should return a list containing that cell."""
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_start_is_wall():
    """Starting on a wall must raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start"):
        astar.find_path((0, 0), (1, 1))


def test_end_is_wall():
    """Ending on a wall must raise ValueError."""
    grid = [[1, 1], [1, 0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="End"):
        astar.find_path((0, 0), (1, 1))


def test_out_of_bounds():
    """Coordinates outside the grid must raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start"):
        astar.find_path((2, 0), (0, 0))   # x is out of bounds
    with pytest.raises(ValueError, match="End"):
        astar.find_path((0, 0), (0, 2))   # y is out of bounds


def test_simple_path():
    """A simple 3×3 grid with uniform cost – the path should be optimal."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # The algorithm should produce a shortest Manhattan path (4 moves → 5 nodes)
    assert len(path) == 5
    # Verify that the path is monotonic (only moves right/down)
    assert all(
        (path[i][0] <= path[i + 1][0] and path[i][1] <= path[i + 1][1])
        for i in range(len(path) - 1)
    )
    # The exact shape may vary, but it must start and end correctly
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)


def test_weighted_path():
    """Path must respect cell weights and avoid higher‑cost cells when possible."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [100, 1, 1],
    ]
    astar = AStarGrid(grid)
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert astar.find_path((0, 0), (2, 2)) == expected