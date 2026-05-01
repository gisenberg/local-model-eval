"""A* path‑finder for a weighted 2‑D grid.

The grid is a matrix of integers where ``0`` denotes a wall (impassable)
and any positive integer denotes a traversable cell whose weight is the
cost of stepping onto that cell.

The implementation is 4‑directional (up, down, left, right) and uses a
Manhattan distance heuristic multiplied by the smallest traversable
cell weight, which makes it admissible and guarantees optimality.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    Parameters
    ----------
    grid :
        2‑D list of integers. ``0`` means wall, any positive integer is a
        traversable cell whose value is the cost of entering that cell.

    Notes
    -----
    * Only 4‑directional moves are allowed.
    * The heuristic is ``min_weight * Manhattan distance`` where
      ``min_weight`` is the smallest non‑zero cell value in the grid.
      This makes the heuristic admissible and ensures that the returned
      path is optimal.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("Grid must be non‑empty")

        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        # smallest positive cell weight (used for an admissible heuristic)
        weights = [
            cell for row in grid for cell in row if cell != 0
        ]
        self.min_weight: int = min(weights, default=1)

        # sanity check – start/end must be inside the grid and not walls
        if not any(
            grid[x][y] != 0
            for x in range(self.rows)
            for y in range(self.cols)
        ):
            # grid contains only walls → no path is ever possible
            pass

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if *pos* lies inside the grid."""
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols

    def _heuristic(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance multiplied by the smallest traversable weight."""
        dx = abs(pos[0] - self.end[0])
        dy = abs(pos[1] - self.end[1])
        return (dx + dy) * self.min_weight

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[
        List[Tuple[int, int]]
    ]:
        """
        Find the optimal path from *start* to *end* using A*.

        Parameters
        ----------
        start, end :
            Tuple ``(row, col)`` coordinates.

        Returns
        -------
        list[tuple[int, int]] | None
            The reconstructed path (including start and end) if one exists,
            otherwise ``None``.

        Raises
        ------
        ValueError
            * If ``start`` or ``end`` lie outside the grid.
            * If ``start`` or ``end`` is a wall (cell value ``0``).
            * If ``start`` and ``end`` are the same coordinate (returns ``[start]``).
        """
        # ---- basic validation ------------------------------------------------
        for p in (start, end):
            x, y = p
            if not self._in_bounds(p):
                raise ValueError("Start or end coordinate is out of bounds")
            if self.grid[x][y] == 0:
                raise ValueError("Start or end is placed on a wall")

        if start == end:
            return [start]

        # ---- A* main loop ----------------------------------------------------
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_set: List[Tuple[int, int, Tuple[int, int]]] = []  # (f, g, node)
        heapq.heappush(open_set, (self._heuristic(start), 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # keep a reference to the end coordinate for the heuristic closure
        self.end = end

        while open_set:
            _, g_current, current = heapq.heappop(open_set)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if not self._in_bounds(neighbor):
                    continue
                cell_weight = self.grid[nx][ny]
                if cell_weight == 0:  # wall
                    continue

                tentative_g = g_current + cell_weight
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor)
                    # push a new entry; old entries with higher f will be ignored
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        # no path found
        return None

import pytest


def test_start_equals_end():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_simple_grid_no_walls():
    """
    3×3 grid with equal cost (1).  Start (0,0) → end (2,2) has Manhattan distance 4.
    Any optimal path length is 5 cells.
    """
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # 4 moves + start cell
    # verify each step is orthogonal and moves closer to the goal
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_path_around_wall():
    """
    Grid with a single wall that forces the algorithm to go around.
    """
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # shortest possible detour length is 6 cells (5 moves)
    assert len(path) == 6
    # ensure no step lands on a wall
    for p in path:
        assert grid[p[0]][p[1]] != 0


def test_start_out_of_bounds():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))  # (2,2) is out of bounds


def test_end_is_wall():
    grid = [[1, 2], [3, 0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start or end is placed on a wall"):
        astar.find_path((0, 0), (1, 1))


def test_no_possible_path():
    """
    Grid completely split by walls – no path should be found (return None).
    """
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None