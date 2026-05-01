"""A* path‑finding on a weighted 2‑D grid.

The grid is a list of lists where a cell value of ``0`` means *wall* (impassable)
and any positive integer is a traversable cell with that movement cost.
Only the four orthogonal directions (up, down, left, right) are allowed.

The heuristic used is the Manhattan distance multiplied by the smallest
traversable cell weight in the grid – this guarantees admissibility
(i.e. it never over‑estimates the true cost).
"""

from __future__ import annotations

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* implementation for a 4‑directional weighted grid.

    Parameters
    ----------
    grid : List[List[int]]
        2‑D list representing the map. ``0`` denotes a wall, any positive
        integer denotes a traversable cell with the given movement cost.

    Methods
    -------
    find_path(start, end) -> Optional[List[Tuple[int, int]]]
        Returns the optimal path from ``start`` to ``end`` as a list of
        ``(row, col)`` tuples, or ``None`` if no path exists.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0

        # pre‑compute the minimal positive weight – needed for an admissible heuristic
        min_weight = float("inf")
        for row in self.grid:
            for cell in row:
                if cell > 0 and cell < min_weight:
                    min_weight = cell
        if min_weight == float("inf"):
            # grid contains only walls – we treat it as an empty map
            self.min_weight = 0
        else:
            self.min_weight = min_weight

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if ``pos`` lies inside the grid."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Return ``True`` if ``pos`` is a wall (value 0)."""
        r, c = pos
        return not self._in_bounds(pos) or self.grid[r][c] == 0

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Yield the walkable neighbours of ``pos`` together with the cost to move
        into each neighbour (the neighbour’s cell weight).
        """
        r, c = pos
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if self._is_wall((nr, nc)):
                continue
            neighbour = (nr, nc)
            cost = self.grid[nr][nc]          # movement cost = cell weight
            yield neighbour, cost

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from ``start`` to ``end`` using A*.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting cell coordinates ``(row, col)``.
        end : Tuple[int, int]
            Destination cell coordinates ``(row, col)``.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            The reconstructed path, inclusive of ``start`` and ``end``.
            Returns ``None`` if no path exists.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` is out of bounds or is a wall.
        """
        # ---- validation ----------------------------------------------------
        if not self._in_bounds(start):
            raise ValueError(f"Start {start} is out of bounds.")
        if not self._in_bounds(end):
            raise ValueError(f"End {end} is out of bounds.")
        if self._is_wall(start):
            raise ValueError(f"Start {start} is a wall.")
        if self._is_wall(end):
            raise ValueError(f"End {end} is a wall.")

        if start == end:
            return [start]

        # ---- A* main loop ---------------------------------------------------
        # priority queue entries are (f_score, tie_breaker, position)
        frontier: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(frontier, (self._heuristic(start, end), 0, start))

        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        tie_breaker = 0

        while frontier:
            current_f, _, current = heapq.heappop(frontier)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, step_cost in self._neighbors(current):
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    tie_breaker += 1
                    heapq.heappush(frontier, (f, tie_breaker, neighbor))

        # no path found
        return None

    # --------------------------------------------------------------------- #
    # Heuristic
    # --------------------------------------------------------------------- #
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Manhattan distance multiplied by the minimal traversable weight.
        This is admissible because each step costs at least ``min_weight``.
        """
        return self.min_weight * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

import pytest
from typing import List, Tuple, Optional


def test_start_end_same():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_start_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start"):
        astar.find_path((0, 0), (1, 1))


def test_end_is_wall():
    grid = [[1, 1], [1, 0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="End"):
        astar.find_path((0, 0), (1, 1))


def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start"):
        astar.find_path((2, 0), (0, 0))   # row out of bounds
    with pytest.raises(ValueError, match="End"):
        astar.find_path((0, 0), (0, 3))   # column out of bounds


def test_optimal_path_with_various_weights():
    """
    Grid where the cheapest route avoids a high‑weight cell.
    The optimal path should be the one with the smallest sum of cell weights.
    """
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)
    # The optimal route is (0,0)->(1,0)->(1,1)->(1,2)->(2,2)
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    assert path == expected

    # Verify that the total cost matches the sum of the cell weights
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 1 + 1 + 1 + 1 + 1  # 5 steps, each weight 1


def test_no_path_exists():
    """
    Grid completely blocked by walls – the algorithm must return ``None``.
    """
    grid = [
        [1, 0],
        [0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None