from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    * ``0`` denotes an impassable wall.
    * Any positive integer denotes a walkable cell; the integer is the
      cost of stepping onto that cell.
    * Movement is allowed only in the four cardinal directions
      (up, down, left, right).
    * The heuristic used is the Manhattan distance multiplied by the
      smallest positive cell weight in the grid – admissible and consistent.

    The public API consists of a single method::

        find_path(start, end) -> Optional[List[Tuple[int, int]]]

    where ``start`` and ``end`` are ``(row, col)`` tuples.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Parameters
        ----------
        grid : List[List[int]]
            2‑D list representing the map. ``0`` cells are walls,
            positive integers are walkable cells with the given cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non‑empty")

        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

        # Minimum positive weight among walkable cells – used for an admissible heuristic.
        self.min_weight: int = min(
            (cell for row in grid for cell in row if cell > 0),
            default=1,
        )

        # The target cell will be stored temporarily for the heuristic.
        self.end: Tuple[int, int] | None = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the optimal path from ``start`` to ``end`` using A*.

        Parameters
        ----------
        start : Tuple[int, int]
            Starting cell coordinates (row, col).
        end : Tuple[int, int]
            Destination cell coordinates (row, col).

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from ``start`` to ``end`` (inclusive) if a
            path exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` are out of bounds or refer to a wall.
        """
        # -----------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------
        for point, name in ((start, "start"), (end, "end")):
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"{name} coordinates out of bounds")
            if self.grid[r][c] == 0:
                raise ValueError(f"{name} is a wall (value 0)")

        if start == end:
            return [start]

        # Store end for heuristic usage
        self.end = end

        # -----------------------------------------------------------------
        # A* initialisation
        # -----------------------------------------------------------------
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_set: List[Tuple[int, int, int, int]] = []  # (f, g, row, col)
        heapq.heappush(open_set, (self._heuristic(start), self._cost(start), start[0], start[1]))

        g_score: Dict[Tuple[int, int], int] = {start: self._cost(start)}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # -----------------------------------------------------------------
        # Main loop
        # -----------------------------------------------------------------
        while open_set:
            _, g_current, r, c = heapq.heappop(open_set)
            current = (r, c)

            if current == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Wall check
                if self.grid[nr][nc] == 0:
                    continue

                tentative_g = g_score[current] + self.grid[nr][nc]  # cost to enter neighbor

                if tentative_g < g_score.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc))
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        # No path found
        return None

    # --------------------------------------------------------------------- #
    # Helper methods (private)
    # --------------------------------------------------------------------- #
    def _cost(self, cell: Tuple[int, int]) -> int:
        """Return the cost of stepping onto *cell* (its grid value)."""
        r, c = cell
        return self.grid[r][c]

    def _heuristic(self, cell: Tuple[int, int]) -> int:
        """Manhattan distance * minimum positive cell weight (admissible)."""
        r, c = cell
        return (abs(r - self.end[0]) + abs(c - self.end[1])) * self.min_weight

import pytest


def test_start_end_same():
    """Start and end are the same cell – should return a list containing that cell."""
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_no_path():
    """Grid with walls that prevent any route from start to end."""
    grid = [
        [1, 0],
        [0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None


def test_out_of_bounds():
    """Coordinates outside the grid must raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))


def test_wall_start():
    """Start cell is a wall – should raise ValueError."""
    grid = [
        [0, 1],
        [1, 1],
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))


def test_weighted_optimal():
    """
    Grid where the cheapest route avoids a high‑weight cell.
    The optimal path is:
        (0,0) → (1,0) → (1,1) → (1,2) → (0,2)
    """
    grid = [
        [1, 4, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert astar.find_path((0, 0), (0, 2)) == expected


def test_simple_path():
    """Simple grid with uniform cost – any optimal Manhattan‑optimal path is acceptable."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # The algorithm will choose the straight‑right route because of tie‑breaking.
    assert path == [(0, 0), (0, 1), (0, 2)]